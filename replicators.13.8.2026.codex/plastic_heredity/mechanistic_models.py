"""Frozen transforms and partially penalized models for attribution tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA

from .mechanistic_features import FEATURE_NAMES, MechanisticRawFeatures

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class BlockTransform:
    name: str
    source_names: tuple[str, ...]
    kept_indices: NDArray[np.int64]
    kept_names: tuple[str, ...]
    dropped: dict[str, str]
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    pca_mean: NDArray[np.float64] | None = None
    pca_components: NDArray[np.float64] | None = None
    pca_explained_variance: NDArray[np.float64] | None = None
    residual_coefficient: NDArray[np.float64] | None = None

    @property
    def output_features(self) -> int:
        if self.pca_components is not None:
            return int(self.pca_components.shape[0])
        return int(self.kept_indices.size)

    def transform(
        self, values: FloatMatrix, residual_base: FloatMatrix | None = None
    ) -> FloatMatrix:
        selected = np.asarray(values, dtype=np.float64)[:, self.kept_indices]
        transformed = (selected - self.mean) / self.scale
        if self.residual_coefficient is not None:
            if residual_base is None:
                raise ValueError(f"{self.name} requires a residualization base")
            augmented = np.column_stack(
                (np.ones(residual_base.shape[0], dtype=np.float64), residual_base)
            )
            transformed = transformed - augmented @ self.residual_coefficient
        if self.pca_components is not None:
            if self.pca_mean is None:
                raise AssertionError("PCA mean missing")
            transformed = (transformed - self.pca_mean) @ self.pca_components.T
        return np.asarray(transformed, dtype=np.float64)


@dataclass(frozen=True)
class RegisteredModel:
    name: str
    blocks: tuple[str, ...]
    coefficient: NDArray[np.float64]
    intercept: float
    penalty_mask: NDArray[np.bool_]
    objective: float
    gradient_max_abs: float
    iterations: int

    def predict(self, block_values: dict[str, FloatMatrix]) -> NDArray[np.float64]:
        design = np.column_stack([block_values[name] for name in self.blocks])
        return expit(self.intercept + design @ self.coefficient)


@dataclass(frozen=True)
class CandidateRegistry:
    candidate: str
    transforms: dict[str, BlockTransform]
    models: dict[str, RegisteredModel]


MODEL_DEFINITIONS: dict[str, tuple[tuple[str, ...], frozenset[str]]] = {
    "h8": (("h8",), frozenset()),
    "h10": (("h10",), frozenset()),
    "state_only": (("state",), frozenset(("state",))),
    "beta_only": (("beta",), frozenset(("beta",))),
    "h10_state": (("h10", "state"), frozenset(("state",))),
    "h10_beta": (("h10", "beta"), frozenset(("beta",))),
    "h10_state_beta": (
        ("h10", "state", "beta"),
        frozenset(("state", "beta")),
    ),
    "h10_state_beta_interaction": (
        ("h10", "state", "beta", "interaction"),
        frozenset(("state", "beta", "interaction")),
    ),
    "h10_duplicate_corrected": (
        ("h10", "duplicate"),
        frozenset(("duplicate",)),
    ),
    "ridge_h10": (("h10",), frozenset(("h10",))),
    "ridge_h10_duplicate": (
        ("h10", "duplicate"),
        frozenset(("h10", "duplicate")),
    ),
}


def _unique_columns(
    values: FloatMatrix,
    names: tuple[str, ...],
    tolerance: float = 1e-12,
) -> tuple[NDArray[np.int64], dict[str, str]]:
    values = np.asarray(values, dtype=np.float64)
    standard_deviation = values.std(axis=0)
    candidates = [
        index for index, value in enumerate(standard_deviation) if value > tolerance
    ]
    dropped = {
        names[index]: "zero_variance"
        for index, value in enumerate(standard_deviation)
        if value <= tolerance
    }
    kept: list[int] = []
    standardized: list[NDArray[np.float64]] = []
    for index in candidates:
        column = (values[:, index] - values[:, index].mean()) / standard_deviation[index]
        duplicate_of: str | None = None
        for previous_index, previous in zip(kept, standardized):
            if np.allclose(column, previous, rtol=0.0, atol=tolerance):
                duplicate_of = names[previous_index]
                break
            if np.allclose(column, -previous, rtol=0.0, atol=tolerance):
                duplicate_of = f"negative_of:{names[previous_index]}"
                break
        if duplicate_of is None:
            kept.append(index)
            standardized.append(column)
        else:
            dropped[names[index]] = f"affine_duplicate_of:{duplicate_of}"
    if not kept:
        raise ValueError("feature block has no nonconstant unique columns")
    return np.asarray(kept, dtype=np.int64), dropped


def fit_block_transform(
    name: str,
    values: FloatMatrix,
    names: tuple[str, ...],
    pca_components: int | None,
) -> BlockTransform:
    kept, dropped = _unique_columns(values, names)
    selected = np.asarray(values, dtype=np.float64)[:, kept]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    standardized = (selected - mean) / scale
    pca_mean = None
    components = None
    explained = None
    if pca_components is not None:
        count = min(pca_components, standardized.shape[0], standardized.shape[1])
        pca = PCA(n_components=count, svd_solver="full").fit(standardized)
        pca_mean = pca.mean_.copy()
        components = pca.components_.copy()
        explained = pca.explained_variance_.copy()
    return BlockTransform(
        name=name,
        source_names=names,
        kept_indices=kept,
        kept_names=tuple(names[index] for index in kept),
        dropped=dropped,
        mean=mean,
        scale=scale,
        pca_mean=pca_mean,
        pca_components=components,
        pca_explained_variance=explained,
    )


def fit_interaction_transform(
    values: FloatMatrix,
    residual_base: FloatMatrix,
    pca_components: int,
) -> BlockTransform:
    initial = fit_block_transform(
        "interaction", values, FEATURE_NAMES["interaction"], pca_components=None
    )
    standardized = initial.transform(values)
    augmented = np.column_stack(
        (np.ones(residual_base.shape[0], dtype=np.float64), residual_base)
    )
    residual_coefficient = np.linalg.lstsq(
        augmented, standardized, rcond=None
    )[0]
    residual = standardized - augmented @ residual_coefficient
    count = min(pca_components, residual.shape[0], residual.shape[1])
    pca = PCA(n_components=count, svd_solver="full").fit(residual)
    return BlockTransform(
        name=initial.name,
        source_names=initial.source_names,
        kept_indices=initial.kept_indices,
        kept_names=initial.kept_names,
        dropped=initial.dropped,
        mean=initial.mean,
        scale=initial.scale,
        pca_mean=pca.mean_.copy(),
        pca_components=pca.components_.copy(),
        pca_explained_variance=pca.explained_variance_.copy(),
        residual_coefficient=residual_coefficient,
    )


def transform_blocks(
    transforms: dict[str, BlockTransform], raw: MechanisticRawFeatures
) -> dict[str, FloatMatrix]:
    blocks = {
        "h8": transforms["h8"].transform(raw.h8),
        "h10": transforms["h10"].transform(raw.h10),
        "state": transforms["state"].transform(raw.state),
        "beta": transforms["beta"].transform(raw.beta),
        "duplicate": transforms["duplicate"].transform(raw.duplicate),
    }
    residual_base = np.column_stack(
        (blocks["h10"], blocks["state"], blocks["beta"])
    )
    blocks["interaction"] = transforms["interaction"].transform(
        raw.interaction, residual_base
    )
    return blocks


def _binomial_objective_and_gradient(
    parameters: NDArray[np.float64],
    design: FloatMatrix,
    successes: NDArray[np.float64],
    trials: NDArray[np.float64],
    penalty_mask: NDArray[np.bool_],
    c: float,
) -> tuple[float, NDArray[np.float64]]:
    intercept = parameters[0]
    coefficient = parameters[1:]
    logits = intercept + design @ coefficient
    loss = float(np.sum(trials * np.logaddexp(0.0, logits) - successes * logits))
    penalized = coefficient[penalty_mask]
    loss += float(0.5 * np.dot(penalized, penalized) / c)
    probability = expit(logits)
    residual = trials * probability - successes
    gradient = np.empty_like(parameters)
    gradient[0] = residual.sum()
    gradient[1:] = design.T @ residual
    gradient[1:][penalty_mask] += coefficient[penalty_mask] / c
    return loss, gradient


def fit_registered_model(
    name: str,
    blocks: tuple[str, ...],
    penalized_blocks: frozenset[str],
    block_values: dict[str, FloatMatrix],
    successes: NDArray[np.float64],
    trials: NDArray[np.float64],
    c: float,
) -> RegisteredModel:
    design_parts = [block_values[block] for block in blocks]
    design = np.column_stack(design_parts)
    penalty_mask = np.concatenate(
        [
            np.full(part.shape[1], block in penalized_blocks, dtype=bool)
            for block, part in zip(blocks, design_parts)
        ]
    )
    prior = float(np.clip(successes.sum() / trials.sum(), 1e-8, 1.0 - 1e-8))
    initial = np.zeros(design.shape[1] + 1, dtype=np.float64)
    initial[0] = np.log(prior / (1.0 - prior))

    def objective(parameters: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        return _binomial_objective_and_gradient(
            parameters, design, successes, trials, penalty_mask, c
        )

    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5_000, "ftol": 1e-13, "gtol": 1e-8, "maxls": 50},
    )
    value, gradient = objective(fitted.x)
    if not fitted.success and np.max(np.abs(gradient)) > 1e-5:
        raise RuntimeError(f"model {name} failed to converge: {fitted.message}")
    return RegisteredModel(
        name=name,
        blocks=blocks,
        coefficient=fitted.x[1:].copy(),
        intercept=float(fitted.x[0]),
        penalty_mask=penalty_mask,
        objective=float(value),
        gradient_max_abs=float(np.max(np.abs(gradient))),
        iterations=int(fitted.nit),
    )


def fit_candidate_registry(
    candidate: str,
    raw: MechanisticRawFeatures,
    branch_labels: NDArray[np.int8],
    pca_components: int = 12,
    c: float = 0.1,
) -> CandidateRegistry:
    transforms = {
        "h8": fit_block_transform(
            "h8", raw.h8, FEATURE_NAMES["h8"], pca_components=None
        ),
        "h10": fit_block_transform(
            "h10", raw.h10, FEATURE_NAMES["h10"], pca_components=None
        ),
        "state": fit_block_transform(
            "state", raw.state, FEATURE_NAMES["state"], pca_components
        ),
        "beta": fit_block_transform(
            "beta", raw.beta, FEATURE_NAMES["beta"], pca_components
        ),
        "duplicate": fit_block_transform(
            "duplicate",
            raw.duplicate,
            FEATURE_NAMES["duplicate"],
            pca_components=None,
        ),
    }
    preliminary = {
        name: transform.transform(getattr(raw, name))
        for name, transform in transforms.items()
        if name != "duplicate"
    }
    residual_base = np.column_stack(
        (preliminary["h10"], preliminary["state"], preliminary["beta"])
    )
    transforms["interaction"] = fit_interaction_transform(
        raw.interaction, residual_base, pca_components
    )
    blocks = transform_blocks(transforms, raw)
    successes = np.asarray(branch_labels.sum(axis=1), dtype=np.float64)
    trials = np.full(branch_labels.shape[0], branch_labels.shape[1], dtype=np.float64)
    models = {
        name: fit_registered_model(
            name,
            definition[0],
            definition[1],
            blocks,
            successes,
            trials,
            c,
        )
        for name, definition in MODEL_DEFINITIONS.items()
    }
    return CandidateRegistry(candidate=candidate, transforms=transforms, models=models)


def predict_candidate_registry(
    registry: CandidateRegistry, raw: MechanisticRawFeatures
) -> dict[str, NDArray[np.float64]]:
    blocks = transform_blocks(registry.transforms, raw)
    return {
        name: model.predict(blocks) for name, model in registry.models.items()
    }


def _transform_metadata(transform: BlockTransform) -> dict[str, Any]:
    return {
        "name": transform.name,
        "source_names": transform.source_names,
        "kept_names": transform.kept_names,
        "dropped": transform.dropped,
        "output_features": transform.output_features,
        "uses_pca": transform.pca_components is not None,
        "residualized": transform.residual_coefficient is not None,
    }


def save_registries(
    archive_path: Path,
    contract_path: Path,
    registries: dict[str, CandidateRegistry],
) -> None:
    arrays: dict[str, NDArray] = {}
    metadata: dict[str, Any] = {
        "format": "mechanistic-ablation-registry-v1",
        "model_definitions": {
            name: {
                "blocks": definition[0],
                "penalized_blocks": sorted(definition[1]),
            }
            for name, definition in MODEL_DEFINITIONS.items()
        },
        "candidates": {},
    }
    for candidate, registry in registries.items():
        candidate_prefix = f"c{candidate}"
        candidate_meta: dict[str, Any] = {"transforms": {}, "models": {}}
        for name, transform in registry.transforms.items():
            prefix = f"{candidate_prefix}__block__{name}"
            arrays[f"{prefix}__kept_indices"] = transform.kept_indices
            arrays[f"{prefix}__mean"] = transform.mean
            arrays[f"{prefix}__scale"] = transform.scale
            if transform.pca_mean is not None:
                arrays[f"{prefix}__pca_mean"] = transform.pca_mean
                arrays[f"{prefix}__pca_components"] = transform.pca_components
                arrays[f"{prefix}__pca_explained_variance"] = (
                    transform.pca_explained_variance
                )
            if transform.residual_coefficient is not None:
                arrays[f"{prefix}__residual_coefficient"] = (
                    transform.residual_coefficient
                )
            candidate_meta["transforms"][name] = _transform_metadata(transform)
        for name, model in registry.models.items():
            prefix = f"{candidate_prefix}__model__{name}"
            arrays[f"{prefix}__coefficient"] = model.coefficient
            arrays[f"{prefix}__intercept"] = np.asarray([model.intercept])
            arrays[f"{prefix}__penalty_mask"] = model.penalty_mask
            candidate_meta["models"][name] = {
                "blocks": model.blocks,
                "objective": model.objective,
                "gradient_max_abs": model.gradient_max_abs,
                "iterations": model.iterations,
                "features": int(model.coefficient.size),
            }
        metadata["candidates"][candidate] = candidate_meta
    np.savez_compressed(archive_path, **arrays)
    contract_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_registries(
    archive_path: Path, contract_path: Path
) -> dict[str, CandidateRegistry]:
    metadata = json.loads(contract_path.read_text(encoding="utf-8"))
    registries: dict[str, CandidateRegistry] = {}
    with np.load(archive_path) as arrays:
        for candidate, candidate_meta in metadata["candidates"].items():
            candidate_prefix = f"c{candidate}"
            transforms: dict[str, BlockTransform] = {}
            for name, item in candidate_meta["transforms"].items():
                prefix = f"{candidate_prefix}__block__{name}"

                def optional(suffix: str) -> NDArray[np.float64] | None:
                    key = f"{prefix}__{suffix}"
                    return arrays[key].copy() if key in arrays else None

                transforms[name] = BlockTransform(
                    name=name,
                    source_names=tuple(item["source_names"]),
                    kept_indices=arrays[f"{prefix}__kept_indices"].copy(),
                    kept_names=tuple(item["kept_names"]),
                    dropped=dict(item["dropped"]),
                    mean=arrays[f"{prefix}__mean"].copy(),
                    scale=arrays[f"{prefix}__scale"].copy(),
                    pca_mean=optional("pca_mean"),
                    pca_components=optional("pca_components"),
                    pca_explained_variance=optional("pca_explained_variance"),
                    residual_coefficient=optional("residual_coefficient"),
                )
            models: dict[str, RegisteredModel] = {}
            for name, item in candidate_meta["models"].items():
                prefix = f"{candidate_prefix}__model__{name}"
                models[name] = RegisteredModel(
                    name=name,
                    blocks=tuple(item["blocks"]),
                    coefficient=arrays[f"{prefix}__coefficient"].copy(),
                    intercept=float(arrays[f"{prefix}__intercept"][0]),
                    penalty_mask=arrays[f"{prefix}__penalty_mask"].copy(),
                    objective=float(item["objective"]),
                    gradient_max_abs=float(item["gradient_max_abs"]),
                    iterations=int(item["iterations"]),
                )
            registries[candidate] = CandidateRegistry(
                candidate=candidate, transforms=transforms, models=models
            )
    return registries
