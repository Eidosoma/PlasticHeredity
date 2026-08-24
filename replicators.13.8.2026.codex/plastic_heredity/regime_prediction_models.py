"""Frozen model menu for the strict-regime prediction campaign.

All validation splits operate on whole catalytic matrices.  Linear additions
are ridge-penalized offsets to an unpenalized h10 baseline.  The sole nonlinear
family is deliberately bounded by a small, registered hyperparameter grid.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier

from .mechanistic_v2_models import (
    RIDGE_LAMBDAS,
    LinearFit,
    _binomial_loss,
    _unique_columns,
    fit_linear,
)
from .regime_prediction_features import PREDICTION_FEATURE_NAMES, PredictionRawFeatures

FloatArray = NDArray[np.float64]

MODEL_FAMILIES = (
    "direct_ridge",
    "hurdle",
    "hierarchical_offset",
    "local_dynamics",
    "auxiliary_stack",
    "guarded_nonlinear",
)
MODEL_SIMPLICITY = MODEL_FAMILIES
CV_FOLDS = 5
BOOTSTRAP_SELECTION_REPETITIONS = 4_096
BOOTSTRAP_SELECTION_FRACTION = 0.75

NONLINEAR_GRID = tuple(
    {
        "max_depth": depth,
        "learning_rate": rate,
        "max_iter": iterations,
        "min_samples_leaf": leaf,
        "l2_regularization": penalty,
    }
    for depth in (2, 3)
    for rate in (0.03, 0.08)
    for iterations in (100, 250)
    for leaf in (64, 128)
    for penalty in (1.0, 10.0)
)


def matrix_folds(
    matrix_ids: NDArray[np.int64], folds: int = CV_FOLDS
) -> NDArray[np.int64]:
    """Assign complete matrices to deterministic balanced folds."""

    values = np.asarray(matrix_ids, dtype=np.int64)
    unique = np.unique(values)
    if unique.size < 2:
        raise ValueError("matrix-grouped fitting requires at least two matrices")
    actual_folds = min(int(folds), int(unique.size))
    mapping = {int(value): index % actual_folds for index, value in enumerate(unique)}
    return np.asarray([mapping[int(value)] for value in values], dtype=np.int64)


@dataclass(frozen=True)
class StandardizedBlock:
    name: str
    source_names: tuple[str, ...]
    kept_indices: NDArray[np.int64]
    mean: FloatArray
    scale: FloatArray
    dropped: dict[str, str]

    def transform(self, values: FloatArray) -> FloatArray:
        selected = np.asarray(values, dtype=np.float64)[:, self.kept_indices]
        if selected.shape[1] == 0:
            return np.empty((selected.shape[0], 0), dtype=np.float64)
        return (selected - self.mean) / self.scale


def _fit_standardized_block(
    name: str, values: FloatArray, names: tuple[str, ...], tolerance: float = 1e-12
) -> StandardizedBlock:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(names):
        raise ValueError(f"invalid raw block {name}")
    deviation = array.std(axis=0)
    if np.all(deviation <= tolerance):
        indices = np.empty(0, dtype=np.int64)
        dropped = {name: "zero_variance" for name in names}
    else:
        # The upstream routine uses rounded byte signatures, so duplicate
        # discovery is effectively linear in the number of coordinates rather
        # than an all-pairs scan. This keeps the complete beta panel practical
        # while removing its exact affine copies before ridge fitting.
        indices, dropped = _unique_columns(array, names, tolerance)
    selected = array[:, indices]
    return StandardizedBlock(
        name=name,
        source_names=names,
        kept_indices=indices,
        mean=selected.mean(axis=0) if indices.size else np.empty(0),
        scale=selected.std(axis=0) if indices.size else np.empty(0),
        dropped=dropped,
    )


def _blocks(
    raw: PredictionRawFeatures | dict[str, FloatArray],
) -> dict[str, FloatArray]:
    if isinstance(raw, PredictionRawFeatures):
        return {
            name: raw.block(name)
            for name in ("h10", "state", "beta", "interaction", "dynamics")
        }
    return {name: np.asarray(value, dtype=np.float64) for name, value in raw.items()}


def _names(raw_blocks: dict[str, FloatArray]) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for name, values in raw_blocks.items():
        if name in PREDICTION_FEATURE_NAMES and values.shape[1] == len(
            PREDICTION_FEATURE_NAMES[name]
        ):
            output[name] = PREDICTION_FEATURE_NAMES[name]
        elif name == "auxiliary":
            output[name] = ("predicted_first5_logit", "predicted_centroid_logit")
        else:
            output[name] = tuple(
                f"{name}_{index:04d}" for index in range(values.shape[1])
            )
    return output


@dataclass(frozen=True)
class SequentialRidgeModel:
    order: tuple[str, ...]
    transforms: dict[str, StandardizedBlock]
    fits: tuple[LinearFit, ...]
    selected_lambdas: dict[str, float]
    cv_scores: dict[str, dict[str, float]]

    def predict_logit(
        self, raw: PredictionRawFeatures | dict[str, FloatArray]
    ) -> FloatArray:
        raw_blocks = _blocks(raw)
        count = next(iter(raw_blocks.values())).shape[0]
        logits = np.zeros(count, dtype=np.float64)
        for stage, fit in zip(self.order, self.fits):
            logits += fit.correction(
                self.transforms[stage].transform(raw_blocks[stage])
            )
        return logits

    def predict(self, raw: PredictionRawFeatures | dict[str, FloatArray]) -> FloatArray:
        return np.clip(expit(self.predict_logit(raw)), 1e-12, 1.0 - 1e-12)


@dataclass
class _Fold:
    train: NDArray[np.bool_]
    validation: NDArray[np.bool_]
    train_blocks: dict[str, FloatArray]
    validation_blocks: dict[str, FloatArray]
    train_logits: FloatArray
    validation_logits: FloatArray


def fit_sequential_ridge(
    raw: PredictionRawFeatures | dict[str, FloatArray],
    successes: FloatArray,
    trials: FloatArray,
    matrix_ids: NDArray[np.int64],
    order: Iterable[str],
    fixed_lambdas: dict[str, float] | None = None,
) -> SequentialRidgeModel:
    raw_blocks_all = _blocks(raw)
    names = _names(raw_blocks_all)
    successes = np.asarray(successes, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    matrix_ids = np.asarray(matrix_ids, dtype=np.int64)
    eligible = trials > 0.0
    if not eligible.any():
        raise ValueError("model stage has no eligible trials")
    raw_blocks = {name: value[eligible] for name, value in raw_blocks_all.items()}
    successes = successes[eligible]
    trials = trials[eligible]
    matrix_ids = matrix_ids[eligible]
    order_tuple = tuple(order)
    assignments = matrix_folds(matrix_ids)
    folds: list[_Fold] = []
    for fold_index in range(int(assignments.max()) + 1):
        validation = assignments == fold_index
        train = ~validation
        transforms = {
            stage: _fit_standardized_block(
                stage, raw_blocks[stage][train], names[stage]
            )
            for stage in order_tuple
        }
        folds.append(
            _Fold(
                train=train,
                validation=validation,
                train_blocks={
                    stage: transforms[stage].transform(raw_blocks[stage][train])
                    for stage in order_tuple
                },
                validation_blocks={
                    stage: transforms[stage].transform(raw_blocks[stage][validation])
                    for stage in order_tuple
                },
                train_logits=np.zeros(int(train.sum()), dtype=np.float64),
                validation_logits=np.zeros(int(validation.sum()), dtype=np.float64),
            )
        )

    selected_lambdas: dict[str, float] = {}
    cv_scores: dict[str, dict[str, float]] = {}
    for stage_index, stage in enumerate(order_tuple):
        candidates = (0.0,) if stage == "h10" else RIDGE_LAMBDAS
        if fixed_lambdas is not None and stage in fixed_lambdas:
            candidates = (float(fixed_lambdas[stage]),)
        scores: dict[str, float] = {}
        for ridge_lambda in candidates:
            weighted_loss = 0.0
            total_trials = 0.0
            for fold_number, fold in enumerate(folds):
                fit = fit_linear(
                    f"cv{fold_number}_{stage_index}_{stage}_{ridge_lambda:g}",
                    stage,
                    fold.train_blocks[stage],
                    successes[fold.train],
                    trials[fold.train],
                    float(ridge_lambda),
                    fold.train_logits,
                )
                logits = fold.validation_logits + fit.correction(
                    fold.validation_blocks[stage]
                )
                fold_trials = float(trials[fold.validation].sum())
                weighted_loss += (
                    _binomial_loss(
                        successes[fold.validation], trials[fold.validation], logits
                    )
                    * fold_trials
                )
                total_trials += fold_trials
            scores[f"{ridge_lambda:g}"] = weighted_loss / total_trials
        minimum = min(scores.values())
        tied = [
            value for value in candidates if scores[f"{value:g}"] <= minimum + 1e-12
        ]
        selected = float(max(tied))
        selected_lambdas[stage] = selected
        cv_scores[stage] = scores
        for fold_number, fold in enumerate(folds):
            fit = fit_linear(
                f"cv{fold_number}_{stage_index}_{stage}_selected",
                stage,
                fold.train_blocks[stage],
                successes[fold.train],
                trials[fold.train],
                selected,
                fold.train_logits,
            )
            fold.train_logits += fit.correction(fold.train_blocks[stage])
            fold.validation_logits += fit.correction(fold.validation_blocks[stage])

    transforms = {
        stage: _fit_standardized_block(stage, raw_blocks[stage], names[stage])
        for stage in order_tuple
    }
    logits = np.zeros(successes.size, dtype=np.float64)
    fits: list[LinearFit] = []
    for stage_index, stage in enumerate(order_tuple):
        design = transforms[stage].transform(raw_blocks[stage])
        fit = fit_linear(
            f"final_{stage_index}_{stage}",
            stage,
            design,
            successes,
            trials,
            selected_lambdas[stage],
            logits,
        )
        fits.append(fit)
        logits += fit.correction(design)
    return SequentialRidgeModel(
        order=order_tuple,
        transforms=transforms,
        fits=tuple(fits),
        selected_lambdas=selected_lambdas,
        cv_scores=cv_scores,
    )


def _counts(labels: NDArray[np.int8]) -> tuple[FloatArray, FloatArray]:
    values = np.asarray(labels, dtype=np.int8)
    return (
        values.sum(axis=1).astype(np.float64),
        np.full(values.shape[0], values.shape[1], dtype=np.float64),
    )


def _crossfit_sequential_predictions(
    raw: PredictionRawFeatures | dict[str, FloatArray],
    labels: NDArray[np.int8],
    matrix_ids: NDArray[np.int64],
    order: tuple[str, ...],
    fixed_lambdas: dict[str, float] | None = None,
) -> FloatArray:
    raw_blocks = _blocks(raw)
    assignments = matrix_folds(matrix_ids)
    predictions = np.empty(labels.shape[0], dtype=np.float64)
    for fold in range(int(assignments.max()) + 1):
        validation = assignments == fold
        train = ~validation
        successes, trials = _counts(labels[train])
        model = fit_sequential_ridge(
            {name: values[train] for name, values in raw_blocks.items()},
            successes,
            trials,
            matrix_ids[train],
            order,
            fixed_lambdas=fixed_lambdas,
        )
        predictions[validation] = model.predict(
            {name: values[validation] for name, values in raw_blocks.items()}
        )
    return predictions


@dataclass(frozen=True)
class HurdleModel:
    break_model: SequentialRidgeModel
    run8_model: SequentialRidgeModel
    strict_model: SequentialRidgeModel

    def predict(self, raw: PredictionRawFeatures) -> FloatArray:
        return np.clip(
            self.break_model.predict(raw)
            * self.run8_model.predict(raw)
            * self.strict_model.predict(raw),
            1e-12,
            1.0 - 1e-12,
        )


@dataclass(frozen=True)
class AuxiliaryStackModel:
    first5_model: SequentialRidgeModel
    centroid_model: SequentialRidgeModel
    strict_model: SequentialRidgeModel

    def _augmented(self, raw: PredictionRawFeatures) -> dict[str, FloatArray]:
        blocks = _blocks(raw)
        auxiliary = np.column_stack(
            (
                logit(np.clip(self.first5_model.predict(raw), 1e-8, 1.0 - 1e-8)),
                logit(np.clip(self.centroid_model.predict(raw), 1e-8, 1.0 - 1e-8)),
            )
        )
        blocks["auxiliary"] = auxiliary
        return blocks

    def predict(self, raw: PredictionRawFeatures) -> FloatArray:
        return self.strict_model.predict(self._augmented(raw))


@dataclass(frozen=True)
class GuardedNonlinearModel:
    beta_model: SequentialRidgeModel
    estimator: HistGradientBoostingClassifier
    calibrator: LinearFit
    hyperparameters: dict[str, Any]
    cv_scores: dict[str, float]

    def design(self, raw: PredictionRawFeatures) -> FloatArray:
        beta_logit = self.beta_model.predict_logit(raw)[:, None]
        return np.column_stack((raw.h10, raw.dynamics, beta_logit))

    def predict(self, raw: PredictionRawFeatures) -> FloatArray:
        uncalibrated = np.clip(
            self.estimator.predict_proba(self.design(raw))[:, 1], 1e-8, 1.0 - 1e-8
        )
        calibration_design = logit(uncalibrated)[:, None]
        return np.clip(
            expit(self.calibrator.correction(calibration_design)),
            1e-12,
            1.0 - 1e-12,
        )


@dataclass(frozen=True)
class PredictionFamilyModel:
    family: str
    candidate: str
    baseline: SequentialRidgeModel
    enhanced: Any

    def predict(self, raw: PredictionRawFeatures) -> dict[str, FloatArray]:
        return {
            "h10": self.baseline.predict(raw),
            "enhanced": self.enhanced.predict(raw),
        }


def _expand_branches(
    design: FloatArray, labels: NDArray[np.int8]
) -> tuple[FloatArray, NDArray[np.int8]]:
    return np.repeat(design, labels.shape[1], axis=0), labels.reshape(-1)


def _fit_nonlinear(
    raw: PredictionRawFeatures,
    labels: NDArray[np.int8],
    matrix_ids: NDArray[np.int64],
    template: GuardedNonlinearModel | None = None,
) -> GuardedNonlinearModel:
    successes, trials = _counts(labels)
    beta_lambdas = (
        template.beta_model.selected_lambdas if template is not None else None
    )
    beta_model = fit_sequential_ridge(
        raw,
        successes,
        trials,
        matrix_ids,
        ("beta",),
        fixed_lambdas=beta_lambdas,
    )
    beta_oof = _crossfit_sequential_predictions(
        raw, labels, matrix_ids, ("beta",), fixed_lambdas=beta_lambdas
    )
    design = np.column_stack(
        (raw.h10, raw.dynamics, logit(np.clip(beta_oof, 1e-8, 1 - 1e-8)))
    )
    assignments = matrix_folds(matrix_ids)
    scores: dict[str, float] = {}
    chosen_oof: dict[int, FloatArray] = {}
    grid = (template.hyperparameters,) if template is not None else NONLINEAR_GRID
    for grid_index, parameters in enumerate(grid):
        oof = np.empty(labels.shape[0], dtype=np.float64)
        for fold in range(int(assignments.max()) + 1):
            validation = assignments == fold
            train = ~validation
            expanded_x, expanded_y = _expand_branches(design[train], labels[train])
            estimator = HistGradientBoostingClassifier(
                loss="log_loss", random_state=0, **parameters
            )
            estimator.fit(expanded_x, expanded_y)
            oof[validation] = estimator.predict_proba(design[validation])[:, 1]
        value = _binomial_loss(
            labels.sum(axis=1),
            np.full(labels.shape[0], labels.shape[1]),
            logit(np.clip(oof, 1e-8, 1.0 - 1e-8)),
        )
        scores[str(grid_index)] = value
        chosen_oof[grid_index] = oof
    best_value = min(scores.values())
    best_index = min(
        index for index in range(len(grid)) if scores[str(index)] <= best_value + 1e-12
    )
    parameters = dict(grid[best_index])
    oof = np.clip(chosen_oof[best_index], 1e-8, 1.0 - 1e-8)
    calibrator = fit_linear(
        "nonlinear_platt_calibration",
        "uncalibrated_logit",
        logit(oof)[:, None],
        labels.sum(axis=1),
        np.full(labels.shape[0], labels.shape[1]),
        ridge_lambda=0.0,
    )
    final_design = np.column_stack(
        (raw.h10, raw.dynamics, beta_model.predict_logit(raw)[:, None])
    )
    expanded_x, expanded_y = _expand_branches(final_design, labels)
    estimator = HistGradientBoostingClassifier(
        loss="log_loss", random_state=0, **parameters
    )
    estimator.fit(expanded_x, expanded_y)
    return GuardedNonlinearModel(
        beta_model=beta_model,
        estimator=estimator,
        calibrator=calibrator,
        hyperparameters=parameters,
        cv_scores=scores,
    )


def fit_prediction_family(
    family: str,
    candidate: str,
    raw: PredictionRawFeatures,
    labels: dict[str, NDArray[np.int8]],
    matrix_ids: NDArray[np.int64],
    template: PredictionFamilyModel | None = None,
) -> PredictionFamilyModel:
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown prediction family: {family}")
    strict_successes, strict_trials = _counts(labels["strict"])
    baseline = fit_sequential_ridge(
        raw,
        strict_successes,
        strict_trials,
        matrix_ids,
        ("h10",),
        fixed_lambdas=(template.baseline.selected_lambdas if template else None),
    )
    if family == "direct_ridge":
        enhanced: Any = fit_sequential_ridge(
            raw,
            strict_successes,
            strict_trials,
            matrix_ids,
            ("h10", "state", "beta", "interaction"),
            fixed_lambdas=(
                template.enhanced.selected_lambdas if template is not None else None
            ),
        )
    elif family == "hierarchical_offset":
        enhanced = fit_sequential_ridge(
            raw,
            strict_successes,
            strict_trials,
            matrix_ids,
            ("beta", "h10", "state", "dynamics"),
            fixed_lambdas=(
                template.enhanced.selected_lambdas if template is not None else None
            ),
        )
    elif family == "local_dynamics":
        enhanced = fit_sequential_ridge(
            raw,
            strict_successes,
            strict_trials,
            matrix_ids,
            ("h10", "beta", "dynamics"),
            fixed_lambdas=(
                template.enhanced.selected_lambdas if template is not None else None
            ),
        )
    elif family == "hurdle":
        break_successes, break_trials = _counts(labels["break"])
        run_successes = labels["run8"].sum(axis=1).astype(np.float64)
        run_trials = labels["break"].sum(axis=1).astype(np.float64)
        conditional_successes = labels["strict"].sum(axis=1).astype(np.float64)
        conditional_trials = labels["run8"].sum(axis=1).astype(np.float64)
        order = ("h10", "state", "beta", "interaction", "dynamics")
        hurdle_template = template.enhanced if template is not None else None
        enhanced = HurdleModel(
            break_model=fit_sequential_ridge(
                raw,
                break_successes,
                break_trials,
                matrix_ids,
                order,
                fixed_lambdas=(
                    hurdle_template.break_model.selected_lambdas
                    if hurdle_template is not None
                    else None
                ),
            ),
            run8_model=fit_sequential_ridge(
                raw,
                run_successes,
                run_trials,
                matrix_ids,
                order,
                fixed_lambdas=(
                    hurdle_template.run8_model.selected_lambdas
                    if hurdle_template is not None
                    else None
                ),
            ),
            strict_model=fit_sequential_ridge(
                raw,
                conditional_successes,
                conditional_trials,
                matrix_ids,
                order,
                fixed_lambdas=(
                    hurdle_template.strict_model.selected_lambdas
                    if hurdle_template is not None
                    else None
                ),
            ),
        )
    elif family == "auxiliary_stack":
        # The auxiliary logits summarize relaxed geometry; the strict stack
        # receives them alongside h10 and dynamics. Keeping the 309-coordinate
        # beta block out of these two nested nuisance fits avoids duplicating
        # the dedicated beta-propensity and hierarchical families.
        auxiliary_order = ("h10", "dynamics")
        auxiliary_template = template.enhanced if template is not None else None
        first5_successes, first5_trials = _counts(labels["first5"])
        centroid_successes, centroid_trials = _counts(labels["centroid"])
        first5_model = fit_sequential_ridge(
            raw,
            first5_successes,
            first5_trials,
            matrix_ids,
            auxiliary_order,
            fixed_lambdas=(
                auxiliary_template.first5_model.selected_lambdas
                if auxiliary_template is not None
                else None
            ),
        )
        centroid_model = fit_sequential_ridge(
            raw,
            centroid_successes,
            centroid_trials,
            matrix_ids,
            auxiliary_order,
            fixed_lambdas=(
                auxiliary_template.centroid_model.selected_lambdas
                if auxiliary_template is not None
                else None
            ),
        )
        first5_oof = _crossfit_sequential_predictions(
            raw,
            labels["first5"],
            matrix_ids,
            auxiliary_order,
            fixed_lambdas=(
                auxiliary_template.first5_model.selected_lambdas
                if auxiliary_template is not None
                else None
            ),
        )
        centroid_oof = _crossfit_sequential_predictions(
            raw,
            labels["centroid"],
            matrix_ids,
            auxiliary_order,
            fixed_lambdas=(
                auxiliary_template.centroid_model.selected_lambdas
                if auxiliary_template is not None
                else None
            ),
        )
        augmented = _blocks(raw)
        augmented["auxiliary"] = np.column_stack(
            (
                logit(np.clip(first5_oof, 1e-8, 1.0 - 1e-8)),
                logit(np.clip(centroid_oof, 1e-8, 1.0 - 1e-8)),
            )
        )
        strict_model = fit_sequential_ridge(
            augmented,
            strict_successes,
            strict_trials,
            matrix_ids,
            ("h10", "dynamics", "auxiliary"),
            fixed_lambdas=(
                auxiliary_template.strict_model.selected_lambdas
                if auxiliary_template is not None
                else None
            ),
        )
        enhanced = AuxiliaryStackModel(first5_model, centroid_model, strict_model)
    else:
        enhanced = _fit_nonlinear(
            raw,
            labels["strict"],
            matrix_ids,
            template=(template.enhanced if template is not None else None),
        )
    return PredictionFamilyModel(family, candidate, baseline, enhanced)


def crossfit_prediction_family(
    family: str,
    candidate: str,
    raw: PredictionRawFeatures,
    labels: dict[str, NDArray[np.int8]],
    matrix_ids: NDArray[np.int64],
) -> tuple[dict[str, FloatArray], PredictionFamilyModel]:
    # Hyperparameters are selected once from the development folds. OOF model
    # fits then reuse those frozen values, avoiding a costly and unnecessary
    # nested search: the untouched confirmation, not pilot CV, carries the
    # eventual claim.
    final = fit_prediction_family(family, candidate, raw, labels, matrix_ids)
    assignments = matrix_folds(matrix_ids)
    baseline = np.empty(matrix_ids.size, dtype=np.float64)
    enhanced = np.empty(matrix_ids.size, dtype=np.float64)
    for fold in range(int(assignments.max()) + 1):
        validation = assignments == fold
        train = ~validation
        model = fit_prediction_family(
            family,
            candidate,
            raw.selected(train),
            {name: values[train] for name, values in labels.items()},
            matrix_ids[train],
            template=final,
        )
        predicted = model.predict(raw.selected(validation))
        baseline[validation] = predicted["h10"]
        enhanced[validation] = predicted["enhanced"]
    return {"h10": baseline, "enhanced": enhanced}, final


def model_summary(model: PredictionFamilyModel) -> dict[str, Any]:
    def ridge(item: SequentialRidgeModel) -> dict[str, Any]:
        return {
            "order": item.order,
            "selected_lambdas": item.selected_lambdas,
            "features": {
                name: int(transform.kept_indices.size)
                for name, transform in item.transforms.items()
            },
        }

    enhanced = model.enhanced
    output: dict[str, Any] = {
        "family": model.family,
        "candidate": model.candidate,
        "baseline": ridge(model.baseline),
    }
    if isinstance(enhanced, SequentialRidgeModel):
        output["enhanced"] = ridge(enhanced)
    elif isinstance(enhanced, HurdleModel):
        output["enhanced"] = {
            "break": ridge(enhanced.break_model),
            "run8_given_break": ridge(enhanced.run8_model),
            "strict_given_run8": ridge(enhanced.strict_model),
        }
    elif isinstance(enhanced, AuxiliaryStackModel):
        output["enhanced"] = {
            "first5": ridge(enhanced.first5_model),
            "centroid": ridge(enhanced.centroid_model),
            "strict": ridge(enhanced.strict_model),
        }
    elif isinstance(enhanced, GuardedNonlinearModel):
        output["enhanced"] = {
            "beta_propensity": ridge(enhanced.beta_model),
            "hyperparameters": enhanced.hyperparameters,
            "grid_scores": enhanced.cv_scores,
            "calibration_intercept": enhanced.calibrator.intercept,
            "calibration_slope": float(enhanced.calibrator.coefficient[0]),
        }
    else:  # pragma: no cover
        raise TypeError("unsupported enhanced model")
    return output


def save_prediction_models(
    path: Path, models: dict[str, PredictionFamilyModel]
) -> None:
    with path.open("wb") as handle:
        pickle.dump(models, handle, protocol=5)


def load_prediction_models(path: Path) -> dict[str, PredictionFamilyModel]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or set(value) != {"02", "03"}:
        raise ValueError("invalid prediction model archive")
    for candidate, model in value.items():
        if not isinstance(model, PredictionFamilyModel) or model.candidate != candidate:
            raise ValueError("invalid candidate prediction model")
    return value
