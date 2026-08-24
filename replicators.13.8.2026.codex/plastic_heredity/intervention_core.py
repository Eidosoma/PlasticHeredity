"""Clean-room intervention primitives for the validated F12 process.

The functions in this module are additive wrappers around the sealed Codex
simulator.  They do not alter its growth, fission, daughter, or endpoint
contracts.  Molecular interventions change only the restored composition;
network interventions change only the beta matrix supplied to future calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .config import CANDIDATES, ExperimentConfig, GardConfig
from .features import history_features, state_graph_features
from .processes import evaluate_process
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    advance_fission,
    simulate_future_absorbing,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, order=True)
class MolecularEdit:
    remove_type: int
    add_type: int


@dataclass(frozen=True)
class ScoredEdit:
    edit: MolecularEdit
    predicted_probability: float
    predicted_shift: float


@dataclass(frozen=True)
class SelectedEdits:
    noop_probability: float
    model_up: ScoredEdit
    model_down: ScoredEdit
    random: ScoredEdit


@dataclass(frozen=True)
class BetaSurgery:
    name: str
    beta: FloatArray
    flat_indices: NDArray[np.int64]
    before: FloatArray
    after: FloatArray
    requested_norm: float
    observed_norm: float


@dataclass(frozen=True)
class InterventionOutcome:
    joint_break_run3: bool
    break_event: bool
    run3_after_break: bool
    inherited_boundary_count: int
    first_break_time: int
    renewal_certification_time: int
    completed_horizon: bool
    observed_fissions: int
    total_growth_updates: int
    mean_growth_updates: float
    final_entropy: float
    final_occupied_types: int
    final_composition: IntArray
    boundary_h: FloatArray
    growth_updates: NDArray[np.int32]
    record_digest: str


@dataclass(frozen=True)
class ControlledResult:
    records: tuple[FissionRecord, ...]
    completed_horizon: bool
    final_snapshot: Snapshot
    interventions_applied: int
    selected_edits: tuple[MolecularEdit, ...]


class FrozenFullPredictor:
    """Portable candidate-separated 5x composite predictor without sklearn."""

    def __init__(self, arrays: dict[str, NDArray]):
        self.arrays = {name: np.asarray(value).copy() for name, value in arrays.items()}

    @classmethod
    def load(cls, path: Path | str) -> "FrozenFullPredictor":
        with np.load(path, allow_pickle=False) as archive:
            return cls({name: archive[name] for name in archive.files})

    def predict_features(
        self, candidate: str, state_graph: NDArray, history: NDArray
    ) -> FloatArray:
        state = np.atleast_2d(np.asarray(state_graph, dtype=np.float64))
        direct = np.atleast_2d(np.asarray(history, dtype=np.float64))
        if state.shape[0] != direct.shape[0]:
            raise ValueError("state and history row counts differ")
        base = f"c{candidate}"
        state_scaled = (
            state - self.arrays[f"{base}__full_state_scaler_mean"]
        ) / self.arrays[f"{base}__full_state_scaler_scale"]
        components = (
            state_scaled - self.arrays[f"{base}__full_state_pca_mean"]
        ) @ self.arrays[f"{base}__full_state_pca_components"].T
        unscaled = np.column_stack((components, direct))
        prefix = f"{base}__full"
        transformed = (
            unscaled - self.arrays[f"{prefix}__scaler_mean"]
        ) / self.arrays[f"{prefix}__scaler_scale"]
        logits = (
            transformed @ self.arrays[f"{prefix}__classifier_coef"].T
            + self.arrays[f"{prefix}__classifier_intercept"]
        ).reshape(-1)
        return np.clip(
            1.0 / (1.0 + np.exp(-np.clip(logits, -709.0, 709.0))),
            1e-12,
            1.0 - 1e-12,
        )

    def predict_snapshot(
        self,
        candidate: str,
        snapshot: Snapshot,
        beta: NDArray,
        config: GardConfig,
    ) -> float:
        state = state_graph_features(snapshot.composition, beta, config)
        direct = history_features(snapshot, config)
        return float(self.predict_features(candidate, state, direct)[0])


def enumerate_legal_edits(composition: NDArray) -> tuple[MolecularEdit, ...]:
    values = np.asarray(composition)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("composition must be a one-dimensional integer array")
    if np.any(values < 0):
        raise ValueError("composition cannot contain negative counts")
    present = np.flatnonzero(values > 0)
    return tuple(
        MolecularEdit(int(remove), int(add))
        for remove in present
        for add in range(values.size)
        if remove != add
    )


def apply_molecular_edit(composition: NDArray, edit: MolecularEdit) -> IntArray:
    values = np.asarray(composition)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("composition must be a one-dimensional integer array")
    if edit.remove_type == edit.add_type:
        raise ValueError("same-type substitutions are illegal")
    if not 0 <= edit.remove_type < values.size or not 0 <= edit.add_type < values.size:
        raise ValueError("edit type is outside the composition")
    if values[edit.remove_type] < 1:
        raise ValueError("cannot remove an absent molecule")
    edited = np.asarray(values, dtype=np.int64).copy()
    mass = int(edited.sum())
    edited[edit.remove_type] -= 1
    edited[edit.add_type] += 1
    if np.any(edited < 0) or int(edited.sum()) != mass:
        raise AssertionError("legal substitution violated its mass contract")
    return edited


def _summarize_profiles_many(
    values: FloatArray, composition_weights: FloatArray, active: NDArray
) -> FloatArray:
    """Vectorized equivalent of the sealed 13-summary feature map."""

    quantiles = np.quantile(
        values, (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95), axis=1
    ).T
    active_float = np.asarray(active, dtype=np.float64)
    active_count = np.maximum(active_float.sum(axis=1), 1.0)
    return np.column_stack(
        (
            values.mean(axis=1),
            values.std(axis=1),
            values.min(axis=1),
            quantiles,
            values.max(axis=1),
            np.sum(values * composition_weights, axis=1),
            np.sum(values * active_float, axis=1) / active_count,
        )
    )


def state_graph_features_many(
    compositions: NDArray, beta: NDArray, config: GardConfig
) -> FloatArray:
    """Evaluate the existing invariant 195-feature map for many edits at once.

    This is an algebraic batching of :func:`state_graph_features`, not a new
    representation or screening approximation.  Its numerical agreement with
    the scalar implementation is a mandatory pre-scientific validation.
    """

    x = np.asarray(compositions, dtype=np.float64)
    matrix = np.asarray(beta, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != config.n_types:
        raise ValueError("compositions must have shape (edits, n_types)")
    if matrix.shape != (config.n_types, config.n_types):
        raise ValueError("beta has the wrong shape")
    mass = x.sum(axis=1)
    if np.any(mass <= 0.0):
        raise ValueError("features are undefined for an empty assembly")
    fraction = x / mass[:, None]
    active = x > 0.0
    active_fraction = active / np.maximum(active.sum(axis=1), 1)[:, None]
    incoming = fraction @ matrix.T
    outgoing = fraction @ matrix
    boost = 1.0 + incoming
    join = config.k_join * (1.0 / config.n_types) * mass[:, None] * boost
    leave = config.k_leave * x * boost
    log_beta = np.log(np.maximum(matrix, np.finfo(np.float64).tiny))
    fixed_profiles = (
        log_beta.mean(axis=1),
        log_beta.std(axis=1),
        log_beta.mean(axis=0),
        log_beta.std(axis=0),
        np.log1p(np.maximum(matrix.mean(axis=1), 0.0)),
        np.log1p(np.maximum(matrix.mean(axis=0), 0.0)),
    )
    profiles = (
        x / config.n_max,
        fraction,
        active.astype(np.float64),
        np.log1p(np.maximum(incoming, 0.0)),
        np.log1p(np.maximum(outgoing, 0.0)),
        np.log1p(np.maximum(join, 0.0)),
        np.log1p(np.maximum(leave, 0.0)),
        *(np.broadcast_to(profile, x.shape) for profile in fixed_profiles),
        np.log1p(np.maximum(active_fraction @ matrix.T, 0.0)),
        np.log1p(np.maximum(active_fraction @ matrix, 0.0)),
    )
    output = np.column_stack(
        [
            _summarize_profiles_many(profile, fraction, active)
            for profile in profiles
        ]
    )
    if output.shape != (x.shape[0], 195) or not np.isfinite(output).all():
        raise AssertionError("batched state/graph feature contract failed")
    return output


def edited_snapshot(snapshot: Snapshot, edit: MolecularEdit) -> Snapshot:
    return Snapshot(
        composition=apply_molecular_edit(snapshot.composition, edit),
        generation=snapshot.generation,
        inheritance=snapshot.inheritance,
        boundary_h=snapshot.boundary_h,
        previous_growth_steps=snapshot.previous_growth_steps,
        cumulative_growth_steps=snapshot.cumulative_growth_steps,
    )


def score_legal_edits(
    predictor: FrozenFullPredictor,
    candidate: str,
    snapshot: Snapshot,
    beta: NDArray,
    config: GardConfig,
) -> tuple[float, tuple[ScoredEdit, ...]]:
    direct = history_features(snapshot, config)
    noop = float(
        predictor.predict_features(
            candidate,
            state_graph_features(snapshot.composition, beta, config),
            direct,
        )[0]
    )
    edits = enumerate_legal_edits(snapshot.composition)
    if not edits:
        raise ValueError("restored state has no legal molecular substitutions")
    compositions = np.vstack(
        [apply_molecular_edit(snapshot.composition, edit) for edit in edits]
    )
    features = state_graph_features_many(compositions, beta, config)
    direct_many = np.broadcast_to(direct, (len(edits), direct.size))
    probabilities = predictor.predict_features(candidate, features, direct_many)
    output = tuple(
        ScoredEdit(edit, float(probability), float(probability - noop))
        for edit, probability in zip(edits, probabilities, strict=True)
    )
    return noop, output


def select_scored_edits(
    noop_probability: float,
    scores: tuple[ScoredEdit, ...],
    random_rng: np.random.Generator,
) -> SelectedEdits:
    if not scores:
        raise ValueError("cannot select from an empty legal edit list")
    probabilities = np.asarray(
        [item.predicted_probability for item in scores], dtype=np.float64
    )
    maximum = float(probabilities.max())
    minimum = float(probabilities.min())
    up_index = int(np.flatnonzero(probabilities == maximum)[0])
    down_index = int(np.flatnonzero(probabilities == minimum)[0])
    random_index = int(random_rng.integers(0, len(scores)))
    return SelectedEdits(
        noop_probability=float(noop_probability),
        model_up=scores[up_index],
        model_down=scores[down_index],
        random=scores[random_index],
    )


def catalytic_support(composition: NDArray, beta: NDArray) -> FloatArray:
    """Return target support under Codex's beta[target, catalyst] convention."""

    values = np.asarray(composition, dtype=np.float64)
    matrix = np.asarray(beta, dtype=np.float64)
    if matrix.shape != (values.size, values.size):
        raise ValueError("beta and composition dimensions differ")
    return matrix @ values


def select_rule_edits(composition: NDArray, beta: NDArray) -> dict[str, MolecularEdit]:
    scores = catalytic_support(composition, beta)
    legal = enumerate_legal_edits(composition)
    if not legal:
        raise ValueError("restored state has no legal molecular substitutions")
    differences = np.asarray(
        [scores[item.add_type] - scores[item.remove_type] for item in legal]
    )
    return {
        "RULE_DOWN": legal[int(np.flatnonzero(differences == differences.max())[0])],
        "RULE_UP": legal[int(np.flatnonzero(differences == differences.min())[0])],
    }


def targeted_beta_surgery(
    composition: NDArray, beta: NDArray, fraction: float, tighten: bool
) -> BetaSurgery:
    if not 0.0 < fraction < 1.0:
        raise ValueError("surgery fraction must lie strictly between zero and one")
    matrix = np.asarray(beta, dtype=np.float64)
    present = np.flatnonzero(np.asarray(composition) > 0)
    if present.size == 0:
        raise ValueError("beta surgery is undefined for an empty assembly")
    rows, columns = np.meshgrid(present, present, indexing="ij")
    flat = np.ravel_multi_index((rows.ravel(), columns.ravel()), matrix.shape)
    before = matrix.ravel()[flat].copy()
    factor = 1.0 + fraction if tighten else 1.0 - fraction
    altered = matrix.copy()
    altered.ravel()[flat] *= factor
    after = altered.ravel()[flat].copy()
    requested = fraction * float(np.linalg.norm(before))
    observed = float(np.linalg.norm(altered - matrix))
    return BetaSurgery(
        name="TIGHTEN" if tighten else "LOOSEN",
        beta=altered,
        flat_indices=flat.astype(np.int64),
        before=before,
        after=after,
        requested_norm=requested,
        observed_norm=observed,
    )


def random_beta_surgery(
    composition: NDArray,
    beta: NDArray,
    fraction: float,
    rng: np.random.Generator,
) -> BetaSurgery:
    matrix = np.asarray(beta, dtype=np.float64)
    present = np.flatnonzero(np.asarray(composition) > 0)
    if present.size < 2:
        raise ValueError("balanced random surgery requires at least two present types")
    targeted = matrix[np.ix_(present, present)]
    requested = fraction * float(np.linalg.norm(targeted))
    count = int(present.size**2)
    flat = np.sort(
        np.asarray(
            rng.choice(matrix.size, size=count, replace=False), dtype=np.int64
        )
    )
    before = matrix.ravel()[flat].copy()
    direction = np.asarray(rng.standard_normal(count), dtype=np.float64)
    direction -= direction.mean()
    if not np.any(direction > 0.0) or not np.any(direction < 0.0):
        direction = np.linspace(-1.0, 1.0, count, dtype=np.float64)

    def distance(scale: float) -> float:
        with np.errstate(over="ignore"):
            changed = before * np.exp(scale * direction)
        if not np.isfinite(changed).all():
            return float("inf")
        return float(np.linalg.norm(changed - before))

    upper = 1.0
    for _ in range(128):
        if distance(upper) >= requested:
            break
        upper *= 2.0
    else:  # pragma: no cover - defensive numerical limit
        raise ValueError("could not bracket random surgery magnitude")
    lower = 0.0
    for _ in range(128):
        middle = 0.5 * (lower + upper)
        if distance(middle) < requested:
            lower = middle
        else:
            upper = middle
    scale = 0.5 * (lower + upper)
    after = before * np.exp(scale * direction)
    altered = matrix.copy()
    altered.ravel()[flat] = after
    observed = float(np.linalg.norm(altered - matrix))
    return BetaSurgery(
        name="RANDOM_SURGERY",
        beta=altered,
        flat_indices=flat,
        before=before,
        after=after,
        requested_norm=requested,
        observed_norm=observed,
    )


def _records_digest(records: list[FissionRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(np.ascontiguousarray(record.parent).tobytes())
        digest.update(np.ascontiguousarray(record.daughter).tobytes())
        digest.update(np.asarray((record.h,), dtype=np.float64).tobytes())
        digest.update(np.asarray((record.growth_steps,), dtype=np.int64).tobytes())
    return digest.hexdigest()


def _entropy(composition: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    mass = float(values.sum())
    if mass <= 0.0:
        return 0.0
    positive = values[values > 0.0] / mass
    return float(-np.dot(positive, np.log(positive)))


def outcome_from_records(
    snapshot: Snapshot,
    records: list[FissionRecord],
    completed_horizon: bool,
    horizon: int,
    inheritance_threshold: float = 0.9,
) -> InterventionOutcome:
    process = evaluate_process(records, inheritance_threshold)
    inherited = np.asarray(
        [record.h > inheritance_threshold for record in records], dtype=bool
    )
    breaks = np.flatnonzero(~inherited)
    first_break = int(breaks[0]) if breaks.size else -1
    renewal = -1
    if first_break >= 0:
        after = inherited[first_break + 1 :]
        for start in range(max(0, after.size - 2)):
            if bool(after[start : start + 3].all()):
                renewal = first_break + start + 4
                break
    final = (
        np.asarray(records[-1].daughter, dtype=np.int64).copy()
        if records
        else np.asarray(snapshot.composition, dtype=np.int64).copy()
    )
    boundary_h = np.full(horizon, np.nan, dtype=np.float64)
    growth = np.full(horizon, -1, dtype=np.int32)
    for index, record in enumerate(records):
        boundary_h[index] = record.h
        growth[index] = record.growth_steps
    total_growth = int(sum(record.growth_steps for record in records))
    mean_growth = float(total_growth / len(records)) if records else float("nan")
    return InterventionOutcome(
        joint_break_run3=process.joint_break_run3,
        break_event=process.break_event,
        run3_after_break=bool(process.episode_3 == 1.0),
        inherited_boundary_count=int(inherited.sum()),
        first_break_time=first_break + 1 if first_break >= 0 else -1,
        renewal_certification_time=renewal,
        completed_horizon=bool(completed_horizon),
        observed_fissions=len(records),
        total_growth_updates=total_growth,
        mean_growth_updates=mean_growth,
        final_entropy=_entropy(final),
        final_occupied_types=int(np.count_nonzero(final)),
        final_composition=final,
        boundary_h=boundary_h,
        growth_updates=growth,
        record_digest=_records_digest(records),
    )


def simulate_one_shot(
    snapshot: Snapshot,
    beta: NDArray,
    candidate: str,
    config: GardConfig,
    horizon: int,
    rng: np.random.Generator,
    edit: MolecularEdit | None = None,
) -> InterventionOutcome:
    launch = edited_snapshot(snapshot, edit) if edit is not None else snapshot
    records, completed = simulate_future_absorbing(
        launch, beta, config, CANDIDATES[candidate], horizon, rng
    )
    return outcome_from_records(launch, records, completed, horizon)


Controller = Callable[[Snapshot, NDArray, str, int], MolecularEdit | None]


def simulate_controlled(
    snapshot: Snapshot,
    beta: NDArray,
    candidate: str,
    experiment: ExperimentConfig,
    horizon: int,
    rng: np.random.Generator,
    controller: Controller | None = None,
    release_after: int | None = None,
) -> ControlledResult:
    current = snapshot
    records: list[FissionRecord] = []
    edits: list[MolecularEdit] = []
    cumulative = snapshot.cumulative_growth_steps
    for step in range(horizon):
        try:
            record = advance_fission(
                current.composition,
                beta,
                experiment.gard,
                CANDIDATES[candidate],
                rng,
            )
        except SimulationError:
            return ControlledResult(
                tuple(records), False, current, len(edits), tuple(edits)
            )
        records.append(record)
        cumulative += record.growth_steps
        current = Snapshot(
            composition=record.daughter.copy(),
            generation=current.generation + 1,
            inheritance=current.inheritance
            + (record.h > experiment.gard.inheritance_threshold,),
            boundary_h=current.boundary_h + (float(record.h),),
            previous_growth_steps=record.growth_steps,
            cumulative_growth_steps=cumulative,
        )
        active = release_after is None or step < release_after
        edit = controller(current, beta, candidate, step) if controller and active else None
        if edit is not None:
            current = edited_snapshot(current, edit)
            edits.append(edit)
    return ControlledResult(tuple(records), True, current, len(edits), tuple(edits))
