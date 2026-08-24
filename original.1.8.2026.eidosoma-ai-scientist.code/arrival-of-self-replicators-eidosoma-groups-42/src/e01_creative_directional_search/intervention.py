"""Retrospective frozen-control scoring for the adaptive S13X pilot.

This is deliberately a forensic mechanism, not a prospective estimator. The
completed matched control fixes PhiRL preprocessing, its Fiedler partition, and
all Gaussian densities. Candidate actions are then scored as virtual transitions
from the selected daughter to the one-molecule-edited state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from e01_frozen_timebase_ensemble.core import frozen_clr
from e01_pigozzi_source_audit.core import (
    BOTTOM_ATOM,
    CAUSATION_ATOMS,
    INITIAL_PHIR_ATOM,
    PHIR_ATOMS,
    SYNERGY_ATOM,
    load_safe_lattice,
)


@dataclass(frozen=True, slots=True)
class GaussianModel:
    """One source-compatible Gaussian negative-log-density model."""

    mean: NDArray[np.float64]
    inverse_covariance: NDArray[np.float64]
    log_determinant: float

    @property
    def dimension(self) -> int:
        return len(self.mean)

    def entropy(self, value: NDArray[np.float64]) -> float:
        difference = np.asarray(value, dtype=np.float64) - self.mean
        quadratic = float(difference @ self.inverse_covariance @ difference)
        return 0.5 * (
            self.dimension * math.log(2.0 * math.pi) + self.log_determinant + quadratic
        )


@dataclass(frozen=True, slots=True)
class ConditionalModel:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    joint_model: GaussianModel
    target_model: GaussianModel


@dataclass(frozen=True, slots=True)
class SourceBranchModel:
    source_indices: tuple[int, ...]
    source_model: GaussianModel
    conditionals: tuple[ConditionalModel, ...]


@dataclass(frozen=True, slots=True)
class AtomModel:
    atom: Any
    source_branches: tuple[SourceBranchModel, ...]


@dataclass(frozen=True, slots=True)
class FrozenPhiRLScorer:
    """All immutable objects needed for inexpensive candidate scoring."""

    retained_variables: tuple[int, ...]
    means: NDArray[np.float64]
    standard_deviations: NDArray[np.float64]
    partition_1_local: tuple[int, ...]
    partition_2_local: tuple[int, ...]
    reference_reduced: NDArray[np.float64]
    order: tuple[Any, ...]
    descendants: dict[Any, tuple[Any, ...]]
    atom_models: dict[Any, AtomModel]
    preprocessing_max_abs_error: float

    def reduce_clr(self, clr_rows: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(clr_rows, dtype=np.float64)
        retained = values[:, self.retained_variables]
        processed = (retained - self.means) / self.standard_deviations
        return np.column_stack(
            (
                processed[:, self.partition_1_local].mean(axis=1),
                processed[:, self.partition_2_local].mean(axis=1),
            )
        )

    def score_reduced_pair(
        self,
        previous: NDArray[np.float64],
        candidate: NDArray[np.float64],
    ) -> dict[str, float]:
        partials: dict[Any, float] = {}
        for atom in self.order:
            model = self.atom_models[atom]
            i_plus = math.inf
            i_minus = math.inf
            for source in model.source_branches:
                source_value = previous[list(source.source_indices)]
                i_plus = min(i_plus, source.source_model.entropy(source_value))
                for conditional in source.conditionals:
                    target_value = candidate[list(conditional.target_indices)]
                    joint_value = np.concatenate((source_value, target_value))
                    conditional_entropy = conditional.joint_model.entropy(
                        joint_value
                    ) - conditional.target_model.entropy(target_value)
                    i_minus = min(i_minus, conditional_entropy)
            redundancy = i_plus - i_minus
            if atom == BOTTOM_ATOM:
                partials[atom] = redundancy
            else:
                partials[atom] = redundancy - sum(
                    partials[item] for item in self.descendants[atom]
                )
        synergy = partials[SYNERGY_ATOM]
        downward = sum(partials[item] for item in CAUSATION_ATOMS)
        local_phi_r = partials[INITIAL_PHIR_ATOM] + sum(
            partials[item] for item in PHIR_ATOMS
        )
        return {
            "synergy": float(synergy),
            "downwardCausation": float(downward),
            "emergence": float(synergy + downward),
            "localPhiR": float(local_phi_r),
        }

    def score_count_actions(
        self, state: NDArray[np.integer[Any]]
    ) -> list[dict[str, Any]]:
        parent = np.asarray(state, dtype=np.int64)
        if parent.shape != (100,) or np.any(parent < 0) or int(parent.sum()) <= 0:
            raise ValueError("action state must be a nonempty nonnegative 100-vector")
        action_states: list[NDArray[np.int64]] = []
        identities: list[tuple[str, int, str]] = []
        for component in range(100):
            candidate = parent.copy()
            candidate[component] += 1
            action_states.append(candidate)
            identities.append(("ADD", component, f"ADD_{component + 1:03d}"))
        for component in np.flatnonzero(parent > 0):
            candidate = parent.copy()
            candidate[int(component)] -= 1
            action_states.append(candidate)
            identities.append(
                ("DELETE", int(component), f"DELETE_{int(component) + 1:03d}")
            )
        counts = np.vstack([parent, *action_states])
        clr, _, _ = frozen_clr(counts)
        reduced = self.reduce_clr(clr)
        previous = reduced[0]
        rows = []
        for order, ((operation, component, action_id), target) in enumerate(
            zip(identities, reduced[1:], strict=True)
        ):
            rows.append(
                {
                    "actionOrder": order,
                    "actionId": action_id,
                    "operation": operation,
                    "componentIndexZeroBased": component,
                    **self.score_reduced_pair(previous, target),
                }
            )
        return rows


def _fit_gaussian(reference: NDArray[np.float64]) -> GaussianModel:
    values = np.asarray(reference, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    dimension = values.shape[0]
    mean = values.mean(axis=1)
    if dimension == 1:
        standard_deviation = float(values[0].std())
        covariance = np.asarray([[standard_deviation * standard_deviation]])
    else:
        covariance = np.asarray(np.cov(values, ddof=0), dtype=np.float64)
        covariance += np.eye(dimension) * (
            1e-6 * float(np.trace(covariance)) / dimension
        )
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(log_determinant):
        raise ValueError("frozen Gaussian covariance is not positive definite")
    return GaussianModel(
        mean=np.asarray(mean, dtype=np.float64),
        inverse_covariance=np.linalg.inv(covariance),
        log_determinant=float(log_determinant),
    )


def _atom_model(atom: Any, reduced: NDArray[np.float64]) -> AtomModel:
    source_branches = []
    for source_antichain in atom[0]:
        source_indices = tuple(map(int, source_antichain))
        source_reference = reduced[list(source_indices), :-1]
        conditional_models = []
        for target_antichain in atom[1]:
            target_indices = tuple(map(int, target_antichain))
            target_reference = reduced[list(target_indices), 1:]
            joint_reference = np.vstack((source_reference, target_reference))
            conditional_models.append(
                ConditionalModel(
                    source_indices=source_indices,
                    target_indices=target_indices,
                    joint_model=_fit_gaussian(joint_reference),
                    target_model=_fit_gaussian(target_reference),
                )
            )
        source_branches.append(
            SourceBranchModel(
                source_indices=source_indices,
                source_model=_fit_gaussian(source_reference),
                conditionals=tuple(conditional_models),
            )
        )
    return AtomModel(atom=atom, source_branches=tuple(source_branches))


def build_frozen_phirl_scorer(
    clr: NDArray[np.float64], source_result: Any, safe_lattice_path: str | Path
) -> FrozenPhiRLScorer:
    """Build a scorer from one eligible completed-control PhiRL result."""

    if source_result.implementation != "PHIRL_REGULARIZED_SOURCE":
        raise ValueError("frozen S13X scorer requires the PhiRL source branch")
    if source_result.partition_average is None or source_result.processed is None:
        raise ValueError("source result lacks a completed eligible reference model")
    retained = tuple(map(int, source_result.retained_variables))
    source = np.asarray(clr, dtype=np.float64)
    retained_source = source[:, retained]
    means = retained_source.mean(axis=0)
    standard_deviations = retained_source.std(axis=0)
    if np.any(standard_deviations <= 1e-8):
        raise ValueError("retained-variable reconstruction violates PhiRL filter")
    reconstructed = ((retained_source - means) / standard_deviations).T
    preprocessing_error = float(
        np.max(np.abs(reconstructed - np.asarray(source_result.processed)))
    )
    retained_lookup = {original: local for local, original in enumerate(retained)}
    p1 = tuple(retained_lookup[int(value)] for value in source_result.partition_1)
    p2 = tuple(retained_lookup[int(value)] for value in source_result.partition_2)
    reduced = np.asarray(source_result.partition_average, dtype=np.float64)
    order_list, descendants = load_safe_lattice(safe_lattice_path)
    order = tuple(order_list)
    models = {atom: _atom_model(atom, reduced) for atom in order}
    return FrozenPhiRLScorer(
        retained_variables=retained,
        means=np.asarray(means, dtype=np.float64),
        standard_deviations=np.asarray(standard_deviations, dtype=np.float64),
        partition_1_local=p1,
        partition_2_local=p2,
        reference_reduced=reduced,
        order=order,
        descendants=descendants,
        atom_models=models,
        preprocessing_max_abs_error=preprocessing_error,
    )


def source_replay_max_abs(scorer: FrozenPhiRLScorer, source_result: Any) -> float:
    """Compare fixed point scoring with every source-computed control value."""

    if source_result.emergence is None:
        raise ValueError("source result lacks emergence values")
    differences = []
    for index, expected in enumerate(source_result.emergence):
        observed = scorer.score_reduced_pair(
            scorer.reference_reduced[:, index],
            scorer.reference_reduced[:, index + 1],
        )["emergence"]
        differences.append(abs(observed - float(expected)))
    return float(max(differences, default=0.0))
