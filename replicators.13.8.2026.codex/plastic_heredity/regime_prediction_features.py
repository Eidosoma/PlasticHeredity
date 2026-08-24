"""Prospective, permutation-invariant features for strict-regime prediction.

This module is intentionally separate from the sealed regime-confirmation
implementation.  It adds local dynamical summaries without changing any of
the previously registered feature definitions or result bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .config import CANDIDATES, ExperimentConfig
from .features import (
    SUMMARY_NAMES,
    _summarize_profile,
    history_features,
    state_graph_features,
)
from .mechanistic_features import H8_INDICES
from .mechanistic_v2_features import (
    FEATURE_NAMES as V2_FEATURE_NAMES,
    H10_SPECS,
    INTERACTION_INDICES,
    STATE_ONLY_INDICES,
    STATE_ONLY_SPECS,
    FeatureProvenance,
    FeatureSpec,
    comprehensive_beta_features,
)
from .simulator import FloatMatrix, Snapshot

FloatArray = NDArray[np.float64]


class PredictionCaseLike(Protocol):
    candidate: str
    matrix_id: int
    beta: FloatMatrix
    snapshot: Snapshot
    previous_composition: NDArray[np.int64] | None


DYNAMIC_PROFILE_NAMES = (
    "expected_join",
    "expected_leave",
    "simplex_tangent_drift",
    "absolute_simplex_tangent_drift",
    "event_variance",
    "standardized_tangent_drift",
    "log_join_leave_ratio",
)

DYNAMIC_GLOBAL_NAMES = (
    "expected_mass_drift",
    "total_event_variance",
    "tangent_drift_l1",
    "tangent_drift_l2",
    "drift_noise_l2",
    "entropy_derivative",
    "concentration_derivative",
    "jacobian_maximum_real_eigenvalue",
    "jacobian_spectral_radius",
    "jacobian_stable_fraction",
    "jacobian_largest_singular_value",
    "velocity_available",
    "velocity_l1",
    "velocity_l2",
    "velocity_drift_cosine",
)

DYNAMIC_FEATURE_NAMES = tuple(
    f"dynamics_{profile}__{summary}"
    for profile in DYNAMIC_PROFILE_NAMES
    for summary in SUMMARY_NAMES
) + tuple(f"dynamics_global__{name}" for name in DYNAMIC_GLOBAL_NAMES)


def _dynamic_provenance(name: str) -> FeatureProvenance:
    if name.endswith("velocity_drift_cosine"):
        return FeatureProvenance(
            depends_on_state=True, depends_on_beta=True, depends_on_history=True
        )
    if "velocity_" in name:
        return FeatureProvenance(depends_on_state=True, depends_on_history=True)
    return FeatureProvenance(depends_on_state=True, depends_on_beta=True)


DYNAMIC_FEATURE_SPECS = tuple(
    FeatureSpec(name, _dynamic_provenance(name)) for name in DYNAMIC_FEATURE_NAMES
)


def _safe_cosine(left: FloatArray, right: FloatArray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _local_rates(
    composition: FloatArray,
    beta: FloatMatrix,
    join_scale: float,
    leave_scale: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    x = np.asarray(composition, dtype=np.float64)
    mass = float(x.sum())
    if mass <= 0.0:
        raise ValueError("local dynamics are undefined for an empty assembly")
    probability = x / mass
    incoming = np.asarray(beta, dtype=np.float64) @ probability
    boost = 1.0 + incoming
    join = join_scale * mass * boost
    leave = leave_scale * x * boost
    drift = join - leave
    tangent = drift - probability * float(drift.sum())
    return probability, incoming, boost, join, leave, tangent


def _tangent_jacobian(
    composition: FloatArray,
    beta: FloatMatrix,
    join_scale: float,
    leave_scale: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return rates, tangent drift, and its analytic Jacobian.

    The expression differentiates the same expected Poisson rates used by
    ``simulator._grow_to_fission`` at the restored post-fission state.  It is
    deterministic and does not approximate a future trajectory.
    """

    x = np.asarray(composition, dtype=np.float64)
    mass = float(x.sum())
    if mass <= 0.0:
        raise ValueError("local dynamics are undefined for an empty assembly")
    probability, incoming, boost, join, leave, tangent = _local_rates(
        x, beta, join_scale, leave_scale
    )
    drift = join - leave
    total_drift = float(drift.sum())

    # d(join_i)/d(x_j) = join_scale * (1 + beta_ij).
    join_jacobian = join_scale * (1.0 + beta)
    incoming_derivative = (beta - incoming[:, None]) / mass
    leave_jacobian = leave_scale * (np.diag(boost) + x[:, None] * incoming_derivative)
    drift_jacobian = join_jacobian - leave_jacobian
    probability_jacobian = (
        np.eye(x.size, dtype=np.float64) - probability[:, None]
    ) / mass
    tangent_jacobian = (
        drift_jacobian
        - probability_jacobian * total_drift
        - np.outer(probability, drift_jacobian.sum(axis=0))
    )
    return join, leave, drift, tangent, tangent_jacobian


def local_dynamics_features(
    composition: NDArray,
    beta: FloatMatrix,
    experiment: ExperimentConfig,
    candidate: str,
    previous_composition: NDArray | None = None,
) -> FloatArray:
    """Compute the frozen invariant local-rate/stability feature panel."""

    if candidate not in CANDIDATES:
        raise ValueError(f"unknown simulator candidate: {candidate}")
    config = experiment.gard
    contract = CANDIDATES[candidate]
    x = np.asarray(composition, dtype=np.float64)
    if x.shape != (config.n_types,):
        raise ValueError("composition shape does not match the GARD contract")
    mass = float(x.sum())
    if mass <= 0.0:
        raise ValueError("local dynamics are undefined for an empty assembly")
    probability = x / mass
    active = x > 0.0
    join_scale = config.k_join * (1.0 / config.n_types) * contract.poisson_exposure
    leave_scale = config.k_leave * contract.poisson_exposure
    join, leave, drift, tangent, jacobian = _tangent_jacobian(
        x, beta, join_scale, leave_scale
    )
    variance = join + leave
    standardized = tangent / np.sqrt(variance + np.finfo(np.float64).eps)
    log_ratio = np.log((join + np.finfo(np.float64).tiny) / (leave + 1e-12))
    profiles = np.column_stack(
        (join, leave, tangent, np.abs(tangent), variance, standardized, log_ratio)
    )
    summarized = np.concatenate(
        [
            _summarize_profile(profiles[:, index], probability, active)
            for index in range(profiles.shape[1])
        ]
    )

    tiny = np.finfo(np.float64).tiny
    fraction_drift = tangent / mass
    entropy_derivative = -float(
        np.dot(fraction_drift, np.log(np.maximum(probability, tiny)) + 1.0)
    )
    concentration_derivative = 2.0 * float(np.dot(probability, fraction_drift))
    eigenvalues = np.linalg.eigvals(jacobian)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)

    velocity_available = previous_composition is not None
    if previous_composition is None:
        velocity = np.zeros_like(probability)
    else:
        previous = np.asarray(previous_composition, dtype=np.float64)
        if previous.shape != x.shape or previous.sum() <= 0.0:
            raise ValueError("invalid previous composition for velocity feature")
        velocity = probability - previous / previous.sum()

    globals_ = np.asarray(
        (
            drift.sum(),
            variance.sum(),
            np.linalg.norm(tangent, ord=1),
            np.linalg.norm(tangent),
            np.linalg.norm(tangent) / np.sqrt(variance.sum() + tiny),
            entropy_derivative,
            concentration_derivative,
            np.max(eigenvalues.real),
            np.max(np.abs(eigenvalues)),
            # Tangent conservation creates a structural zero mode.  Classify
            # stability with a fixed numerical dead zone so its sign cannot
            # flip under an otherwise exact permutation-similarity transform.
            np.mean(eigenvalues.real < -1e-12),
            singular_values[0],
            float(velocity_available),
            np.linalg.norm(velocity, ord=1),
            np.linalg.norm(velocity),
            _safe_cosine(velocity, fraction_drift),
        ),
        dtype=np.float64,
    )
    values = np.concatenate((summarized, globals_))
    if values.shape != (len(DYNAMIC_FEATURE_NAMES),):
        raise AssertionError("dynamic feature name/value length mismatch")
    if not np.isfinite(values).all():
        raise ValueError("local-dynamics features contain non-finite values")
    return values


@dataclass(frozen=True)
class PredictionRawFeatures:
    h10: FloatArray
    state: FloatArray
    beta: FloatArray
    interaction: FloatArray
    dynamics: FloatArray

    def selected(self, indices: NDArray[np.bool_]) -> "PredictionRawFeatures":
        return PredictionRawFeatures(
            **{
                name: np.asarray(getattr(self, name))[indices]
                for name in ("h10", "state", "beta", "interaction", "dynamics")
            }
        )

    def block(self, name: str) -> FloatArray:
        return np.asarray(getattr(self, name), dtype=np.float64)


PREDICTION_FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    **V2_FEATURE_NAMES,
    "dynamics": DYNAMIC_FEATURE_NAMES,
}


def _h10(snapshot: Snapshot, experiment: ExperimentConfig) -> FloatArray:
    legacy = history_features(snapshot, experiment.gard)
    clocks = np.asarray(
        (
            snapshot.previous_growth_steps / max(experiment.gard.max_growth_steps, 1),
            snapshot.cumulative_growth_steps
            / max(experiment.gard.generations * experiment.gard.max_growth_steps, 1),
        ),
        dtype=np.float64,
    )
    return np.concatenate((legacy[list(H8_INDICES)], clocks))


def compact_post_break_features(
    snapshot: Snapshot,
    beta: FloatMatrix,
    experiment: ExperimentConfig,
    candidate: str,
    previous_composition: NDArray | None,
) -> FloatArray:
    # This branch-level secondary is evaluated up to 256,000 times.  Use the
    # exact 26 composition-only coordinates without repeatedly recomputing the
    # much larger beta catalogue or the 100x100 stability eigensystem.
    x = np.asarray(snapshot.composition, dtype=np.float64)
    probability = x / x.sum()
    active = x > 0.0
    state_only = np.concatenate(
        (
            _summarize_profile(probability, probability, active),
            _summarize_profile(active.astype(np.float64), probability, active),
        )
    )
    contract = CANDIDATES[candidate]
    join_scale = (
        experiment.gard.k_join
        * (1.0 / experiment.gard.n_types)
        * contract.poisson_exposure
    )
    leave_scale = experiment.gard.k_leave * contract.poisson_exposure
    _, _, _, join, leave, tangent = _local_rates(x, beta, join_scale, leave_scale)
    drift = join - leave
    variance = join + leave
    if previous_composition is None:
        velocity = np.zeros_like(probability)
    else:
        previous = np.asarray(previous_composition, dtype=np.float64)
        velocity = probability - previous / previous.sum()
    cheap_dynamics = np.asarray(
        (
            drift.sum(),
            variance.sum(),
            np.linalg.norm(tangent, ord=1),
            np.linalg.norm(tangent),
            np.linalg.norm(tangent)
            / np.sqrt(variance.sum() + np.finfo(np.float64).tiny),
            np.linalg.norm(velocity, ord=1),
            np.linalg.norm(velocity),
            _safe_cosine(velocity, tangent / x.sum()),
        ),
        dtype=np.float64,
    )
    return np.concatenate(
        (
            _h10(snapshot, experiment),
            state_only,
            cheap_dynamics,
        )
    )


POST_BREAK_FEATURE_NAMES = (
    V2_FEATURE_NAMES["h10"]
    + V2_FEATURE_NAMES["state"]
    + (
        "postbreak_expected_mass_drift",
        "postbreak_total_event_variance",
        "postbreak_tangent_l1",
        "postbreak_tangent_l2",
        "postbreak_drift_noise_l2",
        "postbreak_velocity_l1",
        "postbreak_velocity_l2",
        "postbreak_velocity_drift_cosine",
    )
)

POST_BREAK_FEATURE_SPECS = (
    H10_SPECS
    + STATE_ONLY_SPECS
    + (
        FeatureSpec(
            "postbreak_expected_mass_drift",
            FeatureProvenance(depends_on_state=True, depends_on_beta=True),
        ),
        FeatureSpec(
            "postbreak_total_event_variance",
            FeatureProvenance(depends_on_state=True, depends_on_beta=True),
        ),
        FeatureSpec(
            "postbreak_tangent_l1",
            FeatureProvenance(depends_on_state=True, depends_on_beta=True),
        ),
        FeatureSpec(
            "postbreak_tangent_l2",
            FeatureProvenance(depends_on_state=True, depends_on_beta=True),
        ),
        FeatureSpec(
            "postbreak_drift_noise_l2",
            FeatureProvenance(depends_on_state=True, depends_on_beta=True),
        ),
        FeatureSpec(
            "postbreak_velocity_l1",
            FeatureProvenance(depends_on_state=True, depends_on_history=True),
        ),
        FeatureSpec(
            "postbreak_velocity_l2",
            FeatureProvenance(depends_on_state=True, depends_on_history=True),
        ),
        FeatureSpec(
            "postbreak_velocity_drift_cosine",
            FeatureProvenance(
                depends_on_state=True, depends_on_beta=True, depends_on_history=True
            ),
        ),
    )
)

if tuple(spec.name for spec in POST_BREAK_FEATURE_SPECS) != POST_BREAK_FEATURE_NAMES:
    raise AssertionError("post-break feature provenance/name mismatch")


def extract_prediction_features(
    cases: list[PredictionCaseLike], experiment: ExperimentConfig
) -> PredictionRawFeatures:
    full_state = np.vstack(
        [
            state_graph_features(case.snapshot.composition, case.beta, experiment.gard)
            for case in cases
        ]
    )
    beta_cache: dict[tuple[str, int], FloatArray] = {}
    beta_rows: list[FloatArray] = []
    for case in cases:
        # Cohort is deliberately not part of the key: one extraction call never
        # mixes seed domains, and beta is shared across candidates/landmarks.
        key = ("matrix", int(case.matrix_id))
        if key not in beta_cache:
            beta_cache[key] = comprehensive_beta_features(case.beta, experiment.gard)
        beta_rows.append(beta_cache[key])
    values = PredictionRawFeatures(
        h10=np.vstack([_h10(case.snapshot, experiment) for case in cases]),
        state=full_state[:, list(STATE_ONLY_INDICES)],
        beta=np.vstack(beta_rows),
        interaction=full_state[:, list(INTERACTION_INDICES)],
        dynamics=np.vstack(
            [
                local_dynamics_features(
                    case.snapshot.composition,
                    case.beta,
                    experiment,
                    case.candidate,
                    getattr(case, "previous_composition", None),
                )
                for case in cases
            ]
        ),
    )
    for block, names in PREDICTION_FEATURE_NAMES.items():
        actual = values.block(block)
        if actual.shape != (len(cases), len(names)) or not np.isfinite(actual).all():
            raise ValueError(f"invalid prediction {block} feature block")
    return values


def prediction_provenance_contract() -> dict[str, list[dict[str, object]]]:
    from .mechanistic_v2_features import FEATURE_SPECS

    contract = {
        block: [spec.to_dict() for spec in specs]
        for block, specs in FEATURE_SPECS.items()
    }
    contract["dynamics"] = [spec.to_dict() for spec in DYNAMIC_FEATURE_SPECS]
    contract["post_break"] = [spec.to_dict() for spec in POST_BREAK_FEATURE_SPECS]
    return contract
