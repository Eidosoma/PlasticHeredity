"""Provenance-driven features for the beta-completeness correction.

The v1 mechanistic ablation used hand-selected profile families and compressed
each added block to twelve PCs.  This module makes every dependency explicit
and provides a threshold-free, state-invariant beta catalogue for the v2
prospective correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .config import ExperimentConfig, GardConfig
from .experiment import StateCase
from .features import (
    HISTORY_FEATURE_NAMES,
    STATE_GRAPH_FEATURE_NAMES,
    SUMMARY_NAMES,
    beta_only_features,
    history_features,
    state_graph_features,
)
from .mechanistic_features import H8_INDICES
from .simulator import FloatMatrix

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FeatureProvenance:
    """Declared inputs that can change a feature."""

    depends_on_state: bool = False
    depends_on_beta: bool = False
    depends_on_clock: bool = False
    depends_on_history: bool = False
    depends_on_mass: bool = False
    depends_on_phase: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    provenance: FeatureProvenance

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "provenance": self.provenance.to_dict()}


def _merge_provenance(*items: FeatureProvenance) -> FeatureProvenance:
    fields = FeatureProvenance.__dataclass_fields__
    return FeatureProvenance(
        **{name: any(getattr(item, name) for item in items) for name in fields}
    )


PROFILE_PROVENANCE: dict[str, FeatureProvenance] = {
    "count_over_nmax": FeatureProvenance(depends_on_state=True, depends_on_mass=True),
    "composition_fraction": FeatureProvenance(depends_on_state=True),
    "present": FeatureProvenance(depends_on_state=True),
    "log_in_catalysis": FeatureProvenance(depends_on_state=True, depends_on_beta=True),
    "log_out_catalysis": FeatureProvenance(depends_on_state=True, depends_on_beta=True),
    "log_join_propensity": FeatureProvenance(
        depends_on_state=True, depends_on_beta=True, depends_on_mass=True
    ),
    "log_leave_propensity": FeatureProvenance(
        depends_on_state=True, depends_on_beta=True, depends_on_mass=True
    ),
    "row_log_beta_mean": FeatureProvenance(depends_on_beta=True),
    "row_log_beta_sd": FeatureProvenance(depends_on_beta=True),
    "column_log_beta_mean": FeatureProvenance(depends_on_beta=True),
    "column_log_beta_sd": FeatureProvenance(depends_on_beta=True),
    "log_row_beta_mean": FeatureProvenance(depends_on_beta=True),
    "log_column_beta_mean": FeatureProvenance(depends_on_beta=True),
    "log_active_in_catalysis": FeatureProvenance(
        depends_on_state=True, depends_on_beta=True
    ),
    "log_active_out_catalysis": FeatureProvenance(
        depends_on_state=True, depends_on_beta=True
    ),
}


def _state_graph_specs() -> tuple[FeatureSpec, ...]:
    specs: list[FeatureSpec] = []
    state_weighted = FeatureProvenance(depends_on_state=True)
    for name in STATE_GRAPH_FEATURE_NAMES:
        profile, summary = name.split("__", 1)
        provenance = PROFILE_PROVENANCE[profile]
        if summary in {"composition_weighted_mean", "active_mean"}:
            provenance = _merge_provenance(provenance, state_weighted)
        specs.append(FeatureSpec(name, provenance))
    return tuple(specs)


STATE_GRAPH_FEATURE_SPECS = _state_graph_specs()


def _only(spec: FeatureSpec, dependency: str) -> bool:
    values = spec.provenance.to_dict()
    return bool(values[dependency]) and sum(values.values()) == 1


STATE_ONLY_SPECS = tuple(
    spec for spec in STATE_GRAPH_FEATURE_SPECS if _only(spec, "depends_on_state")
)
INTERACTION_SPECS = tuple(
    spec
    for spec in STATE_GRAPH_FEATURE_SPECS
    if spec.provenance.depends_on_state
    and spec.provenance.depends_on_beta
    and not spec.provenance.depends_on_clock
    and not spec.provenance.depends_on_history
    and not spec.provenance.depends_on_mass
    and not spec.provenance.depends_on_phase
)

STATE_ONLY_INDICES = tuple(
    STATE_GRAPH_FEATURE_NAMES.index(spec.name) for spec in STATE_ONLY_SPECS
)
INTERACTION_INDICES = tuple(
    STATE_GRAPH_FEATURE_NAMES.index(spec.name) for spec in INTERACTION_SPECS
)


# Under the fixed uniform pseudo-composition, the first three profile families
# are constants.  Every other legacy pseudo-state coordinate depends only on
# beta; state-weighted summaries also become beta-only because the weights are
# fixed and uniform.
_CONSTANT_UNIFORM_PROFILES = {
    "count_over_nmax",
    "composition_fraction",
    "present",
}
LEGACY_BETA_ONLY_SPECS = tuple(
    FeatureSpec(
        f"legacy_beta__{name}", FeatureProvenance(depends_on_beta=True)
    )
    for name in STATE_GRAPH_FEATURE_NAMES
    if name.split("__", 1)[0] not in _CONSTANT_UNIFORM_PROFILES
)
LEGACY_BETA_ONLY_INDICES = tuple(
    STATE_GRAPH_FEATURE_NAMES.index(spec.name.removeprefix("legacy_beta__"))
    for spec in LEGACY_BETA_ONLY_SPECS
)

BETA_SUMMARY_NAMES = (
    "mean",
    "sd",
    "minimum",
    "q05",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "q95",
    "maximum",
)
BETA_SUMMARY_FAMILIES = (
    "raw_entries",
    "log_entries",
    "log_row_strength",
    "log_column_strength",
)
BETA_DESCRIPTOR_NAMES = tuple(
    f"beta_{family}__{summary}"
    for family in BETA_SUMMARY_FAMILIES
    for summary in BETA_SUMMARY_NAMES
) + (
    "beta_log_strength__row_column_correlation",
    "beta_log_entries__reciprocity_correlation",
    "beta_matrix__normalized_asymmetry",
) + tuple(f"beta_singular__normalized_{index:03d}" for index in range(1, 101)) + (
    "beta_singular__stable_rank",
    "beta_singular__normalized_spectral_entropy",
    "beta_row_strength__normalized_entropy",
    "beta_column_strength__normalized_entropy",
    "beta_row_strength__herfindahl",
    "beta_column_strength__herfindahl",
)
BETA_DESCRIPTOR_SPECS = tuple(
    FeatureSpec(name, FeatureProvenance(depends_on_beta=True))
    for name in BETA_DESCRIPTOR_NAMES
)
BETA_ONLY_SPECS = LEGACY_BETA_ONLY_SPECS + BETA_DESCRIPTOR_SPECS


H8_NAMES = tuple(HISTORY_FEATURE_NAMES[index] for index in H8_INDICES)
H10_NAMES = H8_NAMES + (
    "normalized_previous_growth_steps",
    "normalized_cumulative_growth_steps",
)
H10_SPECS = (
    FeatureSpec("normalized_generation", FeatureProvenance(depends_on_phase=True)),
    FeatureSpec(
        "normalized_current_mass",
        FeatureProvenance(depends_on_state=True, depends_on_mass=True),
    ),
) + tuple(
    FeatureSpec(name, FeatureProvenance(depends_on_history=True))
    for name in H8_NAMES[2:]
) + (
    FeatureSpec(
        "normalized_previous_growth_steps",
        FeatureProvenance(depends_on_clock=True, depends_on_history=True),
    ),
    FeatureSpec(
        "normalized_cumulative_growth_steps",
        FeatureProvenance(depends_on_clock=True, depends_on_history=True),
    ),
)


FEATURE_SPECS: dict[str, tuple[FeatureSpec, ...]] = {
    "h10": H10_SPECS,
    "state": STATE_ONLY_SPECS,
    "beta": BETA_ONLY_SPECS,
    "interaction": INTERACTION_SPECS,
}
FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    block: tuple(spec.name for spec in specs) for block, specs in FEATURE_SPECS.items()
}


def _summary(values: FloatArray) -> FloatArray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = np.quantile(values, (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95))
    return np.asarray(
        (
            values.mean(),
            values.std(),
            values.min(),
            *quantiles,
            values.max(),
        ),
        dtype=np.float64,
    )


def _safe_correlation(left: FloatArray, right: FloatArray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 2:
        raise ValueError("correlation inputs must have equal length >= 2")
    if left.std() <= np.finfo(np.float64).eps or right.std() <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _normalized_entropy(probability: FloatArray) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    positive = probability[probability > 0.0]
    if probability.size <= 1 or positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)) / np.log(probability.size))


def comprehensive_beta_descriptors(beta: FloatMatrix, config: GardConfig) -> FloatArray:
    """Return the frozen threshold-free beta descriptor panel."""

    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape != (100, 100) or config.n_types != 100:
        raise ValueError("the v2 beta contract requires a 100 x 100 catalytic matrix")
    if not np.isfinite(beta).all() or np.any(beta <= 0.0):
        raise ValueError("beta must be finite and strictly positive")
    tiny = np.finfo(np.float64).tiny
    log_beta = np.log(np.maximum(beta, tiny))
    row_strength = beta.sum(axis=1)
    column_strength = beta.sum(axis=0)
    log_row_strength = np.log(np.maximum(row_strength, tiny))
    log_column_strength = np.log(np.maximum(column_strength, tiny))
    off_diagonal = ~np.eye(config.n_types, dtype=bool)
    frobenius = float(np.linalg.norm(beta, ord="fro"))
    singular = np.linalg.svd(beta, compute_uv=False)
    normalized_singular = singular / frobenius
    spectral_probability = singular**2 / np.sum(singular**2)
    total_strength = float(row_strength.sum())
    row_probability = row_strength / total_strength
    column_probability = column_strength / total_strength
    descriptors = np.concatenate(
        (
            _summary(beta),
            _summary(log_beta),
            _summary(log_row_strength),
            _summary(log_column_strength),
            np.asarray(
                (
                    _safe_correlation(log_row_strength, log_column_strength),
                    _safe_correlation(
                        log_beta[off_diagonal], log_beta.T[off_diagonal]
                    ),
                    np.linalg.norm(beta - beta.T, ord="fro") / frobenius,
                ),
                dtype=np.float64,
            ),
            normalized_singular,
            np.asarray(
                (
                    np.sum(singular**2) / singular[0] ** 2,
                    _normalized_entropy(spectral_probability),
                    _normalized_entropy(row_probability),
                    _normalized_entropy(column_probability),
                    np.sum(row_probability**2),
                    np.sum(column_probability**2),
                ),
                dtype=np.float64,
            ),
        )
    )
    if descriptors.shape != (len(BETA_DESCRIPTOR_NAMES),):
        raise AssertionError("beta descriptor name/value length mismatch")
    if not np.isfinite(descriptors).all():
        raise ValueError("beta descriptors contain non-finite values")
    return descriptors


def comprehensive_beta_features(beta: FloatMatrix, config: GardConfig) -> FloatArray:
    legacy = beta_only_features(beta, config)[list(LEGACY_BETA_ONLY_INDICES)]
    values = np.concatenate((legacy, comprehensive_beta_descriptors(beta, config)))
    if values.shape != (len(BETA_ONLY_SPECS),):
        raise AssertionError("comprehensive beta name/value length mismatch")
    return values


@dataclass(frozen=True)
class MechanisticV2RawFeatures:
    h10: FloatArray
    state: FloatArray
    beta: FloatArray
    interaction: FloatArray

    def selected(self, indices: NDArray[np.bool_]) -> "MechanisticV2RawFeatures":
        return MechanisticV2RawFeatures(
            h10=self.h10[indices],
            state=self.state[indices],
            beta=self.beta[indices],
            interaction=self.interaction[indices],
        )


def extract_mechanistic_v2_features(
    cases: list[StateCase], experiment: ExperimentConfig
) -> MechanisticV2RawFeatures:
    legacy_history = np.vstack(
        [history_features(case.snapshot, experiment.gard) for case in cases]
    )
    h8 = legacy_history[:, H8_INDICES]
    clocks = np.asarray(
        [
            (
                case.snapshot.previous_growth_steps
                / max(experiment.gard.max_growth_steps, 1),
                case.snapshot.cumulative_growth_steps
                / max(experiment.gard.generations * experiment.gard.max_growth_steps, 1),
            )
            for case in cases
        ],
        dtype=np.float64,
    )
    h10 = np.column_stack((h8, clocks))
    full_state = np.vstack(
        [
            state_graph_features(case.snapshot.composition, case.beta, experiment.gard)
            for case in cases
        ]
    )

    beta_cache: dict[tuple[str, int], FloatArray] = {}
    beta_rows: list[FloatArray] = []
    for case in cases:
        key = (case.cohort, case.matrix_id)
        if key not in beta_cache:
            beta_cache[key] = comprehensive_beta_features(case.beta, experiment.gard)
        beta_rows.append(beta_cache[key])
    beta_values = np.vstack(beta_rows)

    values = MechanisticV2RawFeatures(
        h10=h10,
        state=full_state[:, STATE_ONLY_INDICES],
        beta=beta_values,
        interaction=full_state[:, INTERACTION_INDICES],
    )
    for block, names in FEATURE_NAMES.items():
        actual = getattr(values, block)
        if actual.shape != (len(cases), len(names)) or not np.isfinite(actual).all():
            raise ValueError(f"invalid v2 {block} feature block")
    return values


def provenance_contract() -> dict[str, list[dict[str, object]]]:
    return {
        block: [spec.to_dict() for spec in specs]
        for block, specs in FEATURE_SPECS.items()
    }


if set(SUMMARY_NAMES) != {
    "mean",
    "sd",
    "minimum",
    "q05",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "q95",
    "maximum",
    "composition_weighted_mean",
    "active_mean",
}:
    raise AssertionError("upstream summary catalogue changed")
