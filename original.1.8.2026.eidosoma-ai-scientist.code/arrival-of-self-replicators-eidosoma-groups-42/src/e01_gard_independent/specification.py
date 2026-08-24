"""Fail-closed specifications for the independent E01 GARD engine.

The registry remains non-executable.  A :class:`GardSpecification` is therefore
an explicit, versioned branch instance for a bounded run or validation case; it
is never inferred from unresolved registry values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from math import isclose
from typing import Any

import numpy as np


class SpecificationError(ValueError):
    """An independent-engine specification is incomplete or inconsistent."""


class ProfileRole(StrEnum):
    """Evidence boundary attached to a complete specification."""

    HISTORICAL_DISTRIBUTION_COMPARISON = "historical_distribution_comparison"
    PAPER_RECONSTRUCTION_FIXTURE = "paper_reconstruction_fixture"
    MODERN_GILLESPIE_FIXTURE = "modern_gillespie_fixture"


class PropensityEquationBranch(StrEnum):
    """Versioned interpretation of the frozen join/leave equations."""

    HISTORICAL_REFERENCE = "historical_reference"
    PAPER_POISSON_RECONSTRUCTION = "paper_poisson_reconstruction"
    MODERN_GILLESPIE = "modern_gillespie"


class CatalyticMatrixBranch(StrEnum):
    """Orientation and diagonal branch for the catalytic matrix."""

    HISTORICAL_ORIENTATION_WITH_DIAGONAL = "historical_orientation_with_diagonal"
    TRANSPOSED_WITH_DIAGONAL = "transposed_with_diagonal"
    HISTORICAL_ORIENTATION_ZERO_DIAGONAL = "historical_orientation_zero_diagonal"


class ReservoirSemantics(StrEnum):
    """How an explicitly supplied reservoir vector is interpreted."""

    CONSTANT_REQUIRE_SUM_ONE = "constant_require_sum_one"
    CONSTANT_AS_SUPPLIED = "constant_as_supplied"


class UpdateKernel(StrEnum):
    """Stochastic update family."""

    CATEGORICAL_SINGLE_EVENT = "categorical_single_event"
    DIRECT_GILLESPIE = "direct_gillespie"
    VECTOR_POISSON_BATCH = "vector_poisson_batch"


class ClockSemantics(StrEnum):
    """Meaning of the time-like field in event logs."""

    EVENT_INDEX_ONLY = "event_index_only"
    GILLESPIE_EXPONENTIAL = "gillespie_exponential"
    FIXED_POISSON_EXPOSURE = "fixed_poisson_exposure"


class LossNonnegativity(StrEnum):
    """Explicit handling of losses at the nonnegative boundary."""

    EVENTWISE_ZERO_RATE = "eventwise_zero_rate"
    CLIP_BATCH_TO_AVAILABLE = "clip_batch_to_available"
    ERROR_ON_BATCH_EXCESS = "error_on_batch_excess"


class GrowthBoundary(StrEnum):
    """Explicit handling when an update meets or crosses ``n_max``."""

    EVENTWISE_EXACT_STOP = "eventwise_exact_stop"
    RETAIN_BATCH_OVERSHOOT = "retain_batch_overshoot"
    REJECT_BATCH_OVERSHOOT = "reject_batch_overshoot"


class MaxStepsSemantics(StrEnum):
    """What a bounded growth phase does at its event/update limit."""

    UNBOUNDED_HISTORICAL_COMPARISON = "unbounded_historical_comparison"
    FISSION_CURRENT_STATE = "fission_current_state"
    STOP_WITHOUT_FISSION = "stop_without_fission"
    RAISE = "raise"


class ZeroPropensitySemantics(StrEnum):
    """Explicit handling of a nonempty state with zero total propensity."""

    STOP = "stop"
    RAISE = "raise"


class FissionSemantics(StrEnum):
    """Versioned fission family."""

    FIXED_SIZE_WITHOUT_REPLACEMENT_ODD_DISCARD = (
        "fixed_size_without_replacement_odd_discard"
    )
    BINOMIAL_COMPLEMENT = "binomial_complement"


class DaughterSelection(StrEnum):
    """Which returned daughter continues the single lineage."""

    FIRST = "first"
    SECOND = "second"
    UNIFORM_RANDOM = "uniform_random"


class PostFissionSemantics(StrEnum):
    """Handling of the selected daughter before the next generation."""

    CONTINUE_EXACT_SELECTED = "continue_exact_selected"
    ERROR_IF_SELECTED_EMPTY = "error_if_selected_empty"


class InitialStateSemantics(StrEnum):
    """Explicit initial-state construction branch."""

    DISTINCT_TYPES_WITHOUT_REPLACEMENT = "distinct_types_without_replacement"
    WITH_REPLACEMENT_COUNTS = "with_replacement_counts"


ENUM_FIELDS: dict[str, type[StrEnum]] = {
    "profile_role": ProfileRole,
    "propensity_equation_branch": PropensityEquationBranch,
    "catalytic_matrix_branch": CatalyticMatrixBranch,
    "reservoir_semantics": ReservoirSemantics,
    "update_kernel": UpdateKernel,
    "clock_semantics": ClockSemantics,
    "loss_nonnegativity": LossNonnegativity,
    "growth_boundary": GrowthBoundary,
    "max_steps_semantics": MaxStepsSemantics,
    "zero_propensity_semantics": ZeroPropensitySemantics,
    "fission_semantics": FissionSemantics,
    "daughter_selection": DaughterSelection,
    "post_fission_semantics": PostFissionSemantics,
    "initial_state_semantics": InitialStateSemantics,
}

FORBIDDEN_SENTINEL_PREFIXES = (
    "UNRESOLVED::",
    "CONFLICT::",
    "BRANCH_SET::",
    "UNAVAILABLE::",
)


@dataclass(frozen=True, slots=True)
class GardSpecification:
    """A complete branch instance with no model-defining defaults.

    Every field is mandatory at construction.  Values used for S05 fixtures are
    validation-specific and do not update or resolve the v0.3.0 registry.
    """

    specification_id: str
    profile_role: ProfileRole
    n_species: int
    n_min: int
    n_max: int
    n_generations: int
    max_steps: int | None
    beta_a: float
    beta_sigma: float
    k_f: float
    k_b: float
    rho: tuple[float, ...]
    propensity_equation_branch: PropensityEquationBranch
    catalytic_matrix_branch: CatalyticMatrixBranch
    reservoir_semantics: ReservoirSemantics
    update_kernel: UpdateKernel
    clock_semantics: ClockSemantics
    poisson_exposure: float | None
    loss_nonnegativity: LossNonnegativity
    growth_boundary: GrowthBoundary
    max_steps_semantics: MaxStepsSemantics
    zero_propensity_semantics: ZeroPropensitySemantics
    fission_semantics: FissionSemantics
    fission_probability: float | None
    daughter_selection: DaughterSelection
    post_fission_semantics: PostFissionSemantics
    initial_state_semantics: InitialStateSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.specification_id, str) or not self.specification_id:
            raise SpecificationError("specification_id must be a nonempty string.")
        if self.specification_id.startswith(FORBIDDEN_SENTINEL_PREFIXES):
            raise SpecificationError("specification_id cannot be a registry sentinel.")

        for field_name, enum_type in ENUM_FIELDS.items():
            value = getattr(self, field_name)
            if not isinstance(value, enum_type):
                raise SpecificationError(
                    f"{field_name} must be an explicit {enum_type.__name__} value."
                )

        for name in ("n_species", "n_min", "n_max", "n_generations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SpecificationError(f"{name} must be a positive integer.")
        if self.n_min > self.n_species and (
            self.initial_state_semantics
            is InitialStateSemantics.DISTINCT_TYPES_WITHOUT_REPLACEMENT
        ):
            raise SpecificationError(
                "Distinct initialization requires n_min <= n_species."
            )
        if self.n_min >= self.n_max:
            raise SpecificationError("n_min must be smaller than n_max.")

        numeric = {
            "beta_a": self.beta_a,
            "beta_sigma": self.beta_sigma,
            "k_f": self.k_f,
            "k_b": self.k_b,
        }
        for name, value in numeric.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise SpecificationError(f"{name} must be numeric.")
            if not np.isfinite(float(value)):
                raise SpecificationError(f"{name} must be finite.")
        if self.beta_sigma < 0 or self.k_f < 0 or self.k_b < 0:
            raise SpecificationError("beta_sigma, k_f, and k_b must be nonnegative.")

        try:
            reservoir = tuple(float(value) for value in self.rho)
        except (TypeError, ValueError) as exc:
            raise SpecificationError("rho must be a numeric sequence.") from exc
        object.__setattr__(self, "rho", reservoir)
        if len(reservoir) != self.n_species:
            raise SpecificationError(
                f"rho must contain exactly {self.n_species} entries."
            )
        if not all(np.isfinite(value) and value >= 0 for value in reservoir):
            raise SpecificationError("rho must be finite and nonnegative.")
        if (
            self.reservoir_semantics is ReservoirSemantics.CONSTANT_REQUIRE_SUM_ONE
            and not isclose(sum(reservoir), 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise SpecificationError("constant_require_sum_one requires sum(rho) == 1.")

        self._validate_kernel_combination()
        self._validate_growth_limit()
        self._validate_fission()
        self._validate_role()

    def _validate_kernel_combination(self) -> None:
        if self.update_kernel is UpdateKernel.CATEGORICAL_SINGLE_EVENT:
            required = (
                ClockSemantics.EVENT_INDEX_ONLY,
                LossNonnegativity.EVENTWISE_ZERO_RATE,
                GrowthBoundary.EVENTWISE_EXACT_STOP,
            )
        elif self.update_kernel is UpdateKernel.DIRECT_GILLESPIE:
            required = (
                ClockSemantics.GILLESPIE_EXPONENTIAL,
                LossNonnegativity.EVENTWISE_ZERO_RATE,
                GrowthBoundary.EVENTWISE_EXACT_STOP,
            )
        else:
            if self.clock_semantics is not ClockSemantics.FIXED_POISSON_EXPOSURE:
                raise SpecificationError(
                    "vector_poisson_batch requires fixed_poisson_exposure."
                )
            if self.loss_nonnegativity not in {
                LossNonnegativity.CLIP_BATCH_TO_AVAILABLE,
                LossNonnegativity.ERROR_ON_BATCH_EXCESS,
            }:
                raise SpecificationError(
                    "vector_poisson_batch requires an explicit batch loss rule."
                )
            if self.growth_boundary not in {
                GrowthBoundary.RETAIN_BATCH_OVERSHOOT,
                GrowthBoundary.REJECT_BATCH_OVERSHOOT,
            }:
                raise SpecificationError(
                    "vector_poisson_batch requires an explicit batch boundary rule."
                )
            if self.poisson_exposure is None or not np.isfinite(self.poisson_exposure):
                raise SpecificationError(
                    "vector_poisson_batch requires a finite poisson_exposure."
                )
            if self.poisson_exposure <= 0:
                raise SpecificationError("poisson_exposure must be positive.")
            return

        clock, loss, boundary = required
        if self.clock_semantics is not clock:
            raise SpecificationError(
                f"{self.update_kernel.value} requires {clock.value}."
            )
        if self.loss_nonnegativity is not loss:
            raise SpecificationError(
                f"{self.update_kernel.value} requires {loss.value}."
            )
        if self.growth_boundary is not boundary:
            raise SpecificationError(
                f"{self.update_kernel.value} requires {boundary.value}."
            )
        if self.poisson_exposure is not None:
            raise SpecificationError(
                "poisson_exposure must be null outside vector_poisson_batch."
            )

    def _validate_growth_limit(self) -> None:
        if (
            self.max_steps_semantics
            is MaxStepsSemantics.UNBOUNDED_HISTORICAL_COMPARISON
        ):
            if self.max_steps is not None:
                raise SpecificationError(
                    "The historical-comparison unbounded branch requires max_steps=null."
                )
            return
        if (
            not isinstance(self.max_steps, int)
            or isinstance(self.max_steps, bool)
            or self.max_steps <= 0
        ):
            raise SpecificationError(
                "Bounded max_steps semantics require a positive integer max_steps."
            )

    def _validate_fission(self) -> None:
        if self.fission_semantics is FissionSemantics.BINOMIAL_COMPLEMENT:
            if self.fission_probability is None or not np.isfinite(
                self.fission_probability
            ):
                raise SpecificationError(
                    "binomial_complement requires a finite fission_probability."
                )
            if not 0.0 <= self.fission_probability <= 1.0:
                raise SpecificationError("fission_probability must lie in [0, 1].")
        elif self.fission_probability is not None:
            raise SpecificationError(
                "Fixed-size fission requires fission_probability=null."
            )

    def _validate_role(self) -> None:
        expected = {
            ProfileRole.HISTORICAL_DISTRIBUTION_COMPARISON: (
                PropensityEquationBranch.HISTORICAL_REFERENCE,
                UpdateKernel.CATEGORICAL_SINGLE_EVENT,
            ),
            ProfileRole.PAPER_RECONSTRUCTION_FIXTURE: (
                PropensityEquationBranch.PAPER_POISSON_RECONSTRUCTION,
                UpdateKernel.VECTOR_POISSON_BATCH,
            ),
            ProfileRole.MODERN_GILLESPIE_FIXTURE: (
                PropensityEquationBranch.MODERN_GILLESPIE,
                UpdateKernel.DIRECT_GILLESPIE,
            ),
        }[self.profile_role]
        if (self.propensity_equation_branch, self.update_kernel) != expected:
            raise SpecificationError(
                f"{self.profile_role.value} requires branches "
                f"{expected[0].value}/{expected[1].value}."
            )


def specification_from_mapping(payload: Mapping[str, Any]) -> GardSpecification:
    """Construct a specification from a mapping and reject missing/extra fields."""

    expected = {item.name for item in fields(GardSpecification)}
    supplied = set(payload)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        raise SpecificationError(
            f"Specification fields mismatch; missing={missing}, extra={extra}."
        )
    converted = dict(payload)
    for key, value in converted.items():
        if isinstance(value, str) and value.startswith(FORBIDDEN_SENTINEL_PREFIXES):
            raise SpecificationError(f"{key} contains a forbidden registry sentinel.")
    for field_name, enum_type in ENUM_FIELDS.items():
        try:
            converted[field_name] = enum_type(converted[field_name])
        except (TypeError, ValueError) as exc:
            raise SpecificationError(
                f"Invalid explicit branch for {field_name}: {converted[field_name]!r}."
            ) from exc
    if not isinstance(converted["rho"], (list, tuple)):
        raise SpecificationError("rho must be an explicit list or tuple.")
    converted["rho"] = tuple(converted["rho"])
    return GardSpecification(**converted)
