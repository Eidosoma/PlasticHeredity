"""Configuration with paper-reported values and explicit reconstruction choices."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class GardConfig:
    """GARD growth-fission parameters.

    The first seven fields are stated in the preprint. The kinetic constants
    are standard values from the published GARD literature because the
    preprint does not report them. ``tau`` is one Poisson tau-leap.
    """

    n_types: int = 100
    initial_size: int = 40
    beta_log_mean: float = -4.0
    beta_log_sigma: float = 4.0
    generations: int = 100
    max_size: int = 80
    max_steps_per_generation: int = 1000
    forward_rate: float = 1e-2
    backward_rate: float = 1e-4
    environment_concentration: float = 1e-2
    # The leap duration is not reported. A value of 0.5 reproduces the
    # approximately 800-1,300 molecular steps visible in the paper's figures.
    tau: float = 0.5
    # The preprint does not state whether updates with zero sampled events
    # advance the recorded molecular-time trajectory.
    record_zero_event_steps: bool = True

    def validate(self) -> None:
        if self.n_types < 2:
            raise ValueError("n_types must be at least 2")
        if not 0 < self.initial_size <= self.max_size:
            raise ValueError("initial_size must be in (0, max_size]")
        if self.initial_size > self.n_types:
            raise ValueError(
                "initial_size cannot exceed n_types when sampling without replacement"
            )
        if self.beta_log_sigma < 0:
            raise ValueError("beta_log_sigma must be non-negative")
        if self.generations < 1 or self.max_steps_per_generation < 1:
            raise ValueError("generations and max_steps_per_generation must be positive")
        if min(
            self.forward_rate,
            self.backward_rate,
            self.environment_concentration,
            self.tau,
        ) <= 0:
            raise ValueError("kinetic constants and tau must be positive")


@dataclass(frozen=True)
class CausalConfig:
    """Preprocessing and local Gaussian causal-emergence choices."""

    lag: int = 1
    pseudocount: float = 0.5
    drop_last_clr_component: bool = True
    partition_cut: str = "zero"
    covariance_ridge: float = 1e-8
    measure: str = "wms"

    def validate(self) -> None:
        if self.lag < 1:
            raise ValueError("lag must be positive")
        if self.pseudocount <= 0:
            raise ValueError("pseudocount must be positive")
        if self.partition_cut not in {"zero", "median"}:
            raise ValueError("partition_cut must be 'zero' or 'median'")
        if self.covariance_ridge <= 0:
            raise ValueError("covariance_ridge must be positive")
        if self.measure not in {"wms", "mmi_synergy"}:
            raise ValueError("measure must be 'wms' or 'mmi_synergy'")


@dataclass(frozen=True)
class ReplicatorConfig:
    """Dominant-compotype reconstruction of the paper's binary state."""

    similarity_threshold: float = 0.95
    min_recurrences: int = 3
    reference_states: str = "generation_end"
    similarity_metric: str = "cosine"
    reference_method: str = "medoid"

    def validate(self) -> None:
        if not 0 < self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if self.min_recurrences < 2:
            raise ValueError("min_recurrences must be at least 2")
        if self.reference_states not in {"generation_end", "all"}:
            raise ValueError("reference_states must be 'generation_end' or 'all'")
        if self.similarity_metric not in {"cosine", "euclidean"}:
            raise ValueError("similarity_metric must be 'cosine' or 'euclidean'")
        if self.reference_method not in {"medoid", "neighbor_centroid"}:
            raise ValueError(
                "reference_method must be 'medoid' or 'neighbor_centroid'"
            )


@dataclass(frozen=True)
class InterventionConfig:
    """How the local Phi-r score is estimated at intervention time."""

    estimator: str = "online_initial"

    def validate(self) -> None:
        if self.estimator not in {
            "online_initial",
            "online_history",
            "matched_control",
        }:
            raise ValueError(
                "intervention estimator must be 'online_initial', "
                "'online_history', or 'matched_control'"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    runs: int = 100
    base_seed: int = 1729
    bootstrap_repetitions: int = 10
    test_fraction: float = 0.2
    forecast_input_fraction: float = 0.25
    forecast_grid_points: int = 128
    gard: GardConfig = field(default_factory=GardConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    replicator: ReplicatorConfig = field(default_factory=ReplicatorConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)

    def validate(self) -> None:
        if self.runs < 2:
            raise ValueError("runs must be at least 2")
        if self.bootstrap_repetitions < 1:
            raise ValueError("bootstrap_repetitions must be positive")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in (0, 1)")
        if not 0 < self.forecast_input_fraction < 1:
            raise ValueError("forecast_input_fraction must be in (0, 1)")
        if self.forecast_grid_points < 8:
            raise ValueError("forecast_grid_points must be at least 8")
        self.gard.validate()
        self.causal.validate()
        self.replicator.validate()
        self.intervention.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
