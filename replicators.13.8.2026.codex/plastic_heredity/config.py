from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GardConfig:
    """Paper-stated GARD parameters and historical basal rates."""

    n_types: int = 100
    n_min: int = 40
    n_max: int = 80
    beta_log_mean: float = -4.0
    beta_log_sd: float = 4.0
    k_join: float = 1e-2
    k_leave: float = 1e-4
    max_growth_steps: int = 1_000
    generations: int = 100
    inheritance_threshold: float = 0.9


@dataclass(frozen=True)
class SimulationContract:
    """An explicit source-constrained resolution of unavailable simulator details.

    The paper-facing reconstruction named candidates 02 and 03 but did not publish
    their executable contracts. These independent contracts preserve the stated
    vector-Poisson, fission, daughter, and overshoot alternatives without claiming
    identity to the unavailable implementation.
    """

    name: str
    poisson_exposure: float
    overshoot_rule: str
    fission_rule: str
    daughter_rule: str


CANDIDATES: dict[str, SimulationContract] = {
    "02": SimulationContract(
        name="02",
        poisson_exposure=0.10,
        overshoot_rule="trim_whole_assembly",
        fission_rule="fixed_size",
        daughter_rule="first",
    ),
    "03": SimulationContract(
        name="03",
        poisson_exposure=0.125,
        overshoot_rule="admit_joiners_to_capacity",
        fission_rule="binomial",
        daughter_rule="second",
    ),
}


@dataclass(frozen=True)
class CohortConfig:
    matrices: int
    branches_per_state: int
    landmarks: tuple[int, ...] = (20, 35, 50, 65, 80)


@dataclass(frozen=True)
class ExperimentConfig:
    gard: GardConfig = field(default_factory=GardConfig)
    development: CohortConfig = field(
        default_factory=lambda: CohortConfig(matrices=40, branches_per_state=32)
    )
    confirmation: CohortConfig = field(
        default_factory=lambda: CohortConfig(matrices=40, branches_per_state=64)
    )
    horizon: int = 12
    pca_components: int = 12
    logistic_c: float = 0.1
    bootstrap_repetitions: int = 4_096
    permutation_repetitions: int = 512
    regenerate_confirmation: bool = True
    master_seed: str = (
        "7d6e6f7dfe3ff3cd781693f4d95bbdc93f6404c90cbd39076d0c89704b77f21a"
    )

    @classmethod
    def quick(cls) -> "ExperimentConfig":
        return cls(
            development=CohortConfig(
                matrices=6, branches_per_state=8, landmarks=(20, 50, 80)
            ),
            confirmation=CohortConfig(
                matrices=6, branches_per_state=12, landmarks=(20, 50, 80)
            ),
            bootstrap_repetitions=128,
            permutation_repetitions=64,
            regenerate_confirmation=False,
        )

    @classmethod
    def scaled5(cls) -> "ExperimentConfig":
        """Five times the independent matrices and stochastic shooting volume."""

        return cls(
            development=CohortConfig(matrices=200, branches_per_state=32),
            confirmation=CohortConfig(matrices=200, branches_per_state=64),
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["candidates"] = {key: asdict(value) for key, value in CANDIDATES.items()}
        return out
