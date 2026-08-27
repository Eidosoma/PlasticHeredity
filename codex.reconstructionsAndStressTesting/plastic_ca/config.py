"""Frozen contracts for the clean-room experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


SEMANTIC_VALUES: dict[str, tuple[str, ...]] = {
    "launch_anchor": ("prepared_seed", "first_completed_generation"),
    "launch_preparation": ("sweeps", "noiseless_generation"),
    "seed_mode": ("expected_half_hash", "exact_half", "density_stratified"),
    "process_noise": ("pre_rule_each_sweep", "post_rule_each_sweep", "terminal_once"),
    "activity_count": ("realized", "deterministic"),
    "monochrome_death": (
        "terminal_only",
        "realized_immediate",
        "realized_after_minimum",
        "deterministic_immediate",
    ),
    "observed_daughter": ("pre_copy_terminal", "post_copy_offspring"),
}


@dataclass(frozen=True)
class ECASemantics:
    """Execution-order choices left unresolved by the retained prose record.

    Defaults exactly preserve the first clean-room reference run.  Sensitivity
    campaigns replace these values explicitly and emit them in every artifact.
    """

    launch_anchor: str = "prepared_seed"
    launch_preparation: str = "sweeps"
    seed_mode: str = "expected_half_hash"
    process_noise: str = "post_rule_each_sweep"
    activity_count: str = "realized"
    monochrome_death: str = "terminal_only"
    observed_daughter: str = "pre_copy_terminal"

    def __post_init__(self) -> None:
        for field_name, allowed in SEMANTIC_VALUES.items():
            value = getattr(self, field_name)
            if value not in allowed:
                raise ValueError(f"invalid {field_name}={value!r}; choose from {allowed}")


@dataclass(frozen=True)
class ObserverThresholds:
    inherit: float = 0.9
    coherence: float = 0.9
    distinct: float = 0.85
    strict_run: int = 8
    horizon: int = 32
    break_horizon: int = 8


@dataclass(frozen=True)
class ECAConfig:
    width: int = 64
    activity_budget: int | None = None
    min_sweeps: int = 4
    max_sweeps: int = 128
    flip_noise: float = 0.01
    copy_error: float = 0.015
    n_seeds: int = 16
    futures_per_seed: int = 32
    thresholds: ObserverThresholds = field(default_factory=ObserverThresholds)
    observer: str = "raw4"
    seed_namespace: str = "plastic-ca-cleanroom-v1"
    launch_burnin_sweeps: int = 1
    form_mass_quantile: float = 0.5
    semantics: ECASemantics = field(default_factory=ECASemantics)

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.launch_burnin_sweeps < 0:
            raise ValueError("launch_burnin_sweeps must be nonnegative")

    @property
    def resolved_activity_budget(self) -> int:
        return self.activity_budget if self.activity_budget is not None else 4 * self.width

    def to_dict(self) -> dict:
        value = asdict(self)
        value["activity_budget"] = self.resolved_activity_budget
        return value


PROFILES: dict[str, dict[str, int]] = {
    "smoke": {"n_seeds": 4, "futures_per_seed": 4},
    "standard": {"n_seeds": 16, "futures_per_seed": 32},
    "reference": {"n_seeds": 16, "futures_per_seed": 128},
}


def config_for_profile(profile: str, **overrides) -> ECAConfig:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    values = dict(PROFILES[profile])
    values.update({key: value for key, value in overrides.items() if value is not None})
    return ECAConfig(**values)
