"""Immutable diagnostic records emitted by the independent S05 engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PropensityArrays:
    boost: tuple[float, ...]
    join: tuple[float, ...]
    leave: tuple[float, ...]
    concatenated: tuple[float, ...]
    probabilities: tuple[float, ...]
    total: float
    equation_branch: str


@dataclass(frozen=True, slots=True)
class EventLog:
    record_schema_version: str
    specification_id: str
    generation_index_one_based: int
    step_index_one_based: int
    update_kernel: str
    propensity_equation_branch: str
    pre_state: tuple[int, ...]
    post_state: tuple[int, ...]
    pre_mass: int
    post_mass: int
    mass_delta: int
    boost: tuple[float, ...]
    join_propensities: tuple[float, ...]
    leave_propensities: tuple[float, ...]
    event_probabilities: tuple[float, ...]
    total_propensity: float
    selected_event_index_zero_based: int | None
    selected_species_index_zero_based: int | None
    event_kind: str
    selection_probability: float | None
    attempted_join_counts: tuple[int, ...]
    attempted_loss_counts: tuple[int, ...]
    applied_join_counts: tuple[int, ...]
    applied_loss_counts: tuple[int, ...]
    boundary_action: str
    clock_semantics: str
    time_increment: float | None
    model_time_before: float | None
    model_time_after: float | None
    event_rng_stream_id: str
    event_rng_state_sha256_before: str
    event_rng_state_sha256_after: str
    waiting_rng_stream_id: str | None
    waiting_rng_state_sha256_before: str | None
    waiting_rng_state_sha256_after: str | None


@dataclass(frozen=True, slots=True)
class GrowthResult:
    specification_id: str
    generation_index_one_based: int
    initial_state: tuple[int, ...]
    final_state: tuple[int, ...]
    terminal_status: str
    events: tuple[EventLog, ...]
    elapsed_model_time: float | None


@dataclass(frozen=True, slots=True)
class FissionLog:
    record_schema_version: str
    specification_id: str
    generation_index_one_based: int
    fission_semantics: str
    fission_probability: float | None
    parent: tuple[int, ...]
    child_first: tuple[int, ...]
    child_second: tuple[int, ...]
    discarded: tuple[int, ...]
    daughter_selection: str
    selected_daughter_label: str
    selected_daughter: tuple[int, ...]
    post_fission_semantics: str
    conservation_holds: bool
    fission_rng_stream_id: str
    fission_rng_state_sha256_before: str
    fission_rng_state_sha256_after: str
    daughter_rng_stream_id: str
    daughter_rng_consumed: bool
    daughter_rng_state_sha256_before: str
    daughter_rng_state_sha256_after: str


@dataclass(frozen=True, slots=True)
class GenerationResult:
    specification_id: str
    generation_index_one_based: int
    growth: GrowthResult
    fission: FissionLog | None
    next_state: tuple[int, ...] | None
    terminal_status: str


@dataclass(frozen=True, slots=True)
class LineageResult:
    specification_id: str
    initial_state: tuple[int, ...]
    generations: tuple[GenerationResult, ...]
    events: tuple[EventLog, ...]
    fissions: tuple[FissionLog, ...]
    final_state: tuple[int, ...]
    requested_generations: int
    completed_fissions: int
    terminal_status: str
