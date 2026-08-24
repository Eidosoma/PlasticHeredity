"""Read-only clock audit and preregistered ABC-SMC utilities for S12F."""

from __future__ import annotations

import itertools
import math
import pickle
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .core import (
    DAUGHTER_RULES,
    OVERSHOOT_RULES,
    ExposureDefinition,
    SimulationDefinition,
    derive_seed,
    generator,
)

S12E_CACHE = Path("/cache/e01_s12e/trajectories/development")
S12E_RESULT = Path("/artifacts/research_steps/S12E/engine_development_results.parquet")
S12E_ENGINE_IDS = (
    "K1_PAPER_POISSON_RANDOM_NONEMPTY",
    "K2_PAPER_POISSON_FIRST_DAUGHTER",
    "K3_PAPER_POISSON_RANDOM_LITERAL",
)
ABC_CLOCKS = (
    "C0_BATCH_UPDATES_ONLY",
    "C1_SELECTED_DAUGHTER_RETAINED",
    "C2_EXPLICIT_PRE_AND_POST_FISSION",
)


@dataclass(frozen=True, slots=True)
class Particle:
    particle_id: str
    family: str
    round_index: int
    daughter_rule: str
    overshoot_rule: str
    clock_id: str
    h: float | None
    c: float | None
    h_max: float | None
    parent_particle_id: str | None
    proposal_weight: float

    @property
    def simulation_definition(self) -> SimulationDefinition:
        exposure = ExposureDefinition(
            family=self.family, h=self.h, c=self.c, h_max=self.h_max
        )
        return SimulationDefinition(
            daughter_rule=self.daughter_rule,
            overshoot_rule=self.overshoot_rule,
            exposure=exposure,
        )

    @property
    def discrete_group(self) -> str:
        return f"{self.family}__{self.daughter_rule}__{self.overshoot_rule}__{self.clock_id}"

    @property
    def stream_identity(self) -> str:
        return self.particle_id


def particle_row(particle: Particle) -> dict[str, Any]:
    value = asdict(particle)
    value["discreteGroup"] = particle.discrete_group
    value["simulationIdentity"] = particle.simulation_definition.identity
    return value


def _trusted_load_s12e(path: Path) -> Any:
    # These are locally generated, hash-baselined S12E caches, not external input.
    with path.open("rb") as handle:
        return pickle.load(handle)


def _interval_gap(left: tuple[float, float], right: tuple[float, float]) -> float:
    if left[1] < right[0]:
        return float(right[0] - left[1])
    if right[1] < left[0]:
        return float(left[0] - right[1])
    return 0.0


def phase1_clock_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = pd.read_parquet(S12E_RESULT)
    prior_lookup = {
        (str(row.engineId), int(row.matrixIndex)): row
        for row in prior.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for engine_id in S12E_ENGINE_IDS:
        for matrix_index in range(24):
            path = S12E_CACHE / engine_id / f"M{matrix_index:02d}.pickle"
            trajectory = _trusted_load_s12e(path)
            evidence = prior_lookup[(engine_id, matrix_index)]
            counts = {kind: 0 for kind in ("initial_selected_state", "molecular_update", "post_fission")}
            for observation in trajectory.observations:
                counts[observation.observation_kind] += 1
            cardinality_passed = bool(
                counts["initial_selected_state"] == 1
                and counts["molecular_update"] == trajectory.total_batch_steps
                and counts["post_fission"] == trajectory.completed_fissions
                and len(trajectory.observations)
                == 1 + trajectory.total_batch_steps + trajectory.completed_fissions
            )
            clock_values: dict[str, float] = {
                "C0_BATCH_UPDATES_ONLY": float(trajectory.total_batch_steps),
                "C1_SELECTED_DAUGHTER_RETAINED": float(
                    trajectory.total_batch_steps + trajectory.completed_fissions
                ),
                "C2_EXPLICIT_PRE_AND_POST_FISSION": float(
                    trajectory.total_batch_steps + 2 * trajectory.completed_fissions
                ),
                "C3_NONZERO_REACTION_CHANNEL": float("nan"),
                "C4_GROSS_MOLECULAR_EVENT": float("nan"),
            }
            for clock_id, length in clock_values.items():
                recoverable = clock_id in ABC_CLOCKS
                materialized = clock_id in {
                    "C0_BATCH_UPDATES_ONLY",
                    "C1_SELECTED_DAUGHTER_RETAINED",
                }
                rows.append(
                    {
                        "engineId": engine_id,
                        "matrixIndex": matrix_index,
                        "clockId": clock_id,
                        "tPhi": length,
                        "status": "ELIGIBLE_READ_ONLY" if recoverable else "NOT_RECOVERABLE_FROM_S12E_CACHE",
                        "reason": "exact_from_materialized_boundaries" if recoverable else "S12E_cache_did_not_retain_per_batch_Poisson_draw_vectors",
                        "completedFissions": int(trajectory.completed_fissions),
                        "terminalStatus": str(trajectory.terminal_status),
                        "medianPostFissionMass": float(evidence.medianPostFissionMass),
                        "meanOvershoot": float(evidence.meanOvershoot),
                        "maximumOvershoot": float(evidence.maxOvershoot),
                        "stateCardinalityPassed": cardinality_passed,
                        "priorExactReplayPassed": bool(evidence.exactReplayPassed),
                        "materializedNaturally": materialized,
                        "requiresSyntheticDuplicate": clock_id == "C2_EXPLICIT_PRE_AND_POST_FISSION",
                        "primaryGateAdmissible": clock_id in {
                            "C0_BATCH_UPDATES_ONLY",
                            "C1_SELECTED_DAUGHTER_RETAINED",
                        },
                        "trajectorySha256": trajectory.trajectory_sha256,
                        "cachePath": str(path),
                    }
                )
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    endpoints = (800.0, 800.0, 1000.0)
    for (engine_id, clock_id), group in frame.groupby(["engineId", "clockId"], sort=True):
        finite = group[np.isfinite(group["tPhi"])]
        if finite.empty:
            summaries.append(
                {
                    "engineId": engine_id,
                    "clockId": clock_id,
                    "status": "NOT_RECOVERABLE_FROM_S12E_CACHE",
                    "trajectoryCount": int(group.shape[0]),
                    "clockOnlyGatePassed": False,
                    "gateReason": "per_batch_Poisson_draw_vectors_absent",
                }
            )
            continue
        values = finite["tPhi"].to_numpy(float)
        q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
        endpoints_inside = sum(q05 <= endpoint <= q95 for endpoint in endpoints)
        aggregate_compatible = bool(values.max() >= 1090.0 and q95 <= 1314.0)
        median_post = float(finite["medianPostFissionMass"].median())
        natural = bool(finite["materializedNaturally"].all())
        cardinality = bool(finite["stateCardinalityPassed"].all())
        replay = bool(finite["priorExactReplayPassed"].all())
        gate = bool(
            np.count_nonzero(finite["completedFissions"].to_numpy() == 100) == 24
            and endpoints_inside >= 2
            and aggregate_compatible
            and 35.0 <= median_post <= 45.0
            and natural
            and cardinality
            and replay
        )
        reasons = []
        if endpoints_inside < 2:
            reasons.append("fewer_than_two_sample_endpoints_in_q05_q95")
        if not aggregate_compatible:
            reasons.append("aggregate_support_incompatible")
        if not 35.0 <= median_post <= 45.0:
            reasons.append("post_fission_mass_outside_interval")
        if not natural:
            reasons.append("synthetic_duplicate_required")
        if not cardinality:
            reasons.append("state_cardinality_failed")
        if not replay:
            reasons.append("prior_replay_evidence_failed")
        summaries.append(
            {
                "engineId": engine_id,
                "clockId": clock_id,
                "status": "ELIGIBLE_READ_ONLY",
                "trajectoryCount": int(finite.shape[0]),
                "meanTPhi": float(values.mean()),
                "sdTPhi": float(values.std(ddof=1)),
                "q05TPhi": float(q05),
                "medianTPhi": float(q50),
                "q95TPhi": float(q95),
                "minimumTPhi": float(values.min()),
                "maximumTPhi": float(values.max()),
                "sampleEndpointsInsideQ05Q95": int(endpoints_inside),
                "aggregateCompatible": aggregate_compatible,
                "medianPostFissionMass": median_post,
                "medianMeanOvershoot": float(finite["meanOvershoot"].median()),
                "medianDistanceFromTableRatio": abs(q50 - 813.6363636363636),
                "materializedNaturally": natural,
                "requiresSyntheticDuplicate": bool(finite["requiresSyntheticDuplicate"].any()),
                "stateCardinalityPassed": cardinality,
                "priorExactReplayPassed": replay,
                "clockOnlyGatePassed": gate,
                "gateReason": "PASS" if gate else ";".join(reasons),
            }
        )
    return frame, pd.DataFrame(summaries)


def _rng_from_root(root: str, purpose: str, round_index: int) -> np.random.Generator:
    identity = derive_seed(root, "abc_inference", purpose, round_index)
    return generator(identity)


def _reflect(value: float, lower: float, upper: float) -> float:
    while value < lower or value > upper:
        if value < lower:
            value = 2.0 * lower - value
        if value > upper:
            value = 2.0 * upper - value
    return value


def initial_particles(family: str, root: str, count: int = 256) -> list[Particle]:
    rng = _rng_from_root(root, f"{family}_prior", 1)
    categories = list(itertools.product(DAUGHTER_RULES, OVERSHOOT_RULES, ABC_CLOCKS))
    particles: list[Particle] = []
    for index in range(count):
        daughter, overshoot, clock = categories[index % len(categories)]
        if family == "FIXED_COMMON_EXPOSURE":
            h = float(np.exp(rng.uniform(np.log(0.10), np.log(1.25))))
            c = h_max = None
        else:
            h = None
            c = float(np.exp(rng.uniform(np.log(0.5), np.log(16.0))))
            h_max = float(np.exp(rng.uniform(np.log(0.1), np.log(2.0))))
        particles.append(
            Particle(
                particle_id=f"{family}-R1-P{index:03d}",
                family=family,
                round_index=1,
                daughter_rule=daughter,
                overshoot_rule=overshoot,
                clock_id=clock,
                h=h,
                c=c,
                h_max=h_max,
                parent_particle_id=None,
                proposal_weight=1.0 / count,
            )
        )
    return particles


def _systematic_indices(weights: NDArray[np.float64], count: int, rng: np.random.Generator) -> NDArray[np.int64]:
    normalized = weights / weights.sum()
    positions = (float(rng.random()) + np.arange(count)) / count
    cumulative = np.cumsum(normalized)
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def _normal_pdf(value: float, center: float, sd: float) -> float:
    z = (value - center) / sd
    return math.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def propose_particles(
    family: str,
    root: str,
    round_index: int,
    count: int,
    parents: list[Particle],
    parent_weights: NDArray[np.float64],
    log_kernel_sd: float,
) -> list[Particle]:
    rng = _rng_from_root(root, f"{family}_proposal", round_index)
    selected = _systematic_indices(parent_weights, count, rng)
    proposed: list[Particle] = []
    for index, parent_index in enumerate(selected):
        parent = parents[int(parent_index)]
        if family == "FIXED_COMMON_EXPOSURE":
            logh = _reflect(
                math.log(float(parent.h)) + float(rng.normal(0.0, log_kernel_sd)),
                math.log(0.10),
                math.log(1.25),
            )
            h = math.exp(logh)
            c = h_max = None
        else:
            logc = _reflect(
                math.log(float(parent.c)) + float(rng.normal(0.0, log_kernel_sd)),
                math.log(0.5),
                math.log(16.0),
            )
            loghmax = _reflect(
                math.log(float(parent.h_max)) + float(rng.normal(0.0, log_kernel_sd)),
                math.log(0.1),
                math.log(2.0),
            )
            h = None
            c = math.exp(logc)
            h_max = math.exp(loghmax)
        proposed.append(
            Particle(
                particle_id=f"{family}-R{round_index}-P{index:03d}",
                family=family,
                round_index=round_index,
                daughter_rule=parent.daughter_rule,
                overshoot_rule=parent.overshoot_rule,
                clock_id=parent.clock_id,
                h=h,
                c=c,
                h_max=h_max,
                parent_particle_id=parent.particle_id,
                proposal_weight=float(parent_weights[int(parent_index)]),
            )
        )
    return proposed


def importance_weights(
    particles: list[Particle],
    parents: list[Particle] | None,
    parent_weights: NDArray[np.float64] | None,
    log_kernel_sd: float | None,
) -> NDArray[np.float64]:
    if parents is None or parent_weights is None or log_kernel_sd is None:
        return np.full(len(particles), 1.0 / len(particles), dtype=np.float64)
    values: list[float] = []
    for particle in particles:
        denominator = 0.0
        for parent, weight in zip(parents, parent_weights, strict=True):
            if parent.discrete_group != particle.discrete_group:
                continue
            if particle.family == "FIXED_COMMON_EXPOSURE":
                kernel = _normal_pdf(
                    math.log(float(particle.h)), math.log(float(parent.h)), log_kernel_sd
                )
            else:
                kernel = _normal_pdf(
                    math.log(float(particle.c)), math.log(float(parent.c)), log_kernel_sd
                ) * _normal_pdf(
                    math.log(float(particle.h_max)),
                    math.log(float(parent.h_max)),
                    log_kernel_sd,
                )
            denominator += float(weight) * kernel
        values.append(0.0 if denominator <= 0.0 else 1.0 / denominator)
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all() or result.sum() <= 0.0:
        raise ValueError("ABC importance weights are invalid")
    return result / result.sum()


def _clock_column(clock_id: str) -> str:
    return {
        "C0_BATCH_UPDATES_ONLY": "clockC0",
        "C1_SELECTED_DAUGHTER_RETAINED": "clockC1",
        "C2_EXPLICIT_PRE_AND_POST_FISSION": "clockC2",
    }[clock_id]


def particle_summary_and_distance(
    particle: Particle,
    trajectory_rows: pd.DataFrame,
) -> dict[str, Any]:
    values = trajectory_rows[_clock_column(particle.clock_id)].to_numpy(float)
    q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
    q50_pre = float(trajectory_rows["medianPreFissionMass"].median())
    q50_post = float(trajectory_rows["medianPostFissionMass"].median())
    q95_over = float(trajectory_rows["q95Overshoot"].quantile(0.95))
    f100 = float(np.mean(trajectory_rows["completedFissions"].to_numpy() == 100))
    generation_denominator = max(1, int(trajectory_rows["completedFissions"].sum()))
    fmax = float(trajectory_rows["maxstepsTerminations"].sum() / generation_denominator)
    target_intervals = ((792.0, 808.0), (792.0, 808.0), (990.0, 1010.0))
    sample_gap = float(
        np.mean([_interval_gap(interval, (q05, q95)) / 50.0 for interval in target_intervals])
    )
    aggregate_gap = _interval_gap((q95, q95), (1090.0, 1120.0)) / 200.0
    table_gap = _interval_gap(
        (q50, q50), (569.2307692307692, 1075.2941176470588)
    ) / 198.0
    d_time = 0.60 * sample_gap + 0.25 * aggregate_gap + 0.15 * table_gap
    d_mass = _interval_gap((q50_post, q50_post), (35.0, 45.0)) / 5.0
    length_incompatibility = _interval_gap((q95, q95), (0.0, 1314.0)) / 200.0
    overshoot_factor = min(1.0, max(0.0, q95_over) / 80.0)
    d_overshoot = overshoot_factor * (d_mass + length_incompatibility)
    complexity = 1.0
    if particle.overshoot_rule == "TRIM_NEW_ENTRANTS_TO_NMAX":
        complexity += 1.0
    if particle.family == "ADAPTIVE_GROSS_EVENT_EXPOSURE":
        complexity += 2.0
    if particle.clock_id == "C2_EXPLICIT_PRE_AND_POST_FISSION":
        complexity += 4.0
    distance = (
        4.0 * d_time
        + 2.0 * d_mass
        + 1.0 * d_overshoot
        + 20.0 * (1.0 - f100)
        + 10.0 * fmax
        + 0.05 * complexity
    )
    centers = (800.0, 800.0, 1000.0)
    endpoints_inside = int(sum(q05 <= value <= q95 for value in centers))
    aggregate_compatible = bool(
        float(values.max()) >= 1090.0
        and q95 <= 1314.0
        and float(np.mean(values > 1314.0)) <= 0.05
    )
    accepted = bool(
        distance <= 1.0
        and f100 == 1.0
        and fmax <= 0.05
        and endpoints_inside >= 2
        and 35.0 <= q50_post <= 45.0
        and aggregate_compatible
        and particle.clock_id != "C2_EXPLICIT_PRE_AND_POST_FISSION"
    )
    return {
        **particle_row(particle),
        "q05TPhi": float(q05),
        "q50TPhi": float(q50),
        "q95TPhi": float(q95),
        "minimumTPhi": float(values.min()),
        "maximumTPhi": float(values.max()),
        "q50PreFissionMass": q50_pre,
        "q50PostFissionMass": q50_post,
        "q95Overshoot": q95_over,
        "fractionComplete100": f100,
        "fractionMaxsteps": fmax,
        "sampleEndpointsInsideQ05Q95": endpoints_inside,
        "aggregateCompatible": aggregate_compatible,
        "D_T": d_time,
        "D_M": d_mass,
        "D_O": d_overshoot,
        "complexity": complexity,
        "distance": float(distance),
        "developmentAcceptanceEnvelopePassed": accepted,
    }


def retained_particles(
    particles: list[Particle], result_frame: pd.DataFrame, retain_count: int
) -> tuple[list[Particle], pd.DataFrame]:
    ordered = result_frame.sort_values(["distance", "complexity", "particle_id"]).head(retain_count)
    lookup = {particle.particle_id: particle for particle in particles}
    selected = [lookup[identifier] for identifier in ordered["particle_id"]]
    return selected, ordered.reset_index(drop=True)


def candidate_groups(
    final_particles: list[Particle],
    final_results: pd.DataFrame,
    final_weights: NDArray[np.float64],
    maximum: int = 3,
) -> tuple[pd.DataFrame, list[Particle]]:
    weight_lookup = {
        particle.particle_id: float(weight)
        for particle, weight in zip(final_particles, final_weights, strict=True)
    }
    eligible = final_results[final_results["developmentAcceptanceEnvelopePassed"]].copy()
    eligible["posteriorWeight"] = eligible["particle_id"].map(weight_lookup)
    rows: list[dict[str, Any]] = []
    representatives: list[Particle] = []
    lookup = {particle.particle_id: particle for particle in final_particles}
    for group_id, group in eligible.groupby("discreteGroup", sort=True):
        representative_row = group.sort_values(
            ["distance", "complexity", "particle_id"]
        ).iloc[0]
        representative = lookup[str(representative_row["particle_id"])]
        rows.append(
            {
                "candidateGroup": group_id,
                "posteriorMass": float(group["posteriorWeight"].sum()),
                "particleCount": int(group.shape[0]),
                "medianDistance": float(group["distance"].median()),
                "minimumDistance": float(group["distance"].min()),
                "complexity": float(representative_row["complexity"]),
                "representativeParticleId": representative.particle_id,
                "family": representative.family,
                "daughterRule": representative.daughter_rule,
                "overshootRule": representative.overshoot_rule,
                "clockId": representative.clock_id,
                "h": representative.h,
                "c": representative.c,
                "hMax": representative.h_max,
                "developmentAccepted": True,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "candidateGroup", "posteriorMass", "particleCount", "medianDistance",
                "minimumDistance", "complexity", "representativeParticleId", "family",
                "daughterRule", "overshootRule", "clockId", "h", "c", "hMax",
                "developmentAccepted", "confirmationRank",
            ]
        ), []
    frame = pd.DataFrame(rows).sort_values(
        ["posteriorMass", "medianDistance", "complexity", "candidateGroup"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    frame["confirmationRank"] = np.arange(1, len(frame) + 1)
    frame["selectedForConfirmation"] = frame["confirmationRank"] <= maximum
    selected_ids = set(
        frame.loc[frame["selectedForConfirmation"], "representativeParticleId"].astype(str)
    )
    for row in frame.itertuples(index=False):
        if row.representativeParticleId in selected_ids:
            representatives.append(lookup[row.representativeParticleId])
    representatives.sort(
        key=lambda particle: int(
            frame.loc[
                frame["representativeParticleId"] == particle.particle_id,
                "confirmationRank",
            ].iloc[0]
        )
    )
    return frame, representatives


def posterior_weighted_quantile(
    values: Iterable[float], weights: Iterable[float], probability: float
) -> float:
    value_array = np.asarray(list(values), dtype=float)
    weight_array = np.asarray(list(weights), dtype=float)
    order = np.argsort(value_array)
    value_array = value_array[order]
    weight_array = weight_array[order]
    cumulative = np.cumsum(weight_array / weight_array.sum())
    return float(value_array[np.searchsorted(cumulative, probability, side="left")])
