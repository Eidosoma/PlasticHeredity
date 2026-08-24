"""Core implementation for the frozen E01 S12 strict MRR.

This module consumes only the independently implemented GARD engine, S06 seed
hierarchy, S08/S09 transformations, and S10 strict information-dynamic branch.
It deliberately does not import the failed S11 or S11R fixed-window packages.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import adjusted_rand_score

from e01_compositional_preprocessing import helmert_simplex_basis
from e01_gard_independent import (
    CatalyticMatrixBranch,
    ClockSemantics,
    DaughterSelection,
    FissionSemantics,
    GardSpecification,
    GrowthBoundary,
    InitialStateSemantics,
    LossNonnegativity,
    MaxStepsSemantics,
    PostFissionSemantics,
    ProfileRole,
    PropensityEquationBranch,
    ReservoirSemantics,
    UpdateKernel,
    ZeroPropensitySemantics,
    generate_catalytic_matrix,
    initialize_state,
    simulate_lineage,
)
from e01_gard_independent.records import LineageResult
from e01_gard_reproducibility import (
    CouplingPolicy,
    SeedBundle,
    SeedRequest,
    StreamPurpose,
    derive_seed_bundle,
    isolated_stream_namespace,
)
from e01_information_dynamics.validation import (
    gaussian_mmi_oracle,
    gaussian_mutual_information,
    gaussian_partition_objective,
    spectral_partition,
)

PREREGISTRATION_VERSION = "E01-S12-STRICT-MRR-v1.0.0"
GARD_SPECIFICATION_ID = "E01-S12-GARD-HISTORICAL-SOURCE-TRACEABLE-SCALE-v1.0.0"
ENGINE_ID = "e01_gard_independent@1.0.0"
ROOT_SEED_HEX = "12" * 32
MINIMUM_EFFECTIVE_SAMPLES = 512
MAXIMUM_CONDITION_NUMBER = 1.0e12
NUMERIC_TOLERANCE = 1.0e-10
PREPROCESSING_IDS = (
    "E01-S12-PREPROC-ADD0p5-DROPCLR-D100-C100-v1.0.0",
    "E01-S12-PREPROC-ADD0p5-ILR-HELMERT-D100-v1.0.0",
)
REDUNDANCY_IDS = (
    "E01-S10-REDUNDANCY-MMI-v1.0.0",
    "E01-S10-REDUNDANCY-CCS-v1.0.0",
)


@dataclass(frozen=True, slots=True)
class BaselineTrajectory:
    """One complete baseline simulation plus the materialized observation view."""

    matrix_index: int
    trajectory_id: str
    specification: GardSpecification
    seed_payload: dict[str, Any]
    beta: NDArray[np.float64]
    lineage: LineageResult
    states: NDArray[np.int64]
    observation_kinds: tuple[str, ...]
    generations: NDArray[np.int64]
    growth_generations_one_based: NDArray[np.int64]
    molecular_steps: NDArray[np.int64]
    generation_local_steps: NDArray[np.int64]
    trajectory_sha256: str


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """Lossless, no-drop coordinates for both frozen S12 representations."""

    coordinates: dict[str, NDArray[np.float64]]
    zero_counts: NDArray[np.int64]
    masses: NDArray[np.int64]
    maximum_inverse_errors: dict[str, NDArray[np.float64]]
    maximum_closure_errors: dict[str, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class PartitionLock:
    """The first past-only partition to pass every preregistered lock gate."""

    status: str
    reason: str | None
    preprocessing_id: str
    observation_index: int | None
    generation: int | None
    molecular_step: int | None
    part_a: tuple[int, ...] | None
    part_b: tuple[int, ...] | None
    partition_id: str | None
    objective: float | None
    relative_eigengap: float | None
    minimum_side_fraction: float | None
    replay_maximum_objective_error: float | None
    replay_minimum_ari: float | None
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class StrictEstimate:
    """One status-bearing strict expanding estimate or explicit suppression."""

    status: str
    reason: str | None
    n_eff: int
    value: float | None
    numerical_rank: int | None
    rank_tolerance: float | None
    condition_number: float | None
    minimum_eigenvalue: float | None
    lattice_closure_error: float | None
    paper_equation_closure_error: float | None
    total_mutual_information: float | None


def build_baseline_specification() -> GardSpecification:
    """Instantiate the complete source-traceable historical-behavior branch."""

    return GardSpecification(
        specification_id=GARD_SPECIFICATION_ID,
        profile_role=ProfileRole.HISTORICAL_DISTRIBUTION_COMPARISON,
        n_species=100,
        n_min=40,
        n_max=80,
        n_generations=100,
        max_steps=None,
        beta_a=-4.0,
        beta_sigma=4.0,
        k_f=0.01,
        k_b=0.0001,
        rho=tuple([0.01] * 100),
        propensity_equation_branch=PropensityEquationBranch.HISTORICAL_REFERENCE,
        catalytic_matrix_branch=CatalyticMatrixBranch.HISTORICAL_ORIENTATION_WITH_DIAGONAL,
        reservoir_semantics=ReservoirSemantics.CONSTANT_REQUIRE_SUM_ONE,
        update_kernel=UpdateKernel.CATEGORICAL_SINGLE_EVENT,
        clock_semantics=ClockSemantics.EVENT_INDEX_ONLY,
        poisson_exposure=None,
        loss_nonnegativity=LossNonnegativity.EVENTWISE_ZERO_RATE,
        growth_boundary=GrowthBoundary.EVENTWISE_EXACT_STOP,
        max_steps_semantics=MaxStepsSemantics.UNBOUNDED_HISTORICAL_COMPARISON,
        zero_propensity_semantics=ZeroPropensitySemantics.RAISE,
        fission_semantics=FissionSemantics.FIXED_SIZE_WITHOUT_REPLACEMENT_ODD_DISCARD,
        fission_probability=None,
        daughter_selection=DaughterSelection.FIRST,
        post_fission_semantics=PostFissionSemantics.CONTINUE_EXACT_SELECTED,
        initial_state_semantics=InitialStateSemantics.WITH_REPLACEMENT_COUNTS,
    )


def baseline_seed_bundle(matrix_index: int) -> SeedBundle:
    """Derive the canonical isolated nine-stream bundle for one baseline."""

    if matrix_index not in range(12):
        raise ValueError("S12 baseline matrix_index must be in 0..11.")
    trajectory_id = f"E01-S12-B{matrix_index:02d}"
    namespace = isolated_stream_namespace(
        experiment_id="E01",
        specification_id=GARD_SPECIFICATION_ID,
        trajectory_id=trajectory_id,
        replicate_index=matrix_index,
    )
    request = SeedRequest(
        experiment_id="E01",
        specification_id=GARD_SPECIFICATION_ID,
        trajectory_id=trajectory_id,
        replicate_index=matrix_index,
        engine_id=ENGINE_ID,
        root_seed_hex=ROOT_SEED_HEX,
        coupling_policy=CouplingPolicy.TRAJECTORY_ISOLATED,
        coupling_reason=None,
        stream_namespaces={purpose: namespace for purpose in StreamPurpose},
    )
    return derive_seed_bundle(request)


def build_observations(
    lineage: LineageResult,
) -> tuple[
    NDArray[np.int64],
    tuple[str, ...],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Materialize initial, every post-event, and every post-fission state."""

    states: list[tuple[int, ...]] = [lineage.initial_state]
    kinds = ["initial_selected_state"]
    generations = [0]
    growth_generations = [0]
    molecular_steps = [0]
    local_steps = [0]
    molecular_step = 0
    for generation in lineage.generations:
        generation_one_based = generation.generation_index_one_based
        for event in generation.growth.events:
            molecular_step += 1
            states.append(event.post_state)
            kinds.append("molecular_event")
            generations.append(generation_one_based - 1)
            growth_generations.append(generation_one_based)
            molecular_steps.append(molecular_step)
            local_steps.append(event.step_index_one_based)
        if generation.fission is not None:
            states.append(generation.fission.selected_daughter)
            kinds.append("post_fission")
            generations.append(generation_one_based)
            growth_generations.append(generation_one_based)
            molecular_steps.append(molecular_step)
            local_steps.append(0)
    return (
        np.asarray(states, dtype=np.int64),
        tuple(kinds),
        np.asarray(generations, dtype=np.int64),
        np.asarray(growth_generations, dtype=np.int64),
        np.asarray(molecular_steps, dtype=np.int64),
        np.asarray(local_steps, dtype=np.int64),
    )


def _trajectory_digest(
    beta: NDArray[np.float64],
    states: NDArray[np.int64],
    kinds: tuple[str, ...],
    generations: NDArray[np.int64],
    molecular_steps: NDArray[np.int64],
) -> str:
    digest = hashlib.sha256()
    for array in (beta.astype("<f8", copy=False), states.astype("<i8", copy=False)):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    digest.update(json.dumps(kinds, separators=(",", ":")).encode("utf-8"))
    digest.update(generations.astype("<i8", copy=False).tobytes())
    digest.update(molecular_steps.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def simulate_baseline(matrix_index: int) -> BaselineTrajectory:
    """Generate one of exactly twelve preregistered complete baselines."""

    specification = build_baseline_specification()
    bundle = baseline_seed_bundle(matrix_index)
    generators = bundle.fresh_generators()
    streams = bundle.independent_engine_streams(generators)
    beta = generate_catalytic_matrix(specification, streams.catalytic_matrix)
    initial = initialize_state(specification, streams.initialization)
    lineage = simulate_lineage(
        initial,
        beta=beta,
        specification=specification,
        rng_streams=streams,
    )
    observations = build_observations(lineage)
    digest = _trajectory_digest(
        beta, observations[0], observations[1], observations[2], observations[4]
    )
    return BaselineTrajectory(
        matrix_index=matrix_index,
        trajectory_id=f"E01-S12-B{matrix_index:02d}",
        specification=specification,
        seed_payload=bundle.to_payload(),
        beta=beta,
        lineage=lineage,
        states=observations[0],
        observation_kinds=observations[1],
        generations=observations[2],
        growth_generations_one_based=observations[3],
        molecular_steps=observations[4],
        generation_local_steps=observations[5],
        trajectory_sha256=digest,
    )


def _close_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return values / np.sum(values, axis=1, keepdims=True, dtype=np.float64)


def _inverse_log_rows(log_values: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = log_values - np.max(log_values, axis=1, keepdims=True)
    return _close_rows(np.exp(shifted))


@lru_cache(maxsize=1)
def _helmert_basis_100() -> NDArray[np.float64]:
    """Construct the frozen D=100 basis once and never mutate it."""

    basis = helmert_simplex_basis(100)
    basis.setflags(write=False)
    return basis


def preprocess_states(states: NDArray[np.int64]) -> PreprocessingResult:
    """Apply the two frozen additive-0.5, D=100 no-drop representations."""

    integer = np.asarray(states)
    if integer.ndim != 2 or integer.shape[1] != 100:
        raise ValueError("S12 preprocessing requires observations by 100 species.")
    if not np.issubdtype(integer.dtype, np.integer) or np.any(integer < 0):
        raise ValueError("S12 preprocessing requires nonnegative integer states.")
    values = integer.astype(np.float64)
    masses = np.sum(integer, axis=1, dtype=np.int64)
    zero_counts = np.sum(integer == 0, axis=1, dtype=np.int64)
    treated = _close_rows(values + 0.5)
    logs = np.log(treated)
    full_clr = logs - np.mean(logs, axis=1, keepdims=True)

    dropped = full_clr[:, :99].copy()
    dropped_full = np.column_stack([dropped, -np.sum(dropped, axis=1)])
    dropped_inverse = _inverse_log_rows(dropped_full)

    basis = _helmert_basis_100()
    ilr = logs @ basis
    ilr_inverse = _inverse_log_rows(ilr @ basis.T)

    coordinates = {
        PREPROCESSING_IDS[0]: dropped,
        PREPROCESSING_IDS[1]: ilr,
    }
    inverses = {
        PREPROCESSING_IDS[0]: dropped_inverse,
        PREPROCESSING_IDS[1]: ilr_inverse,
    }
    inverse_errors = {
        key: np.max(np.abs(inverse - treated), axis=1)
        for key, inverse in inverses.items()
    }
    closure_errors = {
        key: np.abs(np.sum(inverse, axis=1) - 1.0) for key, inverse in inverses.items()
    }
    if not all(np.all(np.isfinite(value)) for value in coordinates.values()):
        raise ValueError("Frozen S12 preprocessing produced nonfinite coordinates.")
    if (
        max(float(np.max(value)) for value in inverse_errors.values())
        > NUMERIC_TOLERANCE
    ):
        raise ValueError("Frozen S12 preprocessing failed inverse validation.")
    return PreprocessingResult(
        coordinates=coordinates,
        zero_counts=zero_counts,
        masses=masses,
        maximum_inverse_errors=inverse_errors,
        maximum_closure_errors=closure_errors,
    )


def _canonical_part(part: tuple[int, ...], dimension: int) -> tuple[int, ...]:
    selected = tuple(sorted(set(part)))
    if not selected or len(selected) == dimension:
        raise ValueError("partition must be nontrivial")
    if 0 in selected:
        return selected
    chosen = set(selected)
    return tuple(index for index in range(dimension) if index not in chosen)


def _partition_labels(part: tuple[int, ...], dimension: int) -> NDArray[np.int8]:
    labels = np.ones(dimension, dtype=np.int8)
    labels[list(part)] = 0
    return labels


def _partition_id(
    preprocessing_id: str, observation_index: int, part_a: tuple[int, ...]
) -> str:
    payload = {
        "branch": "E01-S12-PARTITION-PASTONLY-FIRST-PASS-LOCK-v1.0.0",
        "observationIndex": observation_index,
        "partA": list(part_a),
        "preprocessingId": preprocessing_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return f"E01-S12-PART-{digest}-v1.0.0"


def find_past_only_partition_lock(
    coordinates: NDArray[np.float64],
    *,
    preprocessing_id: str,
    observation_kinds: tuple[str, ...],
    generations: NDArray[np.int64],
    molecular_steps: NDArray[np.int64],
    estimator_rng: np.random.Generator,
) -> PartitionLock:
    """Find and freeze the first passing post-fission S10 spectral candidate."""

    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 99 or not np.all(np.isfinite(values)):
        raise ValueError("partition lock requires finite N by 99 coordinates")
    history: list[dict[str, Any]] = []
    dimension = values.shape[1]
    for observation_index, kind in enumerate(observation_kinds):
        if kind != "post_fission":
            continue
        base = {
            "preprocessingId": preprocessing_id,
            "observationIndex": observation_index,
            "nEff": observation_index,
            "generation": int(generations[observation_index]),
            "molecularStep": int(molecular_steps[observation_index]),
        }
        if observation_index < MINIMUM_EFFECTIVE_SAMPLES:
            history.append(
                {
                    **base,
                    "status": "INELIGIBLE",
                    "reason": "INSUFFICIENT_EFFECTIVE_SAMPLES",
                    "partA": None,
                    "partB": None,
                }
            )
            continue
        prefix = values[: observation_index + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spectral = spectral_partition(prefix)
        if spectral["status"] != "ELIGIBLE":
            history.append(
                {
                    **base,
                    "status": "INELIGIBLE",
                    "reason": spectral["reason"],
                    "partA": None,
                    "partB": None,
                    "relativeEigengap": spectral.get("relativeEigengap"),
                }
            )
            continue
        part_a = _canonical_part(tuple(spectral["partA"]), dimension)
        part_b = tuple(index for index in range(dimension) if index not in set(part_a))
        minimum_fraction = min(len(part_a), len(part_b)) / dimension
        if minimum_fraction < 0.1:
            history.append(
                {
                    **base,
                    "status": "INELIGIBLE",
                    "reason": "PARTITION_MINIMUM_SIDE_FRACTION_BELOW_0p1",
                    "partA": list(part_a),
                    "partB": list(part_b),
                    "minimumSideFraction": minimum_fraction,
                    "relativeEigengap": spectral["relativeEigengap"],
                }
            )
            continue
        objective = gaussian_partition_objective(
            prefix,
            part_a,
            mapping="group_mean",
            objective="bidirectional_lagged_mi",
            normalization="none",
        )
        if objective["status"] != "ELIGIBLE":
            history.append(
                {
                    **base,
                    "status": "INELIGIBLE",
                    "reason": f"PARTITION_OBJECTIVE::{objective['reason']}",
                    "partA": list(part_a),
                    "partB": list(part_b),
                    "minimumSideFraction": minimum_fraction,
                    "relativeEigengap": spectral["relativeEigengap"],
                }
            )
            continue

        replay_aris: list[float] = []
        replay_errors: list[float] = []
        replay_reason: str | None = None
        for _ in range(3):
            permutation = estimator_rng.permutation(dimension)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                permuted = spectral_partition(prefix[:, permutation])
            if permuted["status"] != "ELIGIBLE":
                replay_reason = f"FEATURE_RELABEL::{permuted['reason']}"
                break
            mapped = _canonical_part(
                tuple(int(permutation[index]) for index in permuted["partA"]),
                dimension,
            )
            ari = float(
                adjusted_rand_score(
                    _partition_labels(part_a, dimension),
                    _partition_labels(mapped, dimension),
                )
            )
            mapped_objective = gaussian_partition_objective(
                prefix,
                mapped,
                mapping="group_mean",
                objective="bidirectional_lagged_mi",
                normalization="none",
            )
            if mapped_objective["status"] != "ELIGIBLE":
                replay_reason = (
                    f"FEATURE_RELABEL_OBJECTIVE::{mapped_objective['reason']}"
                )
                break
            error = abs(
                float(mapped_objective["normalizedObjective"])
                - float(objective["normalizedObjective"])
            )
            replay_aris.append(ari)
            replay_errors.append(error)
            if ari != 1.0 or error > NUMERIC_TOLERANCE:
                replay_reason = "FEATURE_RELABEL_REPLAY_GATE_FAILED"
                break
        if replay_reason is not None or len(replay_aris) != 3:
            history.append(
                {
                    **base,
                    "status": "INELIGIBLE",
                    "reason": replay_reason,
                    "partA": list(part_a),
                    "partB": list(part_b),
                    "minimumSideFraction": minimum_fraction,
                    "relativeEigengap": spectral["relativeEigengap"],
                    "replayMinimumAri": min(replay_aris) if replay_aris else None,
                    "replayMaximumObjectiveError": max(replay_errors)
                    if replay_errors
                    else None,
                }
            )
            continue
        record = {
            **base,
            "status": "ELIGIBLE_LOCKED",
            "reason": None,
            "partA": list(part_a),
            "partB": list(part_b),
            "minimumSideFraction": minimum_fraction,
            "relativeEigengap": float(spectral["relativeEigengap"]),
            "objective": float(objective["normalizedObjective"]),
            "replayMinimumAri": min(replay_aris),
            "replayMaximumObjectiveError": max(replay_errors),
        }
        history.append(record)
        return PartitionLock(
            status="ELIGIBLE_LOCKED",
            reason=None,
            preprocessing_id=preprocessing_id,
            observation_index=observation_index,
            generation=int(generations[observation_index]),
            molecular_step=int(molecular_steps[observation_index]),
            part_a=part_a,
            part_b=part_b,
            partition_id=_partition_id(preprocessing_id, observation_index, part_a),
            objective=float(objective["normalizedObjective"]),
            relative_eigengap=float(spectral["relativeEigengap"]),
            minimum_side_fraction=minimum_fraction,
            replay_maximum_objective_error=max(replay_errors),
            replay_minimum_ari=min(replay_aris),
            history=tuple(history),
        )
    return PartitionLock(
        status="INELIGIBLE",
        reason="NO_POST_FISSION_PARTITION_PASSED",
        preprocessing_id=preprocessing_id,
        observation_index=None,
        generation=None,
        molecular_step=None,
        part_a=None,
        part_b=None,
        partition_id=None,
        objective=None,
        relative_eigengap=None,
        minimum_side_fraction=None,
        replay_maximum_objective_error=None,
        replay_minimum_ari=None,
        history=tuple(history),
    )


def mapped_part_series(
    coordinates: NDArray[np.float64], part_a: tuple[int, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply the frozen S10 arithmetic group-mean partition mapping."""

    values = np.asarray(coordinates, dtype=np.float64)
    selected = set(part_a)
    part_b = tuple(index for index in range(values.shape[1]) if index not in selected)
    return np.mean(values[:, part_a], axis=1), np.mean(values[:, part_b], axis=1)


class RunningStrictEstimator:
    """Binary64 sufficient-statistic implementation of the strict S10 mean scalar."""

    def __init__(self) -> None:
        self.n = 0
        self.sum = np.zeros(4, dtype=np.float64)
        self.cross = np.zeros((4, 4), dtype=np.float64)

    @classmethod
    def from_series(
        cls,
        source: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> RunningStrictEstimator:
        result = cls()
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.ndim != 1 or target.shape != source.shape:
            raise ValueError("strict estimator requires same-length scalar series")
        if source.size > 1:
            transitions = np.column_stack(
                [source[:-1], target[:-1], source[1:], target[1:]]
            )
            result.n = int(transitions.shape[0])
            result.sum = np.sum(transitions, axis=0, dtype=np.float64)
            result.cross = transitions.T @ transitions
        return result

    def update(
        self,
        prior_source: float,
        prior_target: float,
        source: float,
        target: float,
    ) -> None:
        vector = np.asarray(
            [prior_source, prior_target, source, target], dtype=np.float64
        )
        self.n += 1
        self.sum += vector
        self.cross += np.outer(vector, vector)

    def _estimate_from_stats(
        self, n: int, total: NDArray[np.float64], cross: NDArray[np.float64]
    ) -> StrictEstimate:
        if n < MINIMUM_EFFECTIVE_SAMPLES:
            return StrictEstimate(
                "INELIGIBLE",
                "INSUFFICIENT_EFFECTIVE_SAMPLES",
                n,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        if not np.all(np.isfinite(total)) or not np.all(np.isfinite(cross)):
            return StrictEstimate(
                "INELIGIBLE",
                "NONFINITE_INPUT_NO_ROW_DELETION",
                n,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        covariance = (cross - np.outer(total, total) / n) / (n - 1)
        diagonal = np.diag(covariance)
        if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
            return StrictEstimate(
                "INELIGIBLE",
                "ZERO_OR_NONFINITE_SAMPLE_SD",
                n,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        scale = np.sqrt(diagonal)
        normalized = covariance / np.outer(scale, scale)
        singular = np.linalg.svd(normalized, compute_uv=False)
        largest = float(singular[0])
        tolerance = float(max(n, 4) * np.finfo(np.float64).eps * largest)
        rank = int(np.sum(singular > tolerance))
        condition = float(largest / singular[-1]) if singular[-1] > 0 else math.inf
        eigenvalues = np.linalg.eigvalsh(normalized)
        minimum = float(eigenvalues[0])
        if rank != 4:
            reason = "JOINT_COVARIANCE_RANK_DEFICIENT"
        elif minimum <= 0:
            reason = "JOINT_COVARIANCE_NOT_POSITIVE_DEFINITE"
        elif not np.isfinite(condition) or condition > MAXIMUM_CONDITION_NUMBER:
            reason = "JOINT_COVARIANCE_ILL_CONDITIONED"
        else:
            reason = None
        if reason is not None:
            return StrictEstimate(
                "INELIGIBLE",
                reason,
                n,
                None,
                rank,
                tolerance,
                condition,
                minimum,
                None,
                None,
                None,
            )
        total_mi = gaussian_mutual_information(normalized, (0, 1), (2, 3))
        source_mi = gaussian_mutual_information(normalized, (0,), (2, 3))
        target_mi = gaussian_mutual_information(normalized, (1,), (2, 3))
        direct = float(total_mi - source_mi - target_mi)
        mmi = gaussian_mmi_oracle(normalized)
        atoms = mmi["atomMeans"]
        from_atoms = float(
            sum(atoms[key] for key in ("str", "stx", "sty", "sts"))
            - sum(atoms[key] for key in ("rtr", "rtx", "rty", "rts"))
        )
        lattice_error = float(sum(atoms.values()) - mmi["totalMi"])
        equation_error = float(from_atoms - direct)
        if (
            not np.isfinite(direct)
            or abs(lattice_error) > NUMERIC_TOLERANCE
            or abs(equation_error) > NUMERIC_TOLERANCE
        ):
            return StrictEstimate(
                "INELIGIBLE",
                "LATTICE_OR_EQUATION_CLOSURE_FAILED",
                n,
                None,
                rank,
                tolerance,
                condition,
                minimum,
                lattice_error,
                equation_error,
                total_mi,
            )
        return StrictEstimate(
            "ELIGIBLE_NUMERIC_STRICT_EXPANDING",
            None,
            n,
            direct,
            rank,
            tolerance,
            condition,
            minimum,
            lattice_error,
            equation_error,
            total_mi,
        )

    def estimate(self) -> StrictEstimate:
        return self._estimate_from_stats(self.n, self.sum, self.cross)

    def hypothetical(
        self,
        prior_source: float,
        prior_target: float,
        source: float,
        target: float,
    ) -> StrictEstimate:
        vector = np.asarray(
            [prior_source, prior_target, source, target], dtype=np.float64
        )
        return self._estimate_from_stats(
            self.n + 1,
            self.sum + vector,
            self.cross + np.outer(vector, vector),
        )


def expanding_estimates(
    coordinates: NDArray[np.float64], lock: PartitionLock
) -> list[StrictEstimate]:
    """Emit one strict status for every observation under a past-only lock."""

    count = np.asarray(coordinates).shape[0]
    if lock.part_a is None or lock.observation_index is None:
        return [
            StrictEstimate(
                "INELIGIBLE",
                "PARTITION_NOT_YET_LOCKED",
                index,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            for index in range(count)
        ]
    source, target = mapped_part_series(coordinates, lock.part_a)
    output: list[StrictEstimate] = []
    running: RunningStrictEstimator | None = None
    for index in range(count):
        if index < MINIMUM_EFFECTIVE_SAMPLES:
            output.append(
                StrictEstimate(
                    "INELIGIBLE",
                    "INSUFFICIENT_EFFECTIVE_SAMPLES",
                    index,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue
        if index < lock.observation_index:
            output.append(
                StrictEstimate(
                    "INELIGIBLE",
                    "PARTITION_NOT_YET_LOCKED",
                    index,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue
        if running is None:
            running = RunningStrictEstimator.from_series(
                source[: index + 1], target[: index + 1]
            )
        elif index > lock.observation_index:
            running.update(
                source[index - 1], target[index - 1], source[index], target[index]
            )
        output.append(running.estimate())
    return output


def _candidate_states(state: NDArray[np.int64]) -> list[dict[str, Any]]:
    current = np.asarray(state, dtype=np.int64)
    candidates: list[dict[str, Any]] = [
        {
            "candidateId": "noop",
            "actionClass": "noop",
            "speciesIndexZeroBased": None,
            "massDelta": 0,
            "state": current.copy(),
        }
    ]
    for species in range(current.size):
        addition = current.copy()
        addition[species] += 1
        candidates.append(
            {
                "candidateId": f"add:{species}",
                "actionClass": "addition",
                "speciesIndexZeroBased": species,
                "massDelta": 1,
                "state": addition,
            }
        )
    for species in np.flatnonzero(current > 0):
        deletion = current.copy()
        deletion[int(species)] -= 1
        candidates.append(
            {
                "candidateId": f"delete:{int(species)}",
                "actionClass": "deletion",
                "speciesIndexZeroBased": int(species),
                "massDelta": -1,
                "state": deletion,
            }
        )
    return candidates


def transform_one_state(
    state: NDArray[np.int64], preprocessing_id: str
) -> NDArray[np.float64]:
    result = preprocess_states(np.asarray(state, dtype=np.int64)[None, :])
    return result.coordinates[preprocessing_id][0]


def score_action_candidates(
    state: NDArray[np.int64],
    *,
    preprocessing_coordinates: dict[str, NDArray[np.float64]],
    locks: dict[str, PartitionLock],
) -> list[dict[str, Any]]:
    """Score the complete action set under both frozen preprocessing branches."""

    candidates = _candidate_states(np.asarray(state, dtype=np.int64))
    context: dict[
        str,
        tuple[RunningStrictEstimator, tuple[int, ...], float, float],
    ] = {}
    for preprocessing_id in PREPROCESSING_IDS:
        lock = locks[preprocessing_id]
        if lock.part_a is None:
            raise ValueError("candidate scoring requires both partitions locked")
        source, target = mapped_part_series(
            preprocessing_coordinates[preprocessing_id], lock.part_a
        )
        running = RunningStrictEstimator.from_series(source, target)
        context[preprocessing_id] = (
            running,
            lock.part_a,
            float(source[-1]),
            float(target[-1]),
        )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for preprocessing_id in PREPROCESSING_IDS:
            running, part_a, prior_source, prior_target = context[preprocessing_id]
            coordinates = transform_one_state(candidate["state"], preprocessing_id)
            selected = set(part_a)
            part_b = tuple(
                index for index in range(coordinates.size) if index not in selected
            )
            candidate_source = float(np.mean(coordinates[list(part_a)]))
            candidate_target = float(np.mean(coordinates[list(part_b)]))
            estimate = running.hypothetical(
                prior_source,
                prior_target,
                candidate_source,
                candidate_target,
            )
            rows.append(
                {
                    "candidateId": candidate["candidateId"],
                    "actionClass": candidate["actionClass"],
                    "speciesIndexZeroBased": candidate["speciesIndexZeroBased"],
                    "massDelta": candidate["massDelta"],
                    "preprocessingId": preprocessing_id,
                    "status": estimate.status,
                    "reason": estimate.reason,
                    "score": estimate.value,
                    "nEffCandidate": estimate.n_eff,
                    "conditionNumber": estimate.condition_number,
                    "numericalRank": estimate.numerical_rank,
                    "candidateState": candidate["state"].tolist(),
                }
            )
    return rows


def action_null_envelope(
    score_rows: list[dict[str, Any]],
    *,
    direction: Literal["max", "min"],
    rng: np.random.Generator,
    families: int = 4096,
) -> dict[str, Any]:
    """Build the frozen state-matched, complete-candidate gap envelope."""

    if families != 4096:
        raise ValueError("S12 action null requires exactly 4096 families.")
    if any(row["status"] != "ELIGIBLE_NUMERIC_STRICT_EXPANDING" for row in score_rows):
        return {
            "status": "INELIGIBLE",
            "reason": "ONE_OR_MORE_CANDIDATES_FAILED_STRICT_GATE",
            "threshold": None,
        }
    classes: dict[str, NDArray[np.float64]] = {}
    for action_class in ("addition", "deletion"):
        scores = np.asarray(
            [row["score"] for row in score_rows if row["actionClass"] == action_class],
            dtype=np.float64,
        )
        if scores.size:
            classes[action_class] = scores - np.mean(scores)
    if not classes:
        return {
            "status": "INELIGIBLE",
            "reason": "NO_ADDITION_OR_DELETION_CANDIDATES",
            "threshold": None,
        }
    simulated = [np.zeros((families, 1), dtype=np.float64)]
    for residuals in classes.values():
        indices = rng.integers(0, residuals.size, size=(families, residuals.size))
        simulated.append(residuals[indices])
    family_scores = np.concatenate(simulated, axis=1)
    if family_scores.shape[1] < 2:
        return {
            "status": "INELIGIBLE",
            "reason": "FEWER_THAN_TWO_NULL_CANDIDATES",
            "threshold": None,
        }
    ordered = np.sort(family_scores, axis=1)
    gaps = (
        ordered[:, -1] - ordered[:, -2]
        if direction == "max"
        else ordered[:, 1] - ordered[:, 0]
    )
    threshold = float(np.quantile(gaps, 0.99, method="higher"))
    return {
        "status": "ELIGIBLE",
        "reason": None,
        "threshold": threshold,
        "families": families,
        "minimumGap": float(np.min(gaps)),
        "medianGap": float(np.median(gaps)),
        "maximumGap": float(np.max(gaps)),
    }


def lineage_event_rows(trajectory: BaselineTrajectory) -> list[dict[str, Any]]:
    """Return complete event and fission rows suitable for nested Parquet."""

    rows: list[dict[str, Any]] = []
    global_event = 0
    for generation in trajectory.lineage.generations:
        for event in generation.growth.events:
            global_event += 1
            payload = asdict(event)
            payload.update(
                {
                    "trajectoryId": trajectory.trajectory_id,
                    "matrixIndex": trajectory.matrix_index,
                    "recordType": "molecular_event",
                    "globalEventIndexOneBased": global_event,
                }
            )
            rows.append(payload)
        if generation.fission is not None:
            payload = asdict(generation.fission)
            payload.update(
                {
                    "trajectoryId": trajectory.trajectory_id,
                    "matrixIndex": trajectory.matrix_index,
                    "recordType": "fission",
                    "globalEventIndexOneBased": global_event,
                }
            )
            rows.append(payload)
    return rows
