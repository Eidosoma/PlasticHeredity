"""Frozen core for E01 S12D.

This module exposes the three exact source-defined atom arrays needed by S12D
without changing the already confirmed S12C wrapper.  Scientific execution
uses only the audited safe JSON lattice.  The pinned repositories are invoked
only by the isolated source-equivalence adapter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from e01_gard_independent import (
    generate_catalytic_matrix,
    initialize_state,
    simulate_lineage,
)
from e01_gard_reproducibility import (
    CouplingPolicy,
    SeedBundle,
    SeedRequest,
    StreamPurpose,
    derive_seed_bundle,
    isolated_stream_namespace,
)
from e01_pigozzi_source_audit.core import (
    BOTTOM_ATOM,
    CAUSATION_ATOMS,
    INITIAL_PHIR_ATOM,
    PHIR_ATOMS,
    SYNERGY_ATOM,
    SourceImplementation,
    _local_phi_min,
    load_safe_lattice,
)
from e01_pigozzi_source_equivalence_confirmation.core import (
    ConfirmedAuditResult,
    run_source_pipeline,
)
from e01_pigozzi_source_equivalence_confirmation.core import (
    fixture_array as s12c_fixture_array,
)
from e01_strict_mrr.core import (
    ENGINE_ID,
    BaselineTrajectory,
    _trajectory_digest,
    build_baseline_specification,
    build_observations,
)

VERSION = "E01-S12D-SOURCE-EMERGENCE-METRIC-IDENTITY-CONFIRMATION-v1.0.0"
RESEARCH_STEP_ID = "S12D"
SOURCE_RELATIONSHIP = "SOURCE_INFORMED_METRIC_IDENTITY"
EVIDENCE_CLASS = "SOURCE_INFORMED_METRIC_IDENTITY_CONFIRMATION"
ROOT_SEED_HEX = "14e4e325819ebcda15c9bba605859da22a19a88d283d8c76cc7b859270c8c36f"
GARD_SPECIFICATION_ID = "E01-S12D-GARD-HISTORICAL-SOURCE-TRACEABLE-v1.0.0"
ANALYSIS_SPECIFICATION_ID = "E01-S12D-SOURCE-METRIC-ANALYSIS-v1.0.0"
EXPLORATORY_DATASET_ROLE = "EXPLORATORY_EXISTING_TRAJECTORIES"
CONFIRMATION_DATASET_ROLE = "UNTOUCHED_CONFIRMATION_TRAJECTORIES"
HISTORICAL_LABEL_ID = "HISTORICAL_H090_REPLICATOR"
PAST_ONLY_LABEL_ID = "PAST_ONLY_COSINE_REPLICATOR"

ATOM_KEY_STRINGS = (
    "[[[0,1]],[[0,1]]]",
    "[[[0,1]],[[0]]]",
    "[[[0,1]],[[1]]]",
)


@dataclass(frozen=True)
class EmergenceAuditResult:
    """Status-bearing source result with explicit metric components."""

    implementation: str
    status: str
    reason: str | None
    retained_available: bool
    retained_variables: tuple[int, ...]
    processed: NDArray[np.float64] | None
    mi_matrix: NDArray[np.float64] | None
    fiedler_vector: NDArray[np.float64] | None
    partition_1: tuple[int, ...]
    partition_2: tuple[int, ...]
    partition_average: NDArray[np.float64] | None
    synergy: NDArray[np.float64] | None
    downward_causation: NDArray[np.float64] | None
    emergence: NDArray[np.float64] | None
    local_phi_r: NDArray[np.float64] | None
    local_offset: int
    component_identity_max_abs_error: float | None


def derive_legacy_seed(*identity: object) -> int:
    """Derive a replayable 32-bit legacy RandomState seed in the S12D domain."""

    material = "\x1f".join(
        ["E01-S12D-LEGACY-RANDOMSTATE-v1", ROOT_SEED_HEX, *map(str, identity)]
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def canonical_atom_key(atom: Any) -> str:
    """Serialize a two-antichain atom without tuple-order ambiguity."""

    return json.dumps(atom, separators=(",", ":"))


def _partials(
    reduced: NDArray[np.float64],
    implementation: SourceImplementation,
    safe_lattice_path: str | Path,
) -> dict[Any, NDArray[np.float64]]:
    order, descendants = load_safe_lattice(safe_lattice_path)
    output: dict[Any, NDArray[np.float64]] = {}
    for atom in order:
        redundancy = _local_phi_min(atom, reduced, implementation)
        if atom == BOTTOM_ATOM:
            output[atom] = redundancy
        else:
            output[atom] = redundancy - np.vstack(
                [output[item] for item in descendants[atom]]
            ).sum(axis=0)
    return output


def component_arrays(
    reduced: NDArray[np.float64],
    implementation: SourceImplementation,
    safe_lattice_path: str | Path,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Return synergy, downward causation, emergence, and corrected local Phi-r."""

    partials = _partials(reduced, implementation, safe_lattice_path)
    synergy = partials[SYNERGY_ATOM].copy()
    downward = partials[CAUSATION_ATOMS[0]] + partials[CAUSATION_ATOMS[1]]
    emergence = synergy + downward
    phi_r = partials[INITIAL_PHIR_ATOM].copy()
    for atom in PHIR_ATOMS:
        phi_r += partials[atom]
    return synergy, downward, emergence, phi_r


def _from_base(
    base: ConfirmedAuditResult,
    *,
    synergy: NDArray[np.float64] | None = None,
    downward: NDArray[np.float64] | None = None,
    emergence: NDArray[np.float64] | None = None,
    phi_r: NDArray[np.float64] | None = None,
    status: str | None = None,
    reason: str | None = None,
    identity_error: float | None = None,
) -> EmergenceAuditResult:
    return EmergenceAuditResult(
        implementation=base.implementation,
        status=base.status if status is None else status,
        reason=base.reason if reason is None else reason,
        retained_available=base.retained_available,
        retained_variables=base.retained_variables,
        processed=base.processed,
        mi_matrix=base.mi_matrix,
        fiedler_vector=base.fiedler_vector,
        partition_1=base.partition_1,
        partition_2=base.partition_2,
        partition_average=base.partition_average,
        synergy=synergy,
        downward_causation=downward,
        emergence=emergence,
        local_phi_r=phi_r,
        local_offset=base.local_offset,
        component_identity_max_abs_error=identity_error,
    )


def run_emergence_pipeline(
    observations: NDArray[np.float64],
    implementation: SourceImplementation | str,
    safe_lattice_path: str | Path,
    *,
    preprocessing_seed: int,
    partition_seed: int,
) -> EmergenceAuditResult:
    """Run the immutable S12C pipeline and expose exact source metric components."""

    branch = SourceImplementation(implementation)
    base = run_source_pipeline(
        observations,
        branch,
        safe_lattice_path,
        preprocessing_seed=preprocessing_seed,
        partition_seed=partition_seed,
    )
    if base.partition_average is None:
        return _from_base(base)
    try:
        synergy, downward, emergence, phi_r = component_arrays(
            base.partition_average, branch, safe_lattice_path
        )
        differences: list[float] = []
        if base.emergence is not None and base.emergence.shape == emergence.shape:
            with np.errstate(invalid="ignore"):
                finite = np.isfinite(base.emergence) & np.isfinite(emergence)
                if np.any(finite):
                    differences.append(
                        float(
                            np.max(np.abs(base.emergence[finite] - emergence[finite]))
                        )
                    )
        if base.local_phi_r is not None and base.local_phi_r.shape == phi_r.shape:
            with np.errstate(invalid="ignore"):
                finite = np.isfinite(base.local_phi_r) & np.isfinite(phi_r)
                if np.any(finite):
                    differences.append(
                        float(np.max(np.abs(base.local_phi_r[finite] - phi_r[finite])))
                    )
        identity_error = max(differences, default=0.0)
        status = base.status
        reason = base.reason
        if not all(
            np.array_equal(left, right, equal_nan=True)
            for left, right in (
                (emergence, synergy + downward),
                (emergence, base.emergence),
                (phi_r, base.local_phi_r),
            )
            if right is not None
        ):
            status = "INELIGIBLE_INTERNAL_COMPONENT_IDENTITY_MISMATCH"
            reason = "component_arrays_do_not_exactly_replay_confirmed_s12c_outputs"
        return _from_base(
            base,
            synergy=synergy,
            downward=downward,
            emergence=emergence,
            phi_r=phi_r,
            status=status,
            reason=reason,
            identity_error=identity_error,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed with exact exception.
        return _from_base(
            base,
            status="INELIGIBLE_SOURCE_METRIC_COMPONENT_EXCEPTION",
            reason=f"{type(exc).__name__}:{exc}",
        )


def result_replay_equal(
    left: EmergenceAuditResult, right: EmergenceAuditResult
) -> bool:
    """Exact same-seed replay, including every nonfinite mask."""

    scalar = (
        left.implementation == right.implementation
        and left.status == right.status
        and left.reason == right.reason
        and left.retained_available == right.retained_available
        and left.retained_variables == right.retained_variables
        and left.partition_1 == right.partition_1
        and left.partition_2 == right.partition_2
        and left.local_offset == right.local_offset
        and left.component_identity_max_abs_error
        == right.component_identity_max_abs_error
    )
    if not scalar:
        return False
    for name in (
        "processed",
        "mi_matrix",
        "fiedler_vector",
        "partition_average",
        "synergy",
        "downward_causation",
        "emergence",
        "local_phi_r",
    ):
        a, b = getattr(left, name), getattr(right, name)
        if (a is None) != (b is None):
            return False
        if a is not None and not np.array_equal(a, b, equal_nan=True):
            return False
    return True


NEW_FIXTURE_IDS = (
    "S12D_ORDINARY_BLOCK_GAUSSIAN_A",
    "S12D_ORDINARY_COUPLED_VAR_B",
    "S12D_SINGULAR_DUPLICATE_A",
    "S12D_SINGULAR_LOWRANK_B",
    "S12D_NEAR_SINGULAR_A",
    "S12D_NEAR_SINGULAR_B",
)


def confirmation_fixture_array(fixture_id: str) -> NDArray[np.float64]:
    """Generate one of six new, untouched S12D identity fixtures."""

    rng = np.random.RandomState(derive_legacy_seed("metric_identity", fixture_id))
    n, d = 512, 10
    if fixture_id == "S12D_ORDINARY_BLOCK_GAUSSIAN_A":
        data = rng.normal(size=(n, d))
        data[:, 5:] += 0.45 * data[:, :5]
        data[1:] += 0.15 * data[:-1]
        return data
    if fixture_id == "S12D_ORDINARY_COUPLED_VAR_B":
        innovations = rng.normal(size=(n, d))
        data = np.zeros((n, d), dtype=np.float64)
        for index in range(1, n):
            data[index] = 0.48 * data[index - 1] + innovations[index]
            data[index, 5:] += 0.31 * data[index - 1, :5]
        return data
    if fixture_id == "S12D_SINGULAR_DUPLICATE_A":
        base = rng.normal(size=(n, 5))
        return np.column_stack((base, base))
    if fixture_id == "S12D_SINGULAR_LOWRANK_B":
        latent = rng.normal(size=(n, 2))
        mixing = rng.normal(size=(2, d))
        return latent @ mixing
    if fixture_id == "S12D_NEAR_SINGULAR_A":
        base = rng.normal(size=(n, 5))
        return np.column_stack((base, base + rng.normal(scale=1e-12, size=base.shape)))
    if fixture_id == "S12D_NEAR_SINGULAR_B":
        base = rng.normal(size=(n, 5))
        return np.column_stack((base, base + rng.normal(scale=1e-9, size=base.shape)))
    raise ValueError(f"unregistered S12D confirmation fixture: {fixture_id}")


def all_metric_identity_fixtures() -> list[tuple[str, str, NDArray[np.float64]]]:
    """Return S12C development/confirmation fixtures plus six new fixtures."""

    fixtures: list[tuple[str, str, NDArray[np.float64]]] = []
    roots = {
        "S12C_DEVELOPMENT": "d312c0d312c0d312c0d312c0d312c0d312c0d312c0d312c0d312c0d312c0d3",
        "S12C_CONFIRMATION": "c012c0c012c0c012c0c012c0c012c0c012c0c012c0c012c0c012c0c012c0c0",
    }
    s12c_ids = (
        "COUPLED_GAUSSIAN",
        "COUPLED_AUTOREGRESSIVE",
        "CONSTANT_INPUT",
        "SINGULAR_DUPLICATE_INPUT",
        "NEAR_SINGULAR_DUPLICATE_INPUT",
        "LOW_RANK_LINEAR_COMBINATION_INPUT",
        "REPLAY_PARTIAL_CONSTANT_INPUT",
    )
    for suite, root in roots.items():
        phase = "development" if suite.endswith("DEVELOPMENT") else "confirmation"
        for fixture_id in s12c_ids:
            fixtures.append(
                (suite, fixture_id, s12c_fixture_array(fixture_id, phase, root))
            )
    for fixture_id in NEW_FIXTURE_IDS:
        fixtures.append(
            (
                "S12D_UNTOUCHED_CONFIRMATION",
                fixture_id,
                confirmation_fixture_array(fixture_id),
            )
        )
    return fixtures


def confirmation_seed_bundle(matrix_index: int) -> SeedBundle:
    """Derive one of exactly 24 trajectory-isolated S12D nine-stream bundles."""

    if matrix_index not in range(24):
        raise ValueError("S12D confirmation matrix_index must be in 0..23")
    trajectory_id = f"E01-S12D-C{matrix_index:02d}"
    namespace = isolated_stream_namespace(
        experiment_id="E01",
        specification_id=GARD_SPECIFICATION_ID,
        trajectory_id=trajectory_id,
        replicate_index=matrix_index,
    )
    return derive_seed_bundle(
        SeedRequest(
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
    )


def simulate_confirmation_trajectory(matrix_index: int) -> BaselineTrajectory:
    """Run the unchanged S12 GARD branch under the new S12D seed identity."""

    specification = replace(
        build_baseline_specification(), specification_id=GARD_SPECIFICATION_ID
    )
    bundle = confirmation_seed_bundle(matrix_index)
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
        trajectory_id=f"E01-S12D-C{matrix_index:02d}",
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


def source_pipeline_seeds(
    implementation: SourceImplementation,
    trajectory_id: str,
    mode_id: str,
    endpoint: int,
) -> tuple[int, int]:
    """Separate preprocessing-noise and Fiedler initialization identities."""

    return (
        derive_legacy_seed(
            "source_pipeline",
            implementation.value,
            trajectory_id,
            mode_id,
            endpoint,
            "preprocessing",
        ),
        derive_legacy_seed(
            "source_pipeline",
            implementation.value,
            trajectory_id,
            mode_id,
            endpoint,
            "partition",
        ),
    )


def statistics_seed(*identity: object) -> int:
    return derive_legacy_seed("statistics", *identity)
