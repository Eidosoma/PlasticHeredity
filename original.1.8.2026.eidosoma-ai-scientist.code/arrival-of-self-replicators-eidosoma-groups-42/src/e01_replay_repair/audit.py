"""Instrumented, behavior-preserving RNG and sequence audit for S12FR."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import e01_latent_timebase.core as frozen_core
from e01_latent_timebase.core import (
    SeedIdentity,
    SimulationDefinition,
    TimebaseTrajectory,
)

from .comparator import FieldDifference

AUDIT_VERSION = "E01-S12FR-RNG-SEQUENCE-AUDIT-v1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    if value is None:
        return b"N"
    if isinstance(value, bool):
        return b"B1" if value else b"B0"
    if isinstance(value, (int, np.integer)):
        return b"I" + str(int(value)).encode()
    if isinstance(value, (float, np.floating)):
        return b"F" + struct.pack(">d", float(value))
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return b"S" + str(len(encoded)).encode() + b":" + encoded
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (
            b"A"
            + str(array.dtype).encode()
            + b"|"
            + json.dumps(array.shape, separators=(",", ":")).encode()
            + b"|"
            + array.tobytes(order="C")
        )
    if isinstance(value, (tuple, list)):
        return b"L" + b"".join(
            len(payload).to_bytes(8, "big")
            + payload
            for payload in (_canonical_bytes(item) for item in value)
        )
    if isinstance(value, dict):
        pieces = []
        for key in sorted(value, key=str):
            key_payload = _canonical_bytes(str(key))
            value_payload = _canonical_bytes(value[key])
            pieces.append(
                len(key_payload).to_bytes(8, "big")
                + key_payload
                + len(value_payload).to_bytes(8, "big")
                + value_payload
            )
        return b"D" + b"".join(pieces)
    raise TypeError(f"unsupported canonical value {type(value)!r}")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _numeric_counts(value: Any) -> tuple[int, int, int]:
    arrays: list[NDArray[np.generic]] = []

    def collect(item: Any) -> None:
        if isinstance(item, np.ndarray):
            arrays.append(item)
        elif isinstance(item, (float, np.floating, int, np.integer)) and not isinstance(item, bool):
            arrays.append(np.asarray([item]))
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)

    collect(value)
    float_count = 0
    finite_float_count = 0
    nonfinite_float_count = 0
    for array in arrays:
        if np.issubdtype(array.dtype, np.floating):
            float_count += int(array.size)
            finite_float_count += int(np.isfinite(array).sum())
            nonfinite_float_count += int((~np.isfinite(array)).sum())
    return float_count, finite_float_count, nonfinite_float_count


@dataclass(frozen=True, slots=True)
class RngCallRecord:
    call_index: int
    method: str
    argument_sha256: str
    result_sha256: str
    result_dtype: str
    result_shape: tuple[int, ...]
    result_element_count: int
    finite_float_argument_count: int
    nonfinite_float_argument_count: int
    finite_float_result_count: int
    nonfinite_float_result_count: int


class RecordingGenerator:
    """Exact delegate that records every RNG call used by the frozen simulator."""

    _ALLOWED = frozenset(
        {
            "standard_normal",
            "choice",
            "poisson",
            "multivariate_hypergeometric",
            "binomial",
            "integers",
        }
    )

    def __init__(self, identity: SeedIdentity):
        self.identity = identity
        self._generator = np.random.Generator(np.random.PCG64DXSM(identity.derived_seed))
        self.start_state_sha256 = canonical_sha256(self._generator.bit_generator.state)
        self.calls: list[RngCallRecord] = []
        self.raw_results: list[tuple[str, NDArray[np.generic]]] = []

    @property
    def bit_generator(self) -> np.random.BitGenerator:
        return self._generator.bit_generator

    @property
    def end_state_sha256(self) -> str:
        return canonical_sha256(self._generator.bit_generator.state)

    def _invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if method not in self._ALLOWED:
            raise RuntimeError(f"unregistered RNG method {method!r}")
        result = getattr(self._generator, method)(*args, **kwargs)
        result_array = np.asarray(result).copy()
        argument_float, argument_finite, argument_nonfinite = _numeric_counts(
            (args, kwargs)
        )
        result_float, result_finite, result_nonfinite = _numeric_counts(result_array)
        if argument_float != argument_finite + argument_nonfinite:
            raise AssertionError("RNG argument float accounting failed")
        if result_float != result_finite + result_nonfinite:
            raise AssertionError("RNG result float accounting failed")
        self.calls.append(
            RngCallRecord(
                call_index=len(self.calls),
                method=method,
                argument_sha256=canonical_sha256((args, kwargs)),
                result_sha256=canonical_sha256(result_array),
                result_dtype=str(result_array.dtype),
                result_shape=tuple(map(int, result_array.shape)),
                result_element_count=int(result_array.size),
                finite_float_argument_count=argument_finite,
                nonfinite_float_argument_count=argument_nonfinite,
                finite_float_result_count=result_finite,
                nonfinite_float_result_count=result_nonfinite,
            )
        )
        self.raw_results.append((method, result_array))
        return result

    def standard_normal(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("standard_normal", *args, **kwargs)

    def choice(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("choice", *args, **kwargs)

    def poisson(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("poisson", *args, **kwargs)

    def multivariate_hypergeometric(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("multivariate_hypergeometric", *args, **kwargs)

    def binomial(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("binomial", *args, **kwargs)

    def integers(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("integers", *args, **kwargs)


class RecordingFactory:
    def __init__(self) -> None:
        self.recorders: dict[str, RecordingGenerator] = {}

    def __call__(self, identity: SeedIdentity) -> RecordingGenerator:
        if identity.purpose in self.recorders:
            raise RuntimeError(f"RNG stream {identity.purpose!r} constructed twice")
        recorder = RecordingGenerator(identity)
        self.recorders[identity.purpose] = recorder
        return recorder


@dataclass(frozen=True, slots=True)
class AuditedSimulation:
    trajectory: TimebaseTrajectory
    seeds: tuple[SeedIdentity, ...]
    rng_manifest: tuple[dict[str, Any], ...]
    raw_results: dict[str, tuple[tuple[str, NDArray[np.generic]], ...]]
    trace_sha256: str


def _rng_manifest(factory: RecordingFactory) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for purpose in sorted(factory.recorders):
        recorder = factory.recorders[purpose]
        rows.append(
            {
                "purpose": purpose,
                "seed": asdict(recorder.identity),
                "startStateSha256": recorder.start_state_sha256,
                "endStateSha256": recorder.end_state_sha256,
                "callCount": len(recorder.calls),
                "calls": [asdict(call) for call in recorder.calls],
            }
        )
    return tuple(rows)


def _trajectory_sequence_payload(trajectory: TimebaseTrajectory) -> dict[str, Any]:
    return {
        "trajectoryId": trajectory.trajectory_id,
        "betaSha256": trajectory.beta_sha256,
        "initialStateSha256": trajectory.initial_state_sha256,
        "trajectorySha256": trajectory.trajectory_sha256,
        "terminalStatus": trajectory.terminal_status,
        "extinctionGeneration": trajectory.extinction_generation,
        "observations": [
            {
                "observationIndex": row.observation_index,
                "observationKind": row.observation_kind,
                "completedFissions": row.completed_fissions,
                "growthGenerationOneBased": row.growth_generation_one_based,
                "batchStep": row.batch_step,
                "generationLocalStep": row.generation_local_step,
                "state": row.state,
            }
            for row in trajectory.observations
        ],
        "generations": [asdict(row) for row in trajectory.generations],
    }


def simulate_audited(
    *,
    phase: str,
    root_hex: str,
    matrix_index: int,
    definition: SimulationDefinition,
    stream_identity: str,
) -> AuditedSimulation:
    factory = RecordingFactory()
    original_factory = frozen_core.generator
    frozen_core.generator = factory  # type: ignore[assignment]
    try:
        trajectory, seeds = frozen_core.simulate_trajectory(
            phase=phase,
            root_hex=root_hex,
            matrix_index=matrix_index,
            definition=definition,
            stream_identity=stream_identity,
        )
    finally:
        frozen_core.generator = original_factory  # type: ignore[assignment]
    manifest = _rng_manifest(factory)
    raw = {
        purpose: tuple(recorder.raw_results)
        for purpose, recorder in factory.recorders.items()
    }
    digest_payload = {
        "version": AUDIT_VERSION,
        "trajectory": _trajectory_sequence_payload(trajectory),
        "rng": manifest,
    }
    return AuditedSimulation(
        trajectory=trajectory,
        seeds=seeds,
        rng_manifest=manifest,
        raw_results=raw,
        trace_sha256=canonical_sha256(digest_payload),
    )


def compare_rng_manifests(
    left: tuple[dict[str, Any], ...], right: tuple[dict[str, Any], ...]
) -> tuple[bool, tuple[FieldDifference, ...]]:
    differences: list[FieldDifference] = []
    if len(left) != len(right):
        differences.append(
            FieldDifference(
                "rng",
                "RNG_STREAM_COUNT_DIVERGENCE",
                str(len(left)),
                str(len(right)),
                False,
                None,
            )
        )
        return False, tuple(differences)
    for stream_index, (left_stream, right_stream) in enumerate(
        zip(left, right, strict=True)
    ):
        if left_stream != right_stream:
            keys = sorted(set(left_stream) | set(right_stream))
            for key in keys:
                if left_stream.get(key) != right_stream.get(key):
                    differences.append(
                        FieldDifference(
                            f"rng[{stream_index}].{key}",
                            "RNG_IDENTITY_OR_CONSUMPTION_DIVERGENCE",
                            repr(left_stream.get(key))[:256],
                            repr(right_stream.get(key))[:256],
                            False,
                            None,
                        )
                    )
    return not differences, tuple(differences)


def _stack_results(
    raw: dict[str, tuple[tuple[str, NDArray[np.generic]], ...]],
    purpose: str,
    method: str,
    *,
    dtype: np.dtype[Any] | None = None,
    width: int = 100,
) -> NDArray[np.generic]:
    resolved_dtype = np.dtype("<i8") if dtype is None else dtype
    values = [
        np.asarray(result, dtype=resolved_dtype).reshape(-1)
        for call_method, result in raw.get(purpose, ())
        if call_method == method
    ]
    if not values:
        return np.empty((0, width), dtype=resolved_dtype)
    if any(value.size != width for value in values):
        raise AssertionError(f"unexpected {purpose}/{method} result width")
    return np.vstack(values)


def write_trace_payload(path: Path, audited: AuditedSimulation) -> dict[str, Any]:
    trajectory = audited.trajectory
    states = np.asarray([row.state for row in trajectory.observations], dtype="<i8")
    observation_meta = np.asarray(
        [
            [
                row.observation_index,
                row.completed_fissions,
                row.growth_generation_one_based,
                row.batch_step,
                row.generation_local_step,
            ]
            for row in trajectory.observations
        ],
        dtype="<i8",
    )
    observation_kind = np.asarray(
        [row.observation_kind for row in trajectory.observations], dtype="U32"
    )
    generation_integer = np.asarray(
        [
            [
                row.generation_one_based,
                row.update_count,
                row.nonzero_reaction_type_count,
                row.gross_sampled_event_count,
                -1 if row.pre_fission_mass is None else row.pre_fission_mass,
                -1 if row.post_fission_mass is None else row.post_fission_mass,
                -1 if row.child_a_mass is None else row.child_a_mass,
                -1 if row.child_b_mass is None else row.child_b_mass,
                -1 if row.overshoot_before_trim is None else row.overshoot_before_trim,
                row.trimmed_new_entrants,
            ]
            for row in trajectory.generations
        ],
        dtype="<i8",
    )
    generation_exposure = np.asarray(
        [[row.maximum_exposure, row.minimum_exposure] for row in trajectory.generations],
        dtype="<f8",
    )
    generation_status = np.asarray(
        [row.terminal_status for row in trajectory.generations], dtype="U32"
    )
    selected_daughter = np.asarray(
        ["NONE_NULL" if row.selected_daughter is None else row.selected_daughter for row in trajectory.generations],
        dtype="U16",
    )
    event_poisson = _stack_results(
        audited.raw_results, "poisson_update", "poisson"
    )
    if event_poisson.shape[0] != 2 * trajectory.total_batch_updates:
        raise AssertionError("Poisson call sequence is not exactly join/loss alternating")
    trim_draws = _stack_results(
        audited.raw_results, "overshoot_trim", "multivariate_hypergeometric"
    )
    fission_draws = _stack_results(audited.raw_results, "fission", "binomial")
    daughter_values = [
        int(np.asarray(result).item())
        for method, result in audited.raw_results.get("daughter_selection", ())
        if method == "integers"
    ]
    initial_choices = [
        np.asarray(result, dtype="<i8").reshape(-1)
        for method, result in audited.raw_results.get("initial_state", ())
        if method == "choice"
    ]
    initial_choice = (
        initial_choices[0]
        if len(initial_choices) == 1
        else np.empty((0,), dtype="<i8")
    )
    metadata = {
        "auditVersion": AUDIT_VERSION,
        "traceSha256": audited.trace_sha256,
        "trajectoryId": trajectory.trajectory_id,
        "betaSha256": trajectory.beta_sha256,
        "initialStateSha256": trajectory.initial_state_sha256,
        "trajectorySha256": trajectory.trajectory_sha256,
        "terminalStatus": trajectory.terminal_status,
        "extinctionGeneration": trajectory.extinction_generation,
        "rngManifest": audited.rng_manifest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=states,
        observation_meta=observation_meta,
        observation_kind=observation_kind,
        generation_integer=generation_integer,
        generation_exposure=generation_exposure,
        generation_status=generation_status,
        selected_daughter=selected_daughter,
        poisson_join_draws=event_poisson[0::2],
        poisson_attempted_loss_draws=event_poisson[1::2],
        trim_removed_draws=trim_draws,
        fission_child_a_draws=fission_draws,
        daughter_selection_draws=np.asarray(daughter_values, dtype="<i8"),
        initial_choice=initial_choice,
        metadata_json=np.frombuffer(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str).encode(),
            dtype=np.uint8,
        ),
    )
    return {
        "path": str(path),
        "sizeBytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "traceSha256": audited.trace_sha256,
        "observationCount": len(trajectory.observations),
        "generationCount": len(trajectory.generations),
        "updateCount": trajectory.total_batch_updates,
        "poissonCallCount": int(event_poisson.shape[0]),
        "trimCallCount": int(trim_draws.shape[0]),
        "fissionCallCount": int(fission_draws.shape[0]),
        "daughterDrawCount": len(daughter_values),
    }
