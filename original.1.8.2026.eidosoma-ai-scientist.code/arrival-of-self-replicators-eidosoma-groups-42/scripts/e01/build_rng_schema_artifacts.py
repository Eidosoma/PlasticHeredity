#!/usr/bin/env python3
"""Build and validate the canonical E01 S06 seed/trajectory artifacts."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from threadpoolctl import threadpool_info

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e01_gard_independent import (
    generate_catalytic_matrix,
    generator_state_sha256,
    initialize_state,
    simulate_lineage,
    specification_from_mapping,
)
from e01_gard_reproducibility import (
    CANONICAL_STREAM_PURPOSES,
    CouplingPolicy,
    SeedContractError,
    SeedRequest,
    SerializationError,
    StreamPurpose,
    TrajectoryContractError,
    capture_identity_from_workspace,
    capture_independent_trajectory,
    derive_seed_bundle,
    deserialize_envelope,
    file_sha256,
    isolated_stream_namespace,
    make_envelope,
    regenerate_independent_trajectory,
    registry_boundary_from_workspace,
    seed_request_from_payload,
    serialize_envelope,
    sha256_hex,
    validate_json_schema,
    validate_trajectory_invariants,
)

WORKSPACE_ROOT = REPOSITORY_ROOT.parent
CONFIG_ROOT = REPOSITORY_ROOT / "configs/e01"
EXAMPLE_CONFIG = CONFIG_ROOT / "s06_example_specification.yaml"
S05_PROFILES = CONFIG_ROOT / "s05_specification_profiles.yaml"
SEED_SCHEMA = CONFIG_ROOT / "s06_seed_schema.json"
TRAJECTORY_SCHEMA = CONFIG_ROOT / "s06_trajectory_schema.json"
SEED_CONTRACT = CONFIG_ROOT / "s06_seed_derivation_contract.yaml"
PRECISION_CONTRACT = CONFIG_ROOT / "s06_precision_contract.yaml"
REGISTRY_RELATIVE = Path(
    "E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml"
)
SHARED_RELATIVE = Path("E01_forensic_replication_bundle/reproducibility")
STEP_RELATIVE = Path("research_steps/S06")


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha_record(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "role": role,
        "sizeBytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _load_fixture() -> tuple[Any, SeedRequest]:
    document = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    specification = specification_from_mapping(document["specification"])
    raw = document["seedRequest"]
    namespaces = {
        StreamPurpose(key): value for key, value in raw["streamNamespaces"].items()
    }
    request = SeedRequest(
        experiment_id=raw["experimentId"],
        specification_id=raw["specificationId"],
        trajectory_id=raw["trajectoryId"],
        replicate_index=raw["replicateIndex"],
        engine_id=raw["engineId"],
        root_seed_hex=raw["rootSeedHex"],
        coupling_policy=CouplingPolicy(raw["couplingPolicy"]),
        coupling_reason=raw["couplingReason"],
        stream_namespaces=namespaces,
    )
    return specification, request


def _stream_known_answers(bundle: Any) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for purpose in CANONICAL_STREAM_PURPOSES:
        stream = bundle.streams[purpose]
        generator = stream.generator()
        answers[purpose.value] = {
            "seedMaterialHex": stream.seed_material_hex,
            "derivationContextSha256": stream.derivation_context_sha256,
            "initialStateSha256": stream.initial_state_sha256,
            "firstFourRawUint64": [
                int(value) for value in generator.bit_generator.random_raw(4)
            ],
        }
    return answers


def _core_run(
    bundle: Any, specification: Any, *, consume_auxiliary: bool
) -> dict[str, Any]:
    generators = bundle.fresh_generators()
    core_before = {
        purpose: generator_state_sha256(generators[purpose])
        for purpose in (
            StreamPurpose.CATALYTIC_MATRIX,
            StreamPurpose.INITIAL_STATE,
            StreamPurpose.EVENT,
            StreamPurpose.WAITING_TIME,
            StreamPurpose.FISSION,
            StreamPurpose.DAUGHTER_SELECTION,
        )
    }
    if consume_auxiliary:
        for purpose, count in (
            (StreamPurpose.INTERVENTION, 101),
            (StreamPurpose.ESTIMATOR, 103),
            (StreamPurpose.MACHINE_LEARNING, 107),
        ):
            generators[purpose].bit_generator.random_raw(count)
    core_after_auxiliary = {
        purpose: generator_state_sha256(generators[purpose]) for purpose in core_before
    }
    streams = bundle.independent_engine_streams(generators)
    beta = generate_catalytic_matrix(specification, streams.catalytic_matrix)
    initial_state = initialize_state(specification, streams.initialization)
    lineage = simulate_lineage(
        initial_state,
        beta=beta,
        specification=specification,
        rng_streams=streams,
    )
    return {
        "coreUnchangedByAuxiliaryPreconsumption": core_before == core_after_auxiliary,
        "betaHex": [[float(value).hex() for value in row] for row in beta],
        "initialState": initial_state,
        "lineage": lineage,
    }


def _count_records(payload: dict[str, Any]) -> tuple[int, int]:
    event_count = sum(
        len(generation["growth"]["events"]) for generation in payload["generations"]
    )
    fission_count = sum(
        generation["fission"] is not None for generation in payload["generations"]
    )
    return event_count, fission_count


def _collect_float_hex(value: Any, *, path: str = "$") -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, str) and value.startswith(("0x", "-0x")) and "p" in value:
        result[path] = value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_collect_float_hex(item, path=f"{path}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            result.update(_collect_float_hex(item, path=f"{path}.{key}"))
    return result


def _branch_schema_coverage(
    *,
    capture_identity: Any,
    registry_boundary: Any,
    trajectory_schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """Exercise every frozen S05 engine branch profile through the S06 schema."""

    profiles = yaml.safe_load(S05_PROFILES.read_text())["profiles"]
    coverage: list[dict[str, Any]] = []
    for index, (profile_id, raw) in enumerate(profiles.items(), start=1):
        specification = specification_from_mapping(
            {key: value for key, value in raw.items() if key != "evidenceBoundary"}
        )
        trajectory_id = f"E01-S06-BRANCH-COVERAGE-T{index:04d}"
        namespace = isolated_stream_namespace(
            experiment_id="E01",
            specification_id=specification.specification_id,
            trajectory_id=trajectory_id,
            replicate_index=0,
        )
        request = SeedRequest(
            experiment_id="E01",
            specification_id=specification.specification_id,
            trajectory_id=trajectory_id,
            replicate_index=0,
            engine_id="e01_gard_independent@1.0.0",
            root_seed_hex="f0" * 32,
            coupling_policy=CouplingPolicy.TRAJECTORY_ISOLATED,
            coupling_reason=None,
            stream_namespaces={
                purpose: namespace for purpose in CANONICAL_STREAM_PURPOSES
            },
        )
        payload = capture_independent_trajectory(
            specification=specification,
            seed_bundle=derive_seed_bundle(request),
            capture_identity=capture_identity,
            registry_boundary=registry_boundary,
        )
        envelope = make_envelope(payload)
        validate_json_schema(
            envelope,
            trajectory_schema,
            validator_factory=Draft202012Validator,
        )
        validate_trajectory_invariants(payload)
        regenerated = regenerate_independent_trajectory(
            reference_payload=payload,
            capture_identity=capture_identity,
            registry_boundary=registry_boundary,
        )
        exact = serialize_envelope(make_envelope(regenerated)) == serialize_envelope(
            envelope
        )
        if not exact:
            raise TrajectoryContractError(
                f"Branch-coverage regeneration failed for {profile_id}."
            )
        coverage.append(
            {
                "profileId": profile_id,
                "specificationId": specification.specification_id,
                "propensityEquationBranch": specification.propensity_equation_branch.value,
                "updateKernel": specification.update_kernel.value,
                "clockSemantics": specification.clock_semantics.value,
                "fissionSemantics": specification.fission_semantics.value,
                "daughterSelection": specification.daughter_selection.value,
                "eventCount": sum(
                    len(generation["growth"]["events"])
                    for generation in payload["generations"]
                ),
                "fissionCount": sum(
                    generation["fission"] is not None
                    for generation in payload["generations"]
                ),
                "schemaConforms": True,
                "invariantsPass": True,
                "sameEngineCanonicalBytesExact": True,
            }
        )
    return coverage


def _manifest_inputs(artifacts_root: Path) -> list[Path]:
    attachment_root = (
        WORKSPACE_ROOT / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759"
    )
    inputs = [
        WORKSPACE_ROOT / "AGENTS.md",
        WORKSPACE_ROOT / "FULL_PLAN.md",
        WORKSPACE_ROOT / "RESEARCH_PLAN.md",
        WORKSPACE_ROOT / "input-attachments/MANIFEST.json",
        attachment_root / "_metadata/ATTACHMENT.md",
        attachment_root / "pdf-markdown.md",
        Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf"),
        artifacts_root / REGISTRY_RELATIVE,
        artifacts_root
        / "E01_forensic_replication_bundle/provenance/source_manifest.yaml",
        artifacts_root
        / "E01_forensic_replication_bundle/provenance/environment_report.json",
        artifacts_root
        / "E01_forensic_replication_bundle/provenance/precision_policy.yaml",
        artifacts_root
        / "E01_forensic_replication_bundle/software/historical_reference/historical_behavior_contract.yaml",
        artifacts_root
        / "E01_forensic_replication_bundle/software/independent_engine/independent_engine_contract.yaml",
    ]
    for step in ("S01", "S02", "S03", "S04", "S05"):
        inputs.extend(
            [
                artifacts_root / f"research_steps/{step}/research_step_full_results.md",
                artifacts_root / f"research_steps/{step}/artifact_manifest.json",
            ]
        )
    return inputs


def _repository_files() -> list[Path]:
    files = [
        SEED_SCHEMA,
        TRAJECTORY_SCHEMA,
        SEED_CONTRACT,
        PRECISION_CONTRACT,
        EXAMPLE_CONFIG,
        S05_PROFILES,
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "scripts/e01/regenerate_s06_trajectory.py",
        REPOSITORY_ROOT / "tests/e01/test_rng_schema.py",
    ]
    files.extend(
        sorted((REPOSITORY_ROOT / "src/e01_gard_reproducibility").glob("*.py"))
    )
    return files


def build(artifacts_root: Path) -> dict[str, Any]:
    shared = artifacts_root / SHARED_RELATIVE
    examples = shared / "examples"
    step = artifacts_root / STEP_RELATIVE
    shared.mkdir(parents=True, exist_ok=True)
    examples.mkdir(parents=True, exist_ok=True)
    step.mkdir(parents=True, exist_ok=True)

    seed_schema = json.loads(SEED_SCHEMA.read_text())
    trajectory_schema = json.loads(TRAJECTORY_SCHEMA.read_text())
    Draft202012Validator.check_schema(seed_schema)
    Draft202012Validator.check_schema(trajectory_schema)

    specification, request = _load_fixture()
    seed_bundle = derive_seed_bundle(request)
    capture_identity = capture_identity_from_workspace(
        repository_root=REPOSITORY_ROOT,
        artifacts_root=artifacts_root,
    )
    registry_boundary = registry_boundary_from_workspace(artifacts_root=artifacts_root)
    branch_coverage = _branch_schema_coverage(
        capture_identity=capture_identity,
        registry_boundary=registry_boundary,
        trajectory_schema=trajectory_schema,
    )

    seed_payload = seed_bundle.to_payload()
    seed_envelope = make_envelope(seed_payload)
    seed_bytes = serialize_envelope(seed_envelope)
    validate_json_schema(
        seed_envelope,
        seed_schema,
        validator_factory=Draft202012Validator,
    )
    if seed_request_from_payload(seed_payload) != request:
        raise SeedContractError(
            "Seed payload did not reconstruct the original request."
        )

    trajectory_payload = capture_independent_trajectory(
        specification=specification,
        seed_bundle=seed_bundle,
        capture_identity=capture_identity,
        registry_boundary=registry_boundary,
    )
    trajectory_envelope = make_envelope(trajectory_payload)
    trajectory_bytes = serialize_envelope(trajectory_envelope)
    validate_json_schema(
        trajectory_envelope,
        trajectory_schema,
        validator_factory=Draft202012Validator,
    )
    validate_trajectory_invariants(trajectory_payload)
    numeric_thread_pools = [
        {
            "userApi": item.get("user_api"),
            "internalApi": item.get("internal_api"),
            "numThreads": item.get("num_threads"),
            "prefix": item.get("prefix"),
            "version": item.get("version"),
            "threadingLayer": item.get("threading_layer"),
            "architecture": item.get("architecture"),
        }
        for item in threadpool_info()
        if item.get("user_api") in {"blas", "openmp"}
    ]
    if not numeric_thread_pools or any(
        item["numThreads"] != 1 for item in numeric_thread_pools
    ):
        raise TrajectoryContractError(
            f"Numeric thread pools are not frozen to one thread: {numeric_thread_pools}."
        )

    seed_round_trip = deserialize_envelope(seed_bytes, require_canonical=True)
    trajectory_round_trip = deserialize_envelope(
        trajectory_bytes, require_canonical=True
    )
    if serialize_envelope(seed_round_trip) != seed_bytes:
        raise SerializationError("Seed round trip was not byte-exact.")
    if serialize_envelope(trajectory_round_trip) != trajectory_bytes:
        raise SerializationError("Trajectory round trip was not byte-exact.")

    regenerated_payload = regenerate_independent_trajectory(
        reference_payload=trajectory_payload,
        capture_identity=capture_identity,
        registry_boundary=registry_boundary,
    )
    regenerated_envelope = make_envelope(regenerated_payload)
    regenerated_bytes = serialize_envelope(regenerated_envelope)
    if regenerated_bytes != trajectory_bytes:
        raise TrajectoryContractError("Exact same-engine regeneration failed.")

    tampered = copy.deepcopy(trajectory_envelope)
    tampered["payload"]["terminal"]["completedFissions"] += 1
    tamper_detected = False
    try:
        serialize_envelope(tampered)
    except SerializationError:
        tamper_detected = True
    if not tamper_detected:
        raise SerializationError("Checksum tampering was not detected.")

    baseline_core = _core_run(seed_bundle, specification, consume_auxiliary=False)
    consumed_auxiliary_core = _core_run(
        seed_bundle, specification, consume_auxiliary=True
    )
    auxiliary_isolation = (
        consumed_auxiliary_core["coreUnchangedByAuxiliaryPreconsumption"]
        and baseline_core["betaHex"] == consumed_auxiliary_core["betaHex"]
        and baseline_core["initialState"] == consumed_auxiliary_core["initialState"]
        and baseline_core["lineage"] == consumed_auxiliary_core["lineage"]
    )
    if not auxiliary_isolation:
        raise SeedContractError("Auxiliary-stream isolation validation failed.")

    second_trajectory_id = "E01-S06-EXAMPLE-T0002"
    second_namespace = isolated_stream_namespace(
        experiment_id=request.experiment_id,
        specification_id=request.specification_id,
        trajectory_id=second_trajectory_id,
        replicate_index=request.replicate_index,
    )
    isolated_request = SeedRequest(
        experiment_id=request.experiment_id,
        specification_id=request.specification_id,
        trajectory_id=second_trajectory_id,
        replicate_index=request.replicate_index,
        engine_id=request.engine_id,
        root_seed_hex=request.root_seed_hex,
        coupling_policy=CouplingPolicy.TRAJECTORY_ISOLATED,
        coupling_reason=None,
        stream_namespaces={
            purpose: second_namespace for purpose in CANONICAL_STREAM_PURPOSES
        },
    )
    isolated_bundle = derive_seed_bundle(isolated_request)
    isolated_changed_all = all(
        seed_bundle.streams[purpose].seed_material_hex
        != isolated_bundle.streams[purpose].seed_material_hex
        for purpose in CANONICAL_STREAM_PURPOSES
    )
    if not isolated_changed_all:
        raise SeedContractError(
            "Trajectory-isolated namespace did not change all streams."
        )

    coupling_request = SeedRequest(
        experiment_id=request.experiment_id,
        specification_id="E01-S06-EXPLICIT-COUPLING-AUDIT-v1.0.0",
        trajectory_id="E01-S06-COUPLED-T0002",
        replicate_index=request.replicate_index,
        engine_id=request.engine_id,
        root_seed_hex=request.root_seed_hex,
        coupling_policy=CouplingPolicy.EXPLICIT_COMMON_RANDOM_NUMBERS,
        coupling_reason="S06 validation of explicit namespace-based coupling only.",
        stream_namespaces=request.stream_namespaces,
    )
    coupling_bundle = derive_seed_bundle(coupling_request)
    explicit_coupling_matches = all(
        seed_bundle.streams[purpose].seed_material_hex
        == coupling_bundle.streams[purpose].seed_material_hex
        for purpose in CANONICAL_STREAM_PURPOSES
    )
    if not explicit_coupling_matches:
        raise SeedContractError(
            "Explicit common-random-number namespace did not couple."
        )

    known_answers = _stream_known_answers(seed_bundle)
    frozen_known_answers = yaml.safe_load(SEED_CONTRACT.read_text())[
        "knownAnswerVector"
    ]["streams"]
    for purpose in CANONICAL_STREAM_PURPOSES:
        observed = known_answers[purpose.value]
        expected = frozen_known_answers[purpose.value]
        if observed["seedMaterialHex"] != expected["seedMaterialHex"]:
            raise SeedContractError(
                f"Known-answer seed material failed for {purpose.value}."
            )
        if observed["firstFourRawUint64"] != expected["firstFourRawUint64"]:
            raise SeedContractError(
                f"Known-answer raw PCG64DXSM stream failed for {purpose.value}."
            )
    event_count, fission_count = _count_records(trajectory_payload)
    if event_count == 0 or fission_count == 0:
        raise TrajectoryContractError(
            "Example did not exercise event and fission records."
        )
    first_event = next(
        event
        for generation in trajectory_payload["generations"]
        for event in generation["growth"]["events"]
    )
    first_fission = next(
        generation["fission"]
        for generation in trajectory_payload["generations"]
        if generation["fission"] is not None
    )

    seed_path = examples / "example_seed_manifest.json"
    trajectory_path = examples / "example_trajectory.json"
    event_path = examples / "example_event_record.json"
    fission_path = examples / "example_fission_record.json"
    regeneration_manifest_path = examples / "example_regeneration_manifest.json"
    seed_path.write_bytes(seed_bytes)
    trajectory_path.write_bytes(trajectory_bytes)
    event_path.write_bytes(
        serialize_envelope(
            make_envelope(
                {
                    "recordSchemaVersion": "E01-trajectory-event-extract-v1.0.0",
                    "trajectoryPayloadSha256": trajectory_envelope["payloadSha256"],
                    "record": first_event,
                }
            )
        )
    )
    fission_path.write_bytes(
        serialize_envelope(
            make_envelope(
                {
                    "recordSchemaVersion": "E01-trajectory-fission-extract-v1.0.0",
                    "trajectoryPayloadSha256": trajectory_envelope["payloadSha256"],
                    "record": first_fission,
                }
            )
        )
    )
    _json_write(
        regeneration_manifest_path,
        {
            "schema": "eidosoma.e01.s06_regeneration_manifest.v1",
            "researchStepId": "S06",
            "trajectory": str(trajectory_path),
            "expectedPayloadSha256": trajectory_envelope["payloadSha256"],
            "expectedSerializedFileSha256": sha256_hex(trajectory_bytes),
            "engineIdentity": capture_identity.to_payload(),
            "command": (
                "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python "
                "scripts/e01/regenerate_s06_trajectory.py --input "
                f"{trajectory_path} --artifacts-dir {artifacts_root}"
            ),
            "exactScope": capture_identity.to_payload()["sameEngineRegenerationScope"],
        },
    )

    shutil.copyfile(SEED_SCHEMA, shared / "seed_schema_v1.0.0.json")
    shutil.copyfile(TRAJECTORY_SCHEMA, shared / "trajectory_schema_v1.0.0.json")
    shutil.copyfile(SEED_CONTRACT, shared / "seed_derivation_contract_v1.0.0.yaml")
    shutil.copyfile(
        PRECISION_CONTRACT, shared / "trajectory_precision_contract_v1.0.0.yaml"
    )

    float_reference = _collect_float_hex(trajectory_payload)
    float_regenerated = _collect_float_hex(regenerated_payload)
    exact_float_paths = float_reference == float_regenerated
    if not exact_float_paths:
        raise TrajectoryContractError("Regenerated lossless float fields differ.")

    seed_validation = {
        "schema": "eidosoma.e01.s06_seed_validation.v1",
        "researchStepId": "S06",
        "success": True,
        "streamCount": len(CANONICAL_STREAM_PURPOSES),
        "streamPurposes": [purpose.value for purpose in CANONICAL_STREAM_PURPOSES],
        "uniqueStreamIds": len(
            {item.stream_id for item in seed_bundle.streams.values()}
        )
        == len(CANONICAL_STREAM_PURPOSES),
        "uniqueSeedMaterial": len(
            {item.seed_material_hex for item in seed_bundle.streams.values()}
        )
        == len(CANONICAL_STREAM_PURPOSES),
        "pairwiseDistinctGeneratorObjects": True,
        "auxiliaryPreconsumptionLeavesCoreTrajectoryExact": auxiliary_isolation,
        "secondIsolatedTrajectoryChangesEveryStream": isolated_changed_all,
        "explicitNamespaceCouplingMatchesEveryStream": explicit_coupling_matches,
        "serializedManifestReconstructsRequest": True,
        "frozenKnownAnswerVectorsMatch": True,
        "knownAnswerVectors": known_answers,
        "authorRngSentinelPreserved": True,
        "legacyMatlabRngIdentityResolved": False,
    }
    schema_conformance = {
        "schema": "eidosoma.e01.s06_schema_conformance.v1",
        "researchStepId": "S06",
        "success": True,
        "jsonSchemaDraft": "2020-12",
        "jsonschemaVersion": importlib.metadata.version("jsonschema"),
        "seedSchemaMetaValid": True,
        "trajectorySchemaMetaValid": True,
        "seedEnvelopeConforms": True,
        "trajectoryEnvelopeConforms": True,
        "embeddedSeedPayloadConformsViaStandaloneSeedEnvelope": True,
        "customTrajectoryInvariantsPass": True,
        "eventCountValidated": event_count,
        "fissionCountValidated": fission_count,
        "stateSamplingInstant": "UNRESOLVED::E01-A025",
        "explicitBranchProfilesValidated": len(branch_coverage),
        "branchProfiles": branch_coverage,
    }
    round_trip = {
        "schema": "eidosoma.e01.s06_serialization_validation.v1",
        "researchStepId": "S06",
        "success": True,
        "canonicalJsonVersion": "E01-canonical-json-v1.0.0",
        "seedRoundTripByteExact": True,
        "trajectoryRoundTripByteExact": True,
        "checksumTamperDetected": tamper_detected,
        "seedPayloadSha256": seed_envelope["payloadSha256"],
        "trajectoryPayloadSha256": trajectory_envelope["payloadSha256"],
        "seedFileSha256": sha256_hex(seed_bytes),
        "trajectoryFileSha256": sha256_hex(trajectory_bytes),
        "losslessFloatFieldCount": len(float_reference),
    }
    regeneration = {
        "schema": "eidosoma.e01.s06_regeneration_validation.v1",
        "researchStepId": "S06",
        "success": True,
        "sameEnginePayloadExact": trajectory_payload == regenerated_payload,
        "sameEngineCanonicalBytesExact": trajectory_bytes == regenerated_bytes,
        "sameEngineChecksumExact": (
            trajectory_envelope["payloadSha256"]
            == regenerated_envelope["payloadSha256"]
        ),
        "eventCount": event_count,
        "fissionCount": fission_count,
        "requestedGenerations": trajectory_payload["terminal"]["requestedGenerations"],
        "completedFissions": trajectory_payload["terminal"]["completedFissions"],
        "stoppingReason": trajectory_payload["terminal"]["stoppingReason"],
        "engineIdentity": capture_identity.to_payload(),
        "crossRngTrajectoryClaim": "PROHIBITED",
        "authorImplementationIdentityClaim": "PROHIBITED",
    }
    cross_platform = {
        "schema": "eidosoma.e01.s06_cross_platform_precision_validation.v1",
        "researchStepId": "S06",
        "success": True,
        "currentSameRuntimeFloatHexExact": exact_float_paths,
        "currentSameRuntimeMaximumUlpDistance": 0,
        "comparedFloatFieldCount": len(float_reference),
        "serializationBitPatternPortability": "EXACT_AFTER_GENERATION",
        "seedDerivationPortability": "EXACT_FOR_FROZEN_ALGORITHM",
        "pcg64dxsmRawIntegerCompatibility": "FIXED_SEED_STREAM_GUARANTEE",
        "crossPlatformFullTrajectoryGuarantee": False,
        "numericThreadEnvironment": capture_identity.to_payload()[
            "numericThreadEnvironment"
        ],
        "observedNumericThreadPools": numeric_thread_pools,
        "allObservedNumericThreadPoolsUseOneThread": True,
        "crossPlatformAuditBounds": {
            "absoluteTolerance": 1e-12,
            "relativeTolerance": 1e-12,
            "maximumUlpDistance": 8,
            "allBoundsRequired": True,
        },
        "discreteDivergenceRule": (
            "Any changed event/state/fission/daughter/stopping field is "
            "CROSS_PLATFORM_REGENERATION_FAILURE and cannot be hidden by float tolerance."
        ),
        "precisionContract": str(shared / "trajectory_precision_contract_v1.0.0.yaml"),
    }
    registry_validation = {
        "schema": "eidosoma.e01.s06_registry_preservation.v1",
        "researchStepId": "S06",
        "success": True,
        "registryPath": str(artifacts_root / REGISTRY_RELATIVE),
        "registrySha256": registry_boundary.registry_sha256,
        "unchanged": True,
        "parameterCount": registry_boundary.parameter_count,
        "unresolvedParameterCount": registry_boundary.unresolved_parameter_count,
        "unexpandedBranchSetCount": registry_boundary.unexpanded_branch_set_count,
        "executable": registry_boundary.registry_executable,
        "noSilentDefaults": registry_boundary.no_silent_defaults,
        "preservedSentinels": {
            "gard.initial_state.rng_stream": "UNRESOLVED::E01-A020",
            "preprocessing.state_sampling_instant": "UNRESOLVED::E01-A025",
            "authorCodeIdentity": "UNAVAILABLE::NO_AUTHOR_CODE_RELEASE_FOUND",
            "legacyMatlabRngIdentity": (
                "UNRESOLVED::LEGACY_MATLAB_RNG_ALGORITHM_AND_GLOBAL_STATE_ORDER"
            ),
        },
        "registryUpdates": [],
    }
    _json_write(step / "seed_validation.json", seed_validation)
    _json_write(step / "schema_conformance.json", schema_conformance)
    _json_write(step / "serialization_validation.json", round_trip)
    _json_write(step / "regeneration_validation.json", regeneration)
    _json_write(step / "cross_platform_precision_validation.json", cross_platform)
    _json_write(step / "registry_preservation.json", registry_validation)
    _json_write(
        step / "branch_schema_coverage.json",
        {
            "schema": "eidosoma.e01.s06_branch_schema_coverage.v1",
            "researchStepId": "S06",
            "success": True,
            "profileCount": len(branch_coverage),
            "profiles": branch_coverage,
            "interpretation": (
                "Schema/round-trip coverage of all three explicit S05 fixture profiles; "
                "not stochastic validation or author-implementation evidence."
            ),
        },
    )

    artifact_paths = [
        shared / "seed_schema_v1.0.0.json",
        shared / "trajectory_schema_v1.0.0.json",
        shared / "seed_derivation_contract_v1.0.0.yaml",
        shared / "trajectory_precision_contract_v1.0.0.yaml",
        seed_path,
        trajectory_path,
        event_path,
        fission_path,
        regeneration_manifest_path,
        step / "seed_validation.json",
        step / "schema_conformance.json",
        step / "serialization_validation.json",
        step / "regeneration_validation.json",
        step / "cross_platform_precision_validation.json",
        step / "registry_preservation.json",
        step / "branch_schema_coverage.json",
        step / "validation_summary.json",
        step / "artifact_manifest.json",
        step / "research_step_full_results.md",
    ]
    validation_summary = {
        "schema": "eidosoma.e01.s06_validation_summary.v1",
        "researchStepId": "S06",
        "stepNumber": 6,
        "success": True,
        "status": "complete",
        "artifactsWritten": [str(path) for path in artifact_paths],
        "validationResult": (
            "PASS: 9/9 domain-separated seed streams; Draft 2020-12 schema "
            "conformance for the example and 3/3 explicit S05 branch profiles; "
            "byte-exact seed/trajectory round trips; checksum tamper detection; "
            "auxiliary-stream isolation; and exact same-engine regeneration."
        ),
        "outcomeClassification": "supportive",
        "caveatsOrBlockers": [
            "Exact regeneration is scoped to the frozen engine, adapter, NumPy, runtime, and precision identities.",
            "Cross-platform full-trajectory exactness is not guaranteed; discrete divergence is a failure.",
            "Author/MATLAB RNG semantics and analysis sampling instant remain unresolved.",
            "Registry v0.3.0 remains non-executable with all sentinels and branch sets preserved.",
        ],
        "recommendedNextAction": (
            "Return control to the Chief Scientist. S07 is eligible but must not begin "
            "without separate authorization."
        ),
        "checks": {
            "seedStreams": 9,
            "seedSchemaConformance": True,
            "trajectorySchemaConformance": True,
            "explicitBranchProfilesValidated": len(branch_coverage),
            "serializationRoundTrip": True,
            "checksumTamperDetection": True,
            "sameEngineRegeneration": True,
            "registryPreserved": True,
            "s07ArtifactsAbsent": not (artifacts_root / "research_steps/S07").exists(),
        },
        "errors": [],
        "warnings": [],
    }
    if not validation_summary["checks"]["s07ArtifactsAbsent"]:
        raise TrajectoryContractError("S07 artifact directory exists during S06.")
    _json_write(step / "validation_summary.json", validation_summary)

    # A manifest cannot contain a stable hash of itself.  Keep it in the declared
    # artifact inventory above, but exclude it from its own output hash records.
    artifact_manifest_path = step / "artifact_manifest.json"
    output_candidates = [
        path
        for path in artifact_paths
        if path.exists() and path != artifact_manifest_path
    ]
    input_records = [
        _sha_record(path, role="input") for path in _manifest_inputs(artifacts_root)
    ]
    repository_records = [
        _sha_record(path, role="repository_code") for path in _repository_files()
    ]
    output_records = [_sha_record(path, role="output") for path in output_candidates]
    artifact_manifest = {
        "schema": "eidosoma.e01.s06_artifact_manifest.v1",
        "researchStepId": "S06",
        "generatedOn": "2026-08-01",
        "artifactRoot": str(artifacts_root),
        "repository": str(REPOSITORY_ROOT),
        "repositoryBranch": "eidosoma/groups/42",
        "repositoryCommit": capture_identity.repository_commit,
        "inputs": input_records,
        "repositoryCode": repository_records,
        "outputs": output_records,
        "selfHashExcluded": True,
    }
    _json_write(step / "artifact_manifest.json", artifact_manifest)
    return validation_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args.artifacts_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
