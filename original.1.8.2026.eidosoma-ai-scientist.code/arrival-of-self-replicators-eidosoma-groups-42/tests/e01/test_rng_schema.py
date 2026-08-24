from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from jsonschema import Draft202012Validator

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
    binary64_ulp_distance,
    canonical_json_bytes,
    capture_identity_from_workspace,
    capture_independent_trajectory,
    derive_seed_bundle,
    deserialize_envelope,
    float_bits_hex,
    float_from_hex,
    float_to_hex,
    isolated_stream_namespace,
    make_envelope,
    regenerate_independent_trajectory,
    registry_boundary_from_workspace,
    seed_request_from_payload,
    serialize_envelope,
    specification_from_payload,
    specification_to_payload,
    validate_json_schema,
    validate_trajectory_invariants,
)

ARTIFACTS_ROOT = Path("/artifacts")
CONFIG_ROOT = REPOSITORY_ROOT / "configs/e01"
EXAMPLE_CONFIG = CONFIG_ROOT / "s06_example_specification.yaml"
SEED_CONTRACT = CONFIG_ROOT / "s06_seed_derivation_contract.yaml"
SEED_SCHEMA = CONFIG_ROOT / "s06_seed_schema.json"
TRAJECTORY_SCHEMA = CONFIG_ROOT / "s06_trajectory_schema.json"


def _fixture() -> tuple[object, SeedRequest, object, object]:
    document = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    specification = specification_from_mapping(document["specification"])
    raw = document["seedRequest"]
    request = SeedRequest(
        experiment_id=raw["experimentId"],
        specification_id=raw["specificationId"],
        trajectory_id=raw["trajectoryId"],
        replicate_index=raw["replicateIndex"],
        engine_id=raw["engineId"],
        root_seed_hex=raw["rootSeedHex"],
        coupling_policy=CouplingPolicy(raw["couplingPolicy"]),
        coupling_reason=raw["couplingReason"],
        stream_namespaces={
            StreamPurpose(key): value for key, value in raw["streamNamespaces"].items()
        },
    )
    identity = capture_identity_from_workspace(
        repository_root=REPOSITORY_ROOT,
        artifacts_root=ARTIFACTS_ROOT,
    )
    registry = registry_boundary_from_workspace(artifacts_root=ARTIFACTS_ROOT)
    return specification, request, identity, registry


def _trajectory() -> tuple[dict[str, object], object, object]:
    specification, request, identity, registry = _fixture()
    payload = capture_independent_trajectory(
        specification=specification,
        seed_bundle=derive_seed_bundle(request),
        capture_identity=identity,
        registry_boundary=registry,
    )
    return payload, identity, registry


def test_seed_known_answer_vectors_and_all_required_domains() -> None:
    _, request, _, _ = _fixture()
    bundle = derive_seed_bundle(request)
    expected = yaml.safe_load(SEED_CONTRACT.read_text())["knownAnswerVector"]["streams"]
    assert tuple(bundle.streams) == CANONICAL_STREAM_PURPOSES
    assert len({item.stream_id for item in bundle.streams.values()}) == 9
    assert len({item.seed_material_hex for item in bundle.streams.values()}) == 9
    for purpose in CANONICAL_STREAM_PURPOSES:
        stream = bundle.streams[purpose]
        assert stream.seed_material_hex == expected[purpose.value]["seedMaterialHex"]
        raw = [int(value) for value in stream.generator().bit_generator.random_raw(4)]
        assert raw == expected[purpose.value]["firstFourRawUint64"]


def test_seed_request_is_fail_closed_and_coupling_is_explicit() -> None:
    _, request, _, _ = _fixture()
    missing = dict(request.stream_namespaces)
    missing.pop(StreamPurpose.ESTIMATOR)
    with pytest.raises(SeedContractError, match="all canonical purposes"):
        SeedRequest(
            experiment_id=request.experiment_id,
            specification_id=request.specification_id,
            trajectory_id=request.trajectory_id,
            replicate_index=request.replicate_index,
            engine_id=request.engine_id,
            root_seed_hex=request.root_seed_hex,
            coupling_policy=request.coupling_policy,
            coupling_reason=request.coupling_reason,
            stream_namespaces=missing,
        )

    mismatched = dict(request.stream_namespaces)
    mismatched[StreamPurpose.EVENT] = "urn:eidosoma:seed-namespace:E01:mismatch"
    with pytest.raises(SeedContractError, match="canonical trajectory namespace"):
        SeedRequest(
            experiment_id=request.experiment_id,
            specification_id=request.specification_id,
            trajectory_id=request.trajectory_id,
            replicate_index=request.replicate_index,
            engine_id=request.engine_id,
            root_seed_hex=request.root_seed_hex,
            coupling_policy=request.coupling_policy,
            coupling_reason=None,
            stream_namespaces=mismatched,
        )

    coupled = SeedRequest(
        experiment_id=request.experiment_id,
        specification_id="E01-S06-COUPLED-SPEC-v1.0.0",
        trajectory_id="E01-S06-COUPLED-T2",
        replicate_index=request.replicate_index,
        engine_id=request.engine_id,
        root_seed_hex=request.root_seed_hex,
        coupling_policy=CouplingPolicy.EXPLICIT_COMMON_RANDOM_NUMBERS,
        coupling_reason="Test-only explicit common-random-number namespace.",
        stream_namespaces=request.stream_namespaces,
    )
    original_bundle = derive_seed_bundle(request)
    coupled_bundle = derive_seed_bundle(coupled)
    assert all(
        original_bundle.streams[purpose].seed_material_hex
        == coupled_bundle.streams[purpose].seed_material_hex
        for purpose in CANONICAL_STREAM_PURPOSES
    )
    with pytest.raises(TypeError):
        request.stream_namespaces[StreamPurpose.EVENT] = "urn:eidosoma:mutated"
    with pytest.raises(TypeError):
        original_bundle.streams[StreamPurpose.EVENT] = original_bundle.streams[
            StreamPurpose.FISSION
        ]


def test_isolated_trajectory_identity_changes_every_stream() -> None:
    _, request, _, _ = _fixture()
    new_trajectory = "E01-S06-EXAMPLE-T9999"
    namespace = isolated_stream_namespace(
        experiment_id=request.experiment_id,
        specification_id=request.specification_id,
        trajectory_id=new_trajectory,
        replicate_index=request.replicate_index,
    )
    changed = SeedRequest(
        experiment_id=request.experiment_id,
        specification_id=request.specification_id,
        trajectory_id=new_trajectory,
        replicate_index=request.replicate_index,
        engine_id=request.engine_id,
        root_seed_hex=request.root_seed_hex,
        coupling_policy=CouplingPolicy.TRAJECTORY_ISOLATED,
        coupling_reason=None,
        stream_namespaces={purpose: namespace for purpose in CANONICAL_STREAM_PURPOSES},
    )
    first = derive_seed_bundle(request)
    second = derive_seed_bundle(changed)
    assert all(
        first.streams[purpose].seed_material_hex
        != second.streams[purpose].seed_material_hex
        for purpose in CANONICAL_STREAM_PURPOSES
    )


def test_auxiliary_stream_consumption_cannot_perturb_engine_streams() -> None:
    specification, request, _, _ = _fixture()
    bundle = derive_seed_bundle(request)

    def run(*, consume_auxiliary: bool) -> tuple[object, object, object]:
        generators = bundle.fresh_generators()
        core = tuple(
            generator_state_sha256(generators[purpose])
            for purpose in CANONICAL_STREAM_PURPOSES[:6]
        )
        if consume_auxiliary:
            for purpose in CANONICAL_STREAM_PURPOSES[6:]:
                generators[purpose].bit_generator.random_raw(500)
        assert core == tuple(
            generator_state_sha256(generators[purpose])
            for purpose in CANONICAL_STREAM_PURPOSES[:6]
        )
        streams = bundle.independent_engine_streams(generators)
        beta = generate_catalytic_matrix(specification, streams.catalytic_matrix)
        initial = initialize_state(specification, streams.initialization)
        lineage = simulate_lineage(
            initial,
            beta=beta,
            specification=specification,
            rng_streams=streams,
        )
        return beta, initial, lineage

    first_beta, first_initial, first_lineage = run(consume_auxiliary=False)
    second_beta, second_initial, second_lineage = run(consume_auxiliary=True)
    np.testing.assert_array_equal(first_beta, second_beta)
    assert first_initial == second_initial
    assert first_lineage == second_lineage


def test_canonical_serialization_and_binary64_are_lossless_and_fail_closed() -> None:
    values = [0.0, -0.0, np.nextafter(0.0, 1.0), 1.0 / 10.0, -123.5]
    for value in values:
        encoded = float_to_hex(value)
        decoded = float_from_hex(encoded)
        assert float_bits_hex(decoded) == float_bits_hex(value)
        assert binary64_ulp_distance(decoded, value) == 0

    with pytest.raises(SeedContractError, match="JSON float"):
        canonical_json_bytes({"forbidden": 0.1})
    with pytest.raises(SerializationError, match="Noncanonical"):
        float_from_hex("0x1.00p+0")

    envelope = make_envelope({"integer": 1, "binary64Hex": float_to_hex(0.1)})
    encoded = serialize_envelope(envelope)
    assert (
        serialize_envelope(deserialize_envelope(encoded, require_canonical=True))
        == encoded
    )
    duplicate = (
        b'{"payload":{},"payload":{},"payloadSha256":"'
        + b"0" * 64
        + b'","serializationVersion":"E01-canonical-json-v1.0.0"}'
    )
    with pytest.raises(SerializationError, match="Duplicate"):
        deserialize_envelope(duplicate, require_canonical=False)


def test_seed_and_trajectory_json_schema_conformance() -> None:
    payload, _, _ = _trajectory()
    seed_envelope = make_envelope(payload["seedIdentity"]["seedManifest"])
    trajectory_envelope = make_envelope(payload)
    seed_schema = json.loads(SEED_SCHEMA.read_text())
    trajectory_schema = json.loads(TRAJECTORY_SCHEMA.read_text())
    Draft202012Validator.check_schema(seed_schema)
    Draft202012Validator.check_schema(trajectory_schema)
    validate_json_schema(
        seed_envelope, seed_schema, validator_factory=Draft202012Validator
    )
    validate_json_schema(
        trajectory_envelope,
        trajectory_schema,
        validator_factory=Draft202012Validator,
    )

    invalid = copy.deepcopy(trajectory_envelope)
    del invalid["payload"]["generations"][0]["growth"]["events"][0]["preEventState"]
    with pytest.raises(SerializationError, match="preEventState"):
        validate_json_schema(
            invalid,
            trajectory_schema,
            validator_factory=Draft202012Validator,
        )


def test_specification_and_seed_payloads_round_trip_exactly() -> None:
    specification, request, _, _ = _fixture()
    specification_payload = specification_to_payload(specification)
    assert specification_from_payload(specification_payload) == specification
    bundle = derive_seed_bundle(request)
    recovered_request = seed_request_from_payload(bundle.to_payload())
    assert recovered_request == request
    assert derive_seed_bundle(recovered_request).to_payload() == bundle.to_payload()


def test_trajectory_covers_event_fission_daughter_stopping_and_sampling_boundary() -> (
    None
):
    payload, _, _ = _trajectory()
    validate_trajectory_invariants(payload)
    events = [
        event
        for generation in payload["generations"]
        for event in generation["growth"]["events"]
    ]
    fissions = [
        generation["fission"]
        for generation in payload["generations"]
        if generation["fission"] is not None
    ]
    assert events and fissions
    assert all(
        "preEventState" in event and "postEventState" in event for event in events
    )
    assert all("totalHex" in event["propensities"] for event in events)
    assert all("daughterChoice" in fission for fission in fissions)
    assert payload["terminal"]["stoppingReason"]
    assert payload["samplingBoundary"]["analysisSamplingInstant"] == (
        "UNRESOLVED::E01-A025"
    )
    assert payload["registryBoundary"]["authorInitialStateRngStream"] == (
        "UNRESOLVED::E01-A020"
    )


def test_trajectory_invariants_reject_identity_state_and_rng_tampering() -> None:
    payload, _, _ = _trajectory()

    invalid_event = copy.deepcopy(payload)
    event = invalid_event["generations"][0]["growth"]["events"][0]
    event["eventIdentity"]["selectedSpeciesIndexZeroBased"] = (
        event["eventIdentity"]["selectedSpeciesIndexZeroBased"] + 1
    ) % len(event["preEventState"])
    with pytest.raises(TrajectoryContractError, match="event identity"):
        validate_trajectory_invariants(invalid_event)

    invalid_state = copy.deepcopy(payload)
    invalid_state["generations"][0]["growth"]["events"][0]["postEventState"][0] += 1
    with pytest.raises(TrajectoryContractError, match="mass fields"):
        validate_trajectory_invariants(invalid_state)

    invalid_rng = copy.deepcopy(payload)
    invalid_rng["generations"][0]["growth"]["events"][0]["rngUses"]["event"][
        "stateSha256Before"
    ] = "0" * 64
    with pytest.raises(TrajectoryContractError, match="RNG state chain"):
        validate_trajectory_invariants(invalid_rng)


def test_checksum_tamper_detection_and_exact_same_engine_regeneration() -> None:
    payload, identity, registry = _trajectory()
    envelope = make_envelope(payload)
    original = serialize_envelope(envelope)
    regenerated = regenerate_independent_trajectory(
        reference_payload=payload,
        capture_identity=identity,
        registry_boundary=registry,
    )
    assert regenerated == payload
    assert serialize_envelope(make_envelope(regenerated)) == original

    tampered = copy.deepcopy(envelope)
    tampered["payload"]["terminal"]["completedFissions"] += 1
    with pytest.raises(SerializationError, match="checksum mismatch"):
        serialize_envelope(tampered)


def test_identity_change_blocks_same_engine_regeneration() -> None:
    payload, identity, registry = _trajectory()
    altered = type(identity)(
        **{
            **{
                field: getattr(identity, field)
                for field in identity.__dataclass_fields__
            },
            "numpy_version": "2.4.7",
        }
    )
    with pytest.raises(TrajectoryContractError, match="capture identity differs"):
        regenerate_independent_trajectory(
            reference_payload=payload,
            capture_identity=altered,
            registry_boundary=registry,
        )


def test_capture_identity_rejects_unfrozen_numeric_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENBLAS_NUM_THREADS")
    with pytest.raises(TrajectoryContractError, match="thread environment"):
        capture_identity_from_workspace(
            repository_root=REPOSITORY_ROOT,
            artifacts_root=ARTIFACTS_ROOT,
        )


def test_registry_and_historical_rng_boundaries_remain_unresolved() -> None:
    registry_path = ARTIFACTS_ROOT / (
        "E01_forensic_replication_bundle/specifications/"
        "specification_registry_v0.3.0.yaml"
    )
    assert hashlib.sha256(registry_path.read_bytes()).hexdigest() == (
        "aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891"
    )
    registry = yaml.safe_load(registry_path.read_text())
    by_name = {item["parameter"]: item for item in registry["parameters"]}
    assert by_name["gard.initial_state.rng_stream"]["value"] == ("UNRESOLVED::E01-A020")
    assert by_name["preprocessing.state_sampling_instant"]["value"] == (
        "UNRESOLVED::E01-A025"
    )
    contract = yaml.safe_load(SEED_CONTRACT.read_text())
    assert contract["engineBindings"]["e01_gard_historical"]["status"] == (
        "NONCONFORMING_TO_CANONICAL_SEEDED_REGENERATION"
    )
    assert registry["executionGate"]["executable"] is False


def test_generated_artifacts_and_fresh_process_regeneration_when_present(
    tmp_path: Path,
) -> None:
    shared = ARTIFACTS_ROOT / "E01_forensic_replication_bundle/reproducibility"
    step = ARTIFACTS_ROOT / "research_steps/S06"
    trajectory = shared / "examples/example_trajectory.json"
    if not trajectory.exists():
        pytest.skip("Canonical S06 artifacts are generated after focused code tests.")
    required = [
        shared / "seed_schema_v1.0.0.json",
        shared / "trajectory_schema_v1.0.0.json",
        shared / "seed_derivation_contract_v1.0.0.yaml",
        shared / "trajectory_precision_contract_v1.0.0.yaml",
        shared / "examples/example_seed_manifest.json",
        shared / "examples/example_event_record.json",
        shared / "examples/example_fission_record.json",
        step / "seed_validation.json",
        step / "schema_conformance.json",
        step / "serialization_validation.json",
        step / "regeneration_validation.json",
        step / "cross_platform_precision_validation.json",
        step / "registry_preservation.json",
        step / "branch_schema_coverage.json",
        step / "validation_summary.json",
        step / "artifact_manifest.json",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
    branch_coverage = json.loads((step / "branch_schema_coverage.json").read_text())
    assert branch_coverage["success"] is True
    assert branch_coverage["profileCount"] == 3
    assert all(
        profile["sameEngineCanonicalBytesExact"]
        for profile in branch_coverage["profiles"]
    )
    precision = json.loads(
        (step / "cross_platform_precision_validation.json").read_text()
    )
    assert precision["allObservedNumericThreadPoolsUseOneThread"] is True
    assert all(
        pool["numThreads"] == 1 for pool in precision["observedNumericThreadPools"]
    )
    output = tmp_path / "regenerated.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/e01/regenerate_s06_trajectory.py",
            "--input",
            str(trajectory),
            "--output",
            str(output),
            "--artifacts-dir",
            str(ARTIFACTS_ROOT),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    summary = json.loads(result.stdout)
    assert summary["canonicalBytesExact"] is True
    assert output.read_bytes() == trajectory.read_bytes()
    assert not (ARTIFACTS_ROOT / "research_steps/S07").exists()
