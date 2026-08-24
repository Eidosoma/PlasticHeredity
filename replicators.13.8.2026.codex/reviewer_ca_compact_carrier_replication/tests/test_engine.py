from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from reviewer_ca_compact_carrier_replication.codec import load_codecs
from reviewer_ca_compact_carrier_replication.contract import (
    DEFAULT_ARTIFACTS,
    sha256_json,
)
from reviewer_ca_compact_carrier_replication.engine import (
    apply_boundary_operations,
    moderate_damage_masks,
    paired_random_fields,
    simulate_pair_cell,
    worker_cell,
)
from reviewer_ca_lineage_renewal_replication_v2.contract import NAMESPACE as V2_NAMESPACE
from reviewer_ca_lineage_renewal_replication_v2.engine import simulate_pair_lineages


def _fixture() -> tuple[dict, dict, str, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    acquisition = __import__("json").loads((DEFAULT_ARTIFACTS / "ACQUISITION.json").read_text())
    cohorts = __import__("json").loads((DEFAULT_ARTIFACTS / "COHORTS.json").read_text())
    launches = __import__("json").loads((DEFAULT_ARTIFACTS / "input/local/LAUNCH_RESETS.json").read_text())
    reference_doc = __import__("json").loads((DEFAULT_ARTIFACTS / "input/local/REFERENCE.json").read_text())
    hypothesis = __import__("json").loads((DEFAULT_ARTIFACTS / "input/local/HYPOTHESIS.json").read_text())
    pair = cohorts["cohorts"]["engineering"][0]
    donor_index = {donor["donor_id"]: donor for donor in acquisition["donors"]}
    donors = {pair["a_donor_id"]: donor_index[pair["a_donor_id"]], pair["b_donor_id"]: donor_index[pair["b_donor_id"]]}
    reset = launches[f"launch{pair['launch_index']}"]
    reference = np.asarray(reference_doc["motif_probability"], dtype=np.float64)
    primary = {label: np.asarray(hypothesis["targets"]["primary"][label], dtype=np.float64) for label in ("A", "B")}
    terminal = {label: np.asarray(hypothesis["targets"]["primary_terminal"][label], dtype=np.float64) for label in ("A", "B")}
    return pair, donors, reset, reference, primary, terminal


def test_random_fields_are_deterministic_and_process_uniforms_pair_environments() -> None:
    first = paired_random_fields("pair", 3, 2)
    second = paired_random_fields("pair", 3, 2)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    ordinary = first[1] < 0.002
    moderate = first[1] < 0.004
    assert np.all(~ordinary | moderate)


def test_moderate_masks_are_candidate_keyed_and_history_condition_paired() -> None:
    first = moderate_damage_masks("pair", "walsh-r016-q04", 1, 64, 16)
    second = moderate_damage_masks("pair", "walsh-r016-q04", 1, 64, 16)
    other = moderate_damage_masks("pair", "pca-r008-q04", 1, 64, 8)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0].shape == (64, 16)
    assert other[0].shape == (64, 8)


def test_boundary_zero_and_shuffle_operations_are_explicit() -> None:
    codec = load_codecs(DEFAULT_ARTIFACTS / "input")["walsh-r016-q04"]
    payload = np.ones((4, 2, 3, codec.rank), dtype=np.int8)
    index = {
        "intact": 0,
        "zero_every_boundary": 1,
        "latent_shuffle_every_boundary": 2,
        "decoded_shuffle_every_boundary": 3,
    }
    entry, decoded = apply_boundary_operations(
        payload,
        intact_payload=payload[0].copy(),
        codec=codec,
        pair_id="boundary-fixture",
        generation=1,
        environment="ordinary",
        condition_index=index,
    )
    assert not np.any(entry[1])
    assert not np.any(decoded[1])
    assert decoded.shape == (4, 2, 3, 512)


def test_identity_lifecycle_is_bitwise_equal_to_local_v2_fixture() -> None:
    pair, donors, reset, reference, primary, terminal = _fixture()
    conditions = [
        "intact",
        "zero_every_boundary",
        "read_disabled",
        "founder_write_disabled",
        "no_rewrite",
        "ablate_after_g2",
        "rescue_same_enter_g4",
        "rescue_opposite_enter_g4",
        "opposite_founder",
    ]
    codec = load_codecs(DEFAULT_ARTIFACTS / "input")["identity-r512-f32"]
    observed = simulate_pair_cell(
        pair=pair,
        donors=donors,
        reset_state_hex=reset,
        reference_probability=reference,
        targets_primary=primary,
        targets_terminal=terminal,
        codec=codec,
        environment="ordinary",
        replicates=2,
        generations=4,
        conditions=conditions,
        rng_namespace=V2_NAMESPACE,
    )
    donor_values = [donors[pair["a_donor_id"]], donors[pair["b_donor_id"]]]
    expected = simulate_pair_lineages(
        pair_id=pair["pair_id"],
        donor_state_hex=[donor["donor_state_hex"] for donor in donor_values],
        donor_initial_state_hex=[donor["initial_state_hex"] for donor in donor_values],
        reset_state_hex=reset,
        reference_probability=reference,
        targets_primary=primary,
        targets_terminal=terminal,
        replicates=2,
        generations=4,
        conditions=conditions,
    )
    for condition in conditions:
        assert observed["conditions"][condition]["outcomes"] == expected["conditions"][condition]["outcomes"]
        for generation, expected_history in expected["conditions"][condition]["carrier_history"].items():
            observed_history = observed["conditions"][condition]["carrier_history"][generation]
            for key in ("entry", "exit", "surviving_futures"):
                assert observed_history[key] == expected_history[key]


def test_worker_payload_is_reproducible_in_a_process() -> None:
    pair, donors, reset, reference, primary, terminal = _fixture()
    argument = {
        "artifacts": str(DEFAULT_ARTIFACTS.resolve()),
        "pair": pair,
        "donors": donors,
        "reset_state_hex": reset,
        "reference_probability": reference.tolist(),
        "targets_primary": {key: value.tolist() for key, value in primary.items()},
        "targets_terminal": {key: value.tolist() for key, value in terminal.items()},
        "candidate_id": "walsh-r016-q04",
        "environment": "moderate_joint",
        "replicates": 2,
        "generations": 2,
        "conditions": ["intact", "zero_every_boundary", "read_disabled"],
    }
    local = worker_cell(argument)
    with ProcessPoolExecutor(max_workers=1) as executor:
        remote = executor.submit(worker_cell, argument).result()
    assert sha256_json(local) == sha256_json(remote)
