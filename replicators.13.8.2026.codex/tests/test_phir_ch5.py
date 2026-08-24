from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.phir_ch5 import (
    CampaignBatch,
    CONFIRMATION_MATRICES,
    PILOT_MATRICES,
    SEED_DOMAINS,
    _batch_digest,
    _paired_summary,
    _public_parity_fixture,
    protocol,
    scientific_spec,
    validation_checks,
)
from plastic_heredity.phir_instruments import (
    ALL_ATOMS,
    PHIR_ATOMS,
    advance_fission_traced,
    fiedler_bipartition,
    lagged_gaussian_mi_graph,
    records_equal,
    revised_phi_from_partition,
    rng_states_equal,
    typeset_whole_minus_parts,
)
from plastic_heredity.simulator import (
    advance_fission,
    generate_beta,
    generate_initial_composition,
)


def test_public_phirl_synthetic_fixture_matches_pinned_value() -> None:
    fixture = _public_parity_fixture()
    first, second = fiedler_bipartition(lagged_gaussian_mi_graph(fixture))
    revised, causation, emergence, synergy, atoms = revised_phi_from_partition(
        fixture, first, second
    )
    assert set(first) == {4, 5}
    assert set(second) == {0, 1, 2, 3}
    assert abs(revised - 0.6944357302999425) < 1e-12
    assert atoms.shape == (16,)
    assert emergence == causation + synergy


def test_typeset_and_text_ratio_are_distinct_registered_readings() -> None:
    fixture = _public_parity_fixture()
    first, second = fiedler_bipartition(lagged_gaussian_mi_graph(fixture))
    numerator, whole = typeset_whole_minus_parts(fixture, first, second)
    assert np.isfinite(numerator)
    assert np.isfinite(whole)
    assert not np.isclose(numerator, numerator / whole)


def test_all_atoms_and_revised_subset_are_explicit() -> None:
    assert len(ALL_ATOMS) == 16
    assert len(PHIR_ATOMS) == 9
    assert set(PHIR_ATOMS).issubset(set(ALL_ATOMS))


def test_traced_simulator_is_bitwise_plain_for_both_candidates() -> None:
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(801))
    initial = generate_initial_composition(config, np.random.default_rng(802))
    for index, candidate in enumerate(CANDIDATES):
        left_rng = np.random.default_rng(900 + index)
        right_rng = np.random.default_rng(900 + index)
        traced = advance_fission_traced(
            initial, beta, config, CANDIDATES[candidate], left_rng
        )
        plain = advance_fission(
            initial, beta, config, CANDIDATES[candidate], right_rng
        )
        assert records_equal(traced.record, plain)
        assert rng_states_equal(left_rng.bit_generator.state, right_rng.bit_generator.state)


def test_pilot_and_confirmation_are_disjoint_and_fixed() -> None:
    pilot = scientific_spec("pilot")
    confirmation = scientific_spec("confirmation")
    assert pilot.matrices == PILOT_MATRICES == 24
    assert confirmation.matrices == CONFIRMATION_MATRICES == 48
    assert pilot.replicates == confirmation.replicates == 2
    assert SEED_DOMAINS["pilot_matrix"] != SEED_DOMAINS["confirmation_matrix"]


def test_confirmation_has_a_manual_hard_barrier() -> None:
    barrier = protocol()["manual_confirmation_barrier"]
    assert barrier["pilot_result_required"]
    assert barrier["user_authorization_artifact_required"]
    assert barrier["automatic_launch_forbidden"]
    assert barrier["pilot_confirmation_pooling_forbidden"]


def test_storage_contract_forbids_raw_molecular_traces() -> None:
    assert "raw molecular traces forbidden" in protocol()["storage"]


def test_batch_digest_ignores_only_its_digest_field() -> None:
    batch = CampaignBatch(
        matrix_id=0,
        beta=np.eye(2),
        initial_composition=np.asarray([1, 0], dtype=np.int16),
        natural_rows=(),
        branch_rows=(),
        bridge_rows=(),
        dose_rows=(),
        probe_rows=(),
        selected_edit_rows=(),
        probe_screen_rows=(),
        no_op_plain_exact=True,
        scientific_digest="placeholder",
    )
    digest = _batch_digest(batch)
    assert _batch_digest(replace(batch, scientific_digest=digest)) == digest
    assert _batch_digest(replace(batch, no_op_plain_exact=False)) != digest


def test_batch_digest_is_stable_across_worker_pickle_boundary() -> None:
    rows = tuple(
        {
            "phase": "fixture",
            "matrix_id": 0,
            "candidate": "02",
            "composition": [1, 2],
            "undefined_reading": float("nan"),
            "record_digest": "a" * 64,
        }
        for _ in range(10)
    )
    batch = CampaignBatch(
        matrix_id=0,
        beta=np.eye(2),
        initial_composition=np.asarray([1, 0], dtype=np.int16),
        natural_rows=rows,
        branch_rows=rows,
        bridge_rows=rows,
        dose_rows=rows,
        probe_rows=rows,
        selected_edit_rows=rows,
        probe_screen_rows=rows,
        no_op_plain_exact=True,
        scientific_digest="",
    )
    digest = _batch_digest(batch)
    transported = pickle.loads(pickle.dumps(batch, protocol=5))
    assert _batch_digest(transported) == digest


def test_matrix_bootstrap_and_randomization_are_reproducible() -> None:
    arrays_a: dict[str, np.ndarray] = {}
    arrays_b: dict[str, np.ndarray] = {}
    values = np.asarray([0.2, 0.1, -0.05, 0.3])
    first = _paired_summary(values, 64, "unit/determinism", arrays_a)
    second = _paired_summary(values, 64, "unit/determinism", arrays_b)
    assert first == second
    assert arrays_a.keys() == arrays_b.keys()
    assert all(np.array_equal(arrays_a[key], arrays_b[key]) for key in arrays_a)


def test_validation_suite_has_exactly_34_passing_checks() -> None:
    checks = validation_checks()
    assert len(checks) == 34
    assert all(checks.values())
