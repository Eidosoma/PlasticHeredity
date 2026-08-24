from __future__ import annotations

import inspect

import numpy as np

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.phir_extension_px4 import (
    ARMS,
    C02_VARIANTS,
    C03_VARIANTS,
    CODEX_VARIANT,
    CPU_SECONDS,
    FULL_PORT,
    HORIZON,
    MATRICES,
    REPLICATES,
    _batch_digest,
    _future_seed,
    _records_digest,
    advance_variant,
    protocol,
    scientific_spec,
    smoke_spec,
    validation_checks,
    variant_factors,
    variants,
)
from plastic_heredity.simulator import (
    advance_fission,
    generate_beta,
    generate_initial_composition,
)


def test_px4_design_is_fixed_and_factorially_complete() -> None:
    spec = scientific_spec()
    assert MATRICES == 24 and spec.matrices == 24
    assert REPLICATES == 2 and HORIZON == 60
    assert CPU_SECONDS == 14 * 3600
    assert ARMS == ("STABILIZE", "DESTABILIZE", "NOOP")
    assert variants("02") == C02_VARIANTS and len(C02_VARIANTS) == 2
    combinations = {
        (
            variant_factors("03", name)["adaptive_exposure"],
            variant_factors("03", name)["allow_overshoot"],
            variant_factors("03", name)["uniform_daughter"],
        )
        for name in C03_VARIANTS
    }
    assert combinations == {
        (exposure, overshoot, daughter)
        for exposure in (0, 1)
        for overshoot in (0, 1)
        for daughter in (0, 1)
    }
    assert FULL_PORT == {"02": "C02_EVENTWISE", "03": "C03_E1_O1_D1"}


def test_codex_variant_is_bit_exact_for_both_candidates() -> None:
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(10))
    initial = generate_initial_composition(config, np.random.default_rng(11))
    for candidate in CANDIDATES:
        left = np.random.default_rng(12)
        right = np.random.default_rng(12)
        direct = advance_fission(initial, beta, config, CANDIDATES[candidate], left)
        wrapped = advance_variant(
            initial, beta, config, candidate, CODEX_VARIANT[candidate], right
        )
        assert _records_digest((direct,)) == _records_digest((wrapped,))
        assert left.bit_generator.state == right.bit_generator.state


def test_neutral_future_stream_excludes_arm_and_variant() -> None:
    assert "arm" not in inspect.signature(_future_seed).parameters
    assert "variant" not in inspect.signature(_future_seed).parameters
    spec = smoke_spec()
    seed = _future_seed(spec, "03", 2, 0)
    assert len({seed for _arm in ARMS for _variant in C03_VARIANTS}) == 1


def test_protocol_keeps_public_and_full_readings_without_claim_inflation() -> None:
    frozen = protocol()
    assert frozen["measurements"]["primary"].startswith("public nine-atom")
    assert "material full-block" in frozen["measurements"]["secondary"]
    assert frozen["external_code_imported_or_executed"] is False
    assert frozen["no_48_matrix_campaign"]
    assert "strict-eight is excluded" in frozen["claim_boundary"]


def test_complete_px4_validation_suite() -> None:
    checks = validation_checks()
    assert len(checks) >= 16
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]
