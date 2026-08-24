from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e01_compositional_preprocessing import (
    CompositionalContractError,
    CoordinateSpecification,
    ZeroTreatment,
    apply_zero_treatment,
    covariance_diagnostics,
    helmert_simplex_basis,
    inverse_coordinates,
    pairwise_euclidean,
    principal_logratio_basis,
    transform_coordinates,
    validate_simplex_basis,
)
from e01_gard_reproducibility.serialization import (
    deserialize_envelope,
    serialize_envelope,
)


def zero(method: str, delta: float | None) -> ZeroTreatment:
    return ZeroTreatment(
        specification_id=f"TEST-ZERO-{method}-{delta}",
        method=method,  # type: ignore[arg-type]
        delta=delta,
        evidence_class="VALIDATION_ONLY_RECONSTRUCTION_NOT_AUTHOR_DEFAULT",
    )


def coordinate(
    family: str,
    dimension: int,
    *,
    dropped: int | None = None,
    fit_scope: str | None = None,
) -> CoordinateSpecification:
    return CoordinateSpecification(
        specification_id=f"TEST-COORD-{family}-{dimension}-{dropped}-{fit_scope}",
        family=family,  # type: ignore[arg-type]
        dimension=dimension,
        evidence_class="VALIDATION_ONLY_RECONSTRUCTION_NOT_AUTHOR_DEFAULT",
        dropped_component_zero_based=dropped,
        basis_fit_scope_id=fit_scope,
    )


def test_additive_pseudocount_formula_and_zero_sum_are_exact() -> None:
    result = apply_zero_treatment([2, 0, 1], zero("additive_pseudocount", 0.5))
    np.testing.assert_allclose(result.composition, [2.5 / 4.5, 0.5 / 4.5, 1.5 / 4.5])
    assert result.status == "ELIGIBLE"
    assert result.replacement_mass_per_zero == pytest.approx(1 / 9)

    empty = apply_zero_treatment([0, 0, 0], zero("additive_pseudocount", 1e-6))
    np.testing.assert_allclose(empty.composition, [1 / 3, 1 / 3, 1 / 3])
    assert empty.status == "ELIGIBLE"


def test_multiplicative_replacement_preserves_positive_ratios_and_statuses_zero_sum() -> (
    None
):
    result = apply_zero_treatment([2, 0, 1], zero("multiplicative_replacement", 0.5))
    np.testing.assert_allclose(result.composition, [16 / 27, 1 / 9, 8 / 27])
    assert result.composition[0] / result.composition[2] == pytest.approx(2.0)

    positive = apply_zero_treatment([2, 3, 5], zero("multiplicative_replacement", 1.0))
    np.testing.assert_allclose(positive.composition, [0.2, 0.3, 0.5])

    empty = apply_zero_treatment([0, 0, 0], zero("multiplicative_replacement", 0.5))
    assert empty.status == "INELIGIBLE"
    assert empty.composition is None
    assert empty.reason == "ZERO_SUM_NO_COMPOSITION_FOR_MULTIPLICATIVE_REPLACEMENT"


def test_every_dropped_clr_and_helmert_ilr_invert_and_ilr_is_isometric() -> None:
    composition = np.array([0.11, 0.22, 0.31, 0.36])
    full_spec = coordinate("full_clr", 4)
    full = transform_coordinates(composition, full_spec, simplex_basis=None)
    full_inverse = inverse_coordinates(full, full_spec, simplex_basis=None)
    np.testing.assert_allclose(full_inverse, composition, atol=1e-14, rtol=1e-14)
    assert np.sum(full) == pytest.approx(0.0, abs=1e-14)

    for dropped in range(4):
        specification = coordinate("dropped_clr", 4, dropped=dropped)
        values = transform_coordinates(composition, specification, simplex_basis=None)
        reconstructed = inverse_coordinates(values, specification, simplex_basis=None)
        np.testing.assert_allclose(reconstructed, composition, atol=1e-14, rtol=1e-14)

    basis = helmert_simplex_basis(4)
    ilr_spec = coordinate("ilr_helmert", 4)
    compositions = np.array(
        [[0.11, 0.22, 0.31, 0.36], [0.20, 0.30, 0.10, 0.40], [0.4, 0.1, 0.2, 0.3]]
    )
    full_matrix = np.vstack(
        [
            transform_coordinates(row, full_spec, simplex_basis=None)
            for row in compositions
        ]
    )
    ilr_matrix = np.vstack(
        [
            transform_coordinates(row, ilr_spec, simplex_basis=basis)
            for row in compositions
        ]
    )
    np.testing.assert_allclose(
        pairwise_euclidean(full_matrix),
        pairwise_euclidean(ilr_matrix),
        atol=1e-14,
        rtol=1e-14,
    )


def test_raw_hellinger_and_principal_log_ratio_inverse() -> None:
    compositions = np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3], [0.4, 0.3, 0.2, 0.1]]
    )
    plr_basis, eigenvalues = principal_logratio_basis(compositions)
    assert eigenvalues.shape == (3,)
    validate_simplex_basis(plr_basis, dimension=4)
    for family, basis in [
        ("raw_proportions", None),
        ("hellinger", None),
        ("principal_log_ratio", plr_basis),
    ]:
        specification = coordinate(
            family,
            4,
            fit_scope="TEST-FIT" if family == "principal_log_ratio" else None,
        )
        for composition in compositions:
            values = transform_coordinates(
                composition, specification, simplex_basis=basis
            )
            reconstructed = inverse_coordinates(
                values, specification, simplex_basis=basis
            )
            np.testing.assert_allclose(
                reconstructed, composition, atol=1e-14, rtol=1e-14
            )


def test_zero_domain_and_invalid_inputs_fail_closed_without_infinities() -> None:
    no_replacement = apply_zero_treatment([1, 0, 2], zero("none", None))
    with pytest.raises(
        CompositionalContractError,
        match="ZERO_COMPONENT_LOG_RATIO_WITHOUT_REPLACEMENT",
    ):
        transform_coordinates(
            no_replacement.composition,
            coordinate("full_clr", 3),
            simplex_basis=None,
        )
    with pytest.raises(CompositionalContractError, match="nonnegative"):
        apply_zero_treatment([1, -1], zero("none", None))
    with pytest.raises(CompositionalContractError, match="finite"):
        apply_zero_treatment([1, np.inf], zero("none", None))
    with pytest.raises(CompositionalContractError, match="sentinel"):
        ZeroTreatment(
            "UNRESOLVED::E01-A027",
            "none",
            None,
            "VALIDATION",
        )
    with pytest.raises(CompositionalContractError, match="orthonormal"):
        validate_simplex_basis([[1.0], [1.0]], dimension=2)


def test_covariance_diagnostics_preserve_structural_singularity() -> None:
    compositions = np.array(
        [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.3, 0.1, 0.6]]
    )
    full = np.log(compositions) - np.mean(np.log(compositions), axis=1, keepdims=True)
    diagnostic = covariance_diagnostics(full, condition_threshold=1e12)
    assert diagnostic["rank"] <= 2
    assert diagnostic["status"] == "SAMPLE_OR_STRUCTURAL_RANK_DEFICIENT"
    assert np.isinf(diagnostic["conditionNumberRaw"])


def test_schema_is_valid_draft_202012() -> None:
    schema = json.loads(
        (REPOSITORY_ROOT / "configs/e01/s09_transform_output_schema.json").read_text()
    )
    Draft202012Validator.check_schema(schema)


def test_builder_writes_complete_lossless_status_bearing_artifacts(
    tmp_path: Path,
) -> None:
    registry_source = Path(
        "/artifacts/E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml"
    )
    fixture_source = Path("/artifacts/research_steps/S08/fixture_catalog.json")
    registry_target = (
        tmp_path
        / "E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml"
    )
    fixture_target = tmp_path / "research_steps/S08/fixture_catalog.json"
    registry_target.parent.mkdir(parents=True)
    fixture_target.parent.mkdir(parents=True)
    shutil.copyfile(registry_source, registry_target)
    shutil.copyfile(fixture_source, fixture_target)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/e01/build_s09_compositional_artifacts.py"),
            "--artifacts-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
    )
    step = tmp_path / "research_steps/S09"
    envelope_bytes = (step / "transform_arrays.json").read_bytes()
    envelope = deserialize_envelope(envelope_bytes, require_canonical=True)
    assert serialize_envelope(envelope) == envelope_bytes
    records = envelope["payload"]["records"]
    assert len(records) == 4901
    assert all(
        record["reason"] for record in records if record["status"] == "INELIGIBLE"
    )
    assert all(
        record["coordinatesHex"] is not None
        for record in records
        if record["status"] == "ELIGIBLE"
    )
    with (step / "zero_frequency_by_observation.csv").open(newline="") as handle:
        zero_rows = list(csv.DictReader(handle))
    assert len(zero_rows) == 43
    assert sum(row["anyZero"] == "True" for row in zero_rows) == 9
    assert sum(row["zeroSum"] == "True" for row in zero_rows) == 2
    validation = json.loads((step / "validation_summary.json").read_text())
    assert validation["researchStepId"] == "S09"
    assert validation["success"] is True
    assert validation["anchorMetrics"]["failureInjectionPassCount"] == 8
    assert not (tmp_path / "research_steps/S10").exists()
