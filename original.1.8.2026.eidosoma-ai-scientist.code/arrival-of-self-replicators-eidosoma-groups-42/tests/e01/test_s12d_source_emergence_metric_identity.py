from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from e01_pigozzi_source_audit.core import SourceImplementation
from e01_source_emergence_metric_identity.core import (
    ATOM_KEY_STRINGS,
    CONFIRMATION_DATASET_ROLE,
    EXPLORATORY_DATASET_ROLE,
    GARD_SPECIFICATION_ID,
    NEW_FIXTURE_IDS,
    ROOT_SEED_HEX,
    SOURCE_RELATIONSHIP,
    all_metric_identity_fixtures,
    confirmation_seed_bundle,
    result_replay_equal,
    run_emergence_pipeline,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/e01/s12d_source_emergence_metric_identity_preregistration.yaml"
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")


def test_metric_formula_is_exact_and_replayable() -> None:
    observations = next(
        array
        for suite, fixture_id, array in all_metric_identity_fixtures()
        if suite == "S12D_UNTOUCHED_CONFIRMATION"
        and fixture_id == "S12D_ORDINARY_BLOCK_GAUSSIAN_A"
    )
    for implementation in SourceImplementation:
        first = run_emergence_pipeline(
            observations,
            implementation,
            SAFE_LATTICE,
            preprocessing_seed=101,
            partition_seed=202,
        )
        second = run_emergence_pipeline(
            observations,
            implementation,
            SAFE_LATTICE,
            preprocessing_seed=101,
            partition_seed=202,
        )
        assert first.status in {"ELIGIBLE", "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"}
        assert first.synergy is not None
        assert first.downward_causation is not None
        assert first.emergence is not None
        assert np.array_equal(
            first.emergence,
            first.synergy + first.downward_causation,
            equal_nan=True,
        )
        assert result_replay_equal(first, second)


def test_fixture_suite_cardinality_and_firewall_labels() -> None:
    fixtures = all_metric_identity_fixtures()
    assert len(fixtures) == 20
    assert len(NEW_FIXTURE_IDS) == 6
    assert sum(suite == "S12C_DEVELOPMENT" for suite, _, _ in fixtures) == 7
    assert sum(suite == "S12C_CONFIRMATION" for suite, _, _ in fixtures) == 7
    assert sum(suite == "S12D_UNTOUCHED_CONFIRMATION" for suite, _, _ in fixtures) == 6
    assert EXPLORATORY_DATASET_ROLE == "EXPLORATORY_EXISTING_TRAJECTORIES"
    assert CONFIRMATION_DATASET_ROLE == "UNTOUCHED_CONFIRMATION_TRAJECTORIES"


def test_confirmation_seed_domain_has_216_unique_streams() -> None:
    stream_ids: set[str] = set()
    materials: set[str] = set()
    for matrix_index in range(24):
        payload = confirmation_seed_bundle(matrix_index).to_payload()
        assert payload["rootSeedHex"] == ROOT_SEED_HEX
        assert payload["specificationId"] == GARD_SPECIFICATION_ID
        assert payload["trajectoryId"] == f"E01-S12D-C{matrix_index:02d}"
        assert len(payload["streams"]) == 9
        stream_ids.update(item["streamId"] for item in payload["streams"].values())
        materials.update(
            item["seedMaterialHex"] for item in payload["streams"].values()
        )
    assert len(stream_ids) == 216
    assert len(materials) == 216
    assert ROOT_SEED_HEX != "12" * 32


def test_atom_serialization_and_source_relationship_are_frozen() -> None:
    assert ATOM_KEY_STRINGS == (
        "[[[0,1]],[[0,1]]]",
        "[[[0,1]],[[0]]]",
        "[[[0,1]],[[1]]]",
    )
    assert all(json.loads(item) for item in ATOM_KEY_STRINGS)
    assert SOURCE_RELATIONSHIP == "SOURCE_INFORMED_METRIC_IDENTITY"


def test_preregistration_preserves_scope_and_global_gates() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["researchStepId"] == "S12D"
    assert (
        config["preregistrationVersion"]
        == "E01-S12D-SOURCE-EMERGENCE-METRIC-IDENTITY-CONFIRMATION-v1.0.0"
    )
    assert config["scopeBoundary"]["exactUntouchedConfirmationMatrixCount"] == 24
    assert config["scopeBoundary"]["newGardTrajectoriesExactly"] == 24
    assert (
        config["scopeBoundary"]["s13StatusThroughout"]
        == "BLOCKED_PENDING_S12D_HUMAN_REVIEW"
    )
    assert config["metricEquivalenceGate"]["expectedRows"] == 40
    assert config["metricEquivalenceGate"]["allRowsMustPass"] is True
    assert config["metricEquivalenceGate"]["maximumAbsoluteDifferenceAtMost"] == 1e-12
    assert config["statistics"]["bootstrapReplicates"] == 4096
    assert config["statistics"]["circularShiftReplicates"] == 4096
    assert config["runtimeAndStorage"]["sourceAnalysisWorkers"] == 6
    assert config["runtimeAndStorage"]["scopeReductionForbidden"] is True
    assert config["scopeBoundary"]["automaticS13Forbidden"] is True
    assert (
        config["metricIdentity"]["hierarchy"]["primaryConfirmatory"]
        == "IIGR_EMERGENCE_FULL"
    )
    assert config["metricIdentity"]["phirlMayNotReplaceIigrAsPrimary"] is True


def test_pinned_source_separates_integrated_and_emergence_assignments() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for source_id in ("IIGR_CORRECTED_SOURCE", "PHIRL_REGULARIZED_SOURCE"):
        path = Path(config["sourceSnapshots"][source_id]["localCheckout"]) / "main.py"
        text = path.read_text(encoding="utf-8")
        assert 'info["integrated"] = local_phi_r(phi_results)' in text
        assert 'info["emergence"] = info["synergy"] + info["causation"]' in text
