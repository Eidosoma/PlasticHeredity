from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e01_replicator_labels import (
    ClusterConfiguration,
    LabelContractError,
    cluster_labels,
    continuous_past_recurrence,
    historical_technique1_labels,
    historical_technique2_diagnostic,
    metric_result,
    strict_distance_adjacency,
    strict_similarity_adjacency,
)


def _configuration(
    metric: str,
    *,
    threshold: float,
    minimum_cluster_size: int = 3,
    temporal_scope: str = "paper_retrospective_full_trace",
) -> ClusterConfiguration:
    family = {"cosine": "Y_C", "euclidean": "Y_E", "aitchison": "Y_A"}[metric]
    comparator = "strict_greater_than" if metric == "cosine" else "strict_less_than"
    return ClusterConfiguration(
        configuration_id=f"TEST-{family}-{temporal_scope}",
        family_id=family,
        family_name=f"{metric}_threshold_graph",
        evidence_class="VALIDATION_ONLY_RECONSTRUCTION_NOT_AUTHOR_DEFAULT",
        metric=metric,  # type: ignore[arg-type]
        representation=f"explicit_{metric}_representation",
        threshold=threshold,
        comparator=comparator,
        minimum_cluster_size=minimum_cluster_size,
        temporal_scope=temporal_scope,  # type: ignore[arg-type]
        zero_policy="explicit_ineligible",
    )


def test_strict_threshold_boundaries_are_not_inclusive() -> None:
    similarity = np.array([[1.0, 0.9], [0.9, 1.0]])
    distance = np.array([[0.0, 0.1], [0.1, 0.0]])
    eligible = np.array([True, True])
    assert not strict_similarity_adjacency(
        similarity, threshold=0.9, eligible=eligible
    ).any()
    assert not strict_distance_adjacency(
        distance, threshold=0.1, eligible=eligible
    ).any()


def test_historical_hc11_oracle_and_source_padding_are_exact() -> None:
    states = np.array([[1.0, 0.0], [1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.0, 0.0]])
    result = historical_technique1_labels(
        states,
        trajectory_id="HC11",
        observation_ids=[f"g{index:04d}" for index in range(1, 6)],
        configuration_id="E01-S08-YH-T1-HGT090-v1.0.0",
        threshold=0.9,
        evidence_class="PINNED_PUBLIC_HISTORICAL_SOURCE_BEHAVIOR",
    )
    assert result.result_status == "OK"
    assert [row.is_replicator for row in result.rows] == [
        True,
        False,
        False,
        False,
        False,
    ]
    np.testing.assert_allclose(
        [row.historical_local_score for row in result.rows], [1.0, 0.9, 0.7, 0.6, 0.0]
    )
    assert result.rows[-1].source_padding is True
    assert (
        result.rows[-1].label_status
        == "LABELED_DRIFT_SOURCE_PADDED_AFTER_FIRST_ZERO_SUM"
    )


def test_historical_source_domain_error_still_emits_every_row() -> None:
    result = historical_technique1_labels(
        [[0, 0], [1, 1], [1, 1]],
        trajectory_id="invalid-leading-zero",
        observation_ids=["g0001", "g0002", "g0003"],
        configuration_id="YH",
        threshold=0.9,
        evidence_class="PINNED_PUBLIC_HISTORICAL_SOURCE_BEHAVIOR",
    )
    assert result.result_status == "ERROR_SOURCE_DOMAIN"
    assert len(result.rows) == 3
    assert all(row.label_status == "ERROR_SOURCE_DOMAIN" for row in result.rows)
    assert all(row.is_replicator is None for row in result.rows)


def test_metric_eligibility_and_aitchison_never_replaces_zeros() -> None:
    states = [[1, 1, 1], [2, 0, 1], [0, 0, 0]]
    cosine = metric_result(states, metric="cosine")
    euclidean = metric_result(states, metric="euclidean")
    aitchison = metric_result(states, metric="aitchison")
    assert cosine.eligible == (True, True, False)
    assert euclidean.eligible == (True, True, False)
    assert aitchison.eligible == (True, False, False)
    assert aitchison.reasons[1] == "ZERO_COMPONENT_STRICT_POSITIVE_AITCHISON"
    assert aitchison.reasons[2] == "ZERO_SUM_COMPOSITION"
    assert np.isnan(aitchison.values[1:]).all()


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "aitchison"])
def test_metric_is_invariant_to_row_scaling_and_common_component_permutation(
    metric: str,
) -> None:
    states = np.array([[4.0, 3.0, 2.0, 1.0], [3.0, 4.0, 1.0, 2.0]])
    scaled = states * np.array([3.0, 7.0])[:, None]
    permuted = states[:, [2, 0, 3, 1]]
    baseline = metric_result(states, metric=metric)  # type: ignore[arg-type]
    np.testing.assert_allclose(
        baseline.values,
        metric_result(scaled, metric=metric).values,  # type: ignore[arg-type]
        atol=1e-14,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        baseline.values,
        metric_result(permuted, metric=metric).values,  # type: ignore[arg-type]
        atol=1e-14,
        rtol=1e-14,
    )


def test_retrospective_graph_labels_retain_ineligible_rows() -> None:
    states = [[9, 1], [18, 2], [27, 3], [0, 0]]
    result = cluster_labels(
        states,
        trajectory_id="trace",
        observation_ids=["g0001", "g0002", "g0003", "g0004"],
        configuration=_configuration("euclidean", threshold=0.01),
    )
    assert [row.is_replicator for row in result.rows] == [True, True, True, None]
    assert result.rows[-1].label_status == "INELIGIBLE"
    assert result.rows[-1].ineligibility_reason == "ZERO_SUM_COMPOSITION"
    assert len(result.rows) == len(states)


def test_past_only_branch_does_not_backfill_from_future_members() -> None:
    states = [[9, 1], [18, 2], [27, 3], [36, 4]]
    ids = [f"g{index:04d}" for index in range(1, 5)]
    retrospective = cluster_labels(
        states,
        trajectory_id="trace",
        observation_ids=ids,
        configuration=_configuration("euclidean", threshold=0.01),
    )
    online = cluster_labels(
        states,
        trajectory_id="trace",
        observation_ids=ids,
        configuration=_configuration(
            "euclidean", threshold=0.01, temporal_scope="past_only_online"
        ),
    )
    assert [row.is_replicator for row in retrospective.rows] == [True] * 4
    assert [row.is_replicator for row in online.rows] == [False, False, True, True]


def test_single_linkage_transitive_component_is_explicit() -> None:
    states = [[100, 0], [95, 5], [90, 10]]
    result = cluster_labels(
        states,
        trajectory_id="chain",
        observation_ids=["g0001", "g0002", "g0003"],
        configuration=_configuration("euclidean", threshold=0.08),
    )
    assert [row.is_replicator for row in result.rows] == [True, True, True]
    assert len({row.component_id for row in result.rows}) == 1


def test_retrospective_order_permutation_preserves_memberships_by_identity() -> None:
    states = np.array([[90, 10], [89, 11], [10, 90], [11, 89], [12, 88]])
    ids = np.array([f"g{index:04d}" for index in range(1, 6)])
    configuration = _configuration("euclidean", threshold=0.05, minimum_cluster_size=2)
    baseline = cluster_labels(
        states,
        trajectory_id="order",
        observation_ids=ids.tolist(),
        configuration=configuration,
    )
    order = np.array([3, 0, 4, 1, 2])
    permuted = cluster_labels(
        states[order],
        trajectory_id="order",
        observation_ids=ids[order].tolist(),
        configuration=configuration,
    )
    first = {
        row.observation_id: (row.is_replicator, row.component_id)
        for row in baseline.rows
    }
    second = {
        row.observation_id: (row.is_replicator, row.component_id)
        for row in permuted.rows
    }
    assert second == first


def test_historical_technique2_source_edge_is_not_repaired() -> None:
    diagnostic = historical_technique2_diagnostic(
        [[1, 0], [1, 0], [1, 0]],
        trajectory_id="source-edge",
        configuration_id="YH-T2",
        threshold=0.9,
        drift_size=2,
    )
    assert diagnostic["status"] == "ERROR_SOURCE_DOMAIN"
    assert diagnostic["errorType"] == "HistoricalSourceDomainError"
    assert diagnostic["sourceRepairApplied"] is False


def test_continuous_recurrence_is_past_only_and_retains_zero_status() -> None:
    recurrence = continuous_past_recurrence([[1, 0], [1, 0], [0, 1], [0, 0]])
    assert recurrence[0] is None
    assert recurrence[1] == pytest.approx(1.0)
    assert recurrence[2] == pytest.approx(0.0)
    assert recurrence[3] is None


def test_configuration_rejects_implicit_or_incompatible_choices() -> None:
    with pytest.raises(LabelContractError, match="comparator"):
        ClusterConfiguration(
            configuration_id="bad",
            family_id="Y_C",
            family_name="cosine",
            evidence_class="validation",
            metric="cosine",
            representation="raw",
            threshold=0.9,
            comparator="strict_less_than",
            minimum_cluster_size=3,
            temporal_scope="paper_retrospective_full_trace",
            zero_policy="explicit",
        )
    with pytest.raises(LabelContractError, match="nonnegative"):
        metric_result([[1, -1]], metric="euclidean")


def test_s08_builder_writes_complete_replayable_artifacts(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/e01/build_s08_label_artifacts.py"),
        "--artifacts-dir",
        str(tmp_path),
    ]
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
    }
    first = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **environment},
    )
    assert '"success": true' in first.stdout.lower()
    step_dir = tmp_path / "research_steps/S08"
    shared_dir = tmp_path / "E01_forensic_replication_bundle/labels"
    validation = json.loads((step_dir / "validation_summary.json").read_text())
    assert validation["researchStepId"] == "S08"
    assert validation["success"] is True
    assert (
        validation["passedValidationCheckCount"] == validation["validationCheckCount"]
    )
    assert validation["canonicalLabelRowCount"] == 301
    assert validation["referenceLabelRowCount"] == 172
    assert validation["thresholdSensitivityRowCount"] == 170
    required = [
        step_dir / "label_outputs.csv",
        step_dir / "label_arrays.json",
        step_dir / "label_overlap_long.csv",
        step_dir / "binary_ari_matrix.csv",
        step_dir / "cluster_ari_matrix.csv",
        step_dir / "run_level_disagreement.csv",
        step_dir / "threshold_sensitivity.csv",
        step_dir / "label_disagreement_map.png",
        step_dir / "artifact_manifest.json",
        shared_dir / "label_family_contract_v1.0.1.yaml",
        shared_dir / "clustering_configurations_v1.0.1.yaml",
        shared_dir / "label_arrays_schema_v1.0.0.json",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
    assert not (tmp_path / "research_steps/S09").exists()

    label_hash = hashlib.sha256(
        (step_dir / "label_arrays.json").read_bytes()
    ).hexdigest()
    record_before = (step_dir / "preregistration_record.json").read_bytes()
    subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **environment},
    )
    assert (
        hashlib.sha256((step_dir / "label_arrays.json").read_bytes()).hexdigest()
        == label_hash
    )
    assert (step_dir / "preregistration_record.json").read_bytes() == record_before
    assert not (tmp_path / "research_steps/S09").exists()
