#!/usr/bin/env python3
"""Build the frozen E01 S08 label-reconstruction evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from jsonschema import Draft202012Validator
from sklearn.metrics import adjusted_rand_score

from e01_gard_reproducibility import (
    CANONICAL_STREAM_PURPOSES,
    CouplingPolicy,
    SeedRequest,
    StreamPurpose,
    derive_seed_bundle,
    deserialize_envelope,
    isolated_stream_namespace,
    make_envelope,
    serialize_envelope,
)
from e01_replicator_labels import (
    ClusterConfiguration,
    LabelContractError,
    LabelTraceResult,
    cluster_labels,
    continuous_past_recurrence,
    historical_technique1_labels,
    historical_technique2_diagnostic,
    metric_result,
    strict_distance_adjacency,
    strict_similarity_adjacency,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/e01/s08_label_reconstruction_preregistration.yaml"
)
STEP_RELATIVE = Path("research_steps/S08")
SHARED_RELATIVE = Path("E01_forensic_replication_bundle/labels")
REFERENCE_FAMILIES = ("Y_H", "Y_C", "Y_E", "Y_A")
CLUSTER_FAMILIES = ("Y_C", "Y_E", "Y_A")
RESULT_ONLY_NAMES = {
    "label_outputs.csv",
    "label_arrays.json",
    "continuous_recurrence.csv",
    "label_overlap_long.csv",
    "binary_jaccard_matrix.csv",
    "binary_ari_matrix.csv",
    "cluster_ari_matrix.csv",
    "run_level_disagreement.csv",
    "disagreement_diagnostics.csv",
    "temporal_scope_diagnostics.csv",
    "threshold_sensitivity.csv",
    "historical_technique2_diagnostics.json",
    "edge_case_validation.json",
    "label_disagreement_map.png",
    "threshold_sensitivity.png",
    "binary_label_overlap.png",
    "registry_preservation.json",
    "validation_summary.json",
    "artifact_manifest.json",
    "research_step_full_results.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("S08 preregistration must be a YAML object.")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _check(condition: bool, check_id: str, details: Any) -> dict[str, Any]:
    return {"checkId": check_id, "success": bool(condition), "details": details}


def freeze_or_verify_preregistration(
    artifact_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    step_dir = artifact_root / STEP_RELATIVE
    step_dir.mkdir(parents=True, exist_ok=True)
    artifact_config = step_dir / "preregistration.yaml"
    record_path = step_dir / "preregistration_record.json"
    source_hash = sha256(CONFIG_PATH)
    freeze_commit = git_output("log", "-1", "--format=%H", "--", str(CONFIG_PATH))
    if not record_path.exists():
        present = sorted(
            name for name in RESULT_ONLY_NAMES if (step_dir / name).exists()
        )
        if present:
            raise RuntimeError(
                "S08 result artifacts existed before preregistration capture: "
                + ", ".join(present)
            )
        shutil.copyfile(CONFIG_PATH, artifact_config)
        record = {
            "schema": "eidosoma.e01.s08_preregistration_record.v1",
            "researchStepId": "S08",
            "preregistrationVersion": config["preregistrationVersion"],
            "configurationCollectionVersion": config["configurationCollectionVersion"],
            "frozenAtUtc": datetime.now(UTC).isoformat(),
            "preregistrationCommit": freeze_commit,
            "preregistrationSourceSha256": source_hash,
            "preregistrationArtifactSha256": sha256(artifact_config),
            "canonicalOutcomeArtifactsAbsentAtFreeze": True,
            "priorVersionCommit": config["amendmentHistory"][0]["priorGitCommit"],
            "thresholdsChangedByAmendment": False,
        }
        write_json(record_path, record)
    else:
        record = load_json(record_path)
    errors: list[str] = []
    if not artifact_config.is_file():
        errors.append("artifact preregistration copy is missing")
    else:
        if CONFIG_PATH.read_bytes() != artifact_config.read_bytes():
            errors.append("repository and artifact preregistration bytes differ")
        if sha256(artifact_config) != record["preregistrationArtifactSha256"]:
            errors.append("artifact preregistration hash differs from freeze record")
    if source_hash != record["preregistrationSourceSha256"]:
        errors.append("source preregistration hash differs from freeze record")
    if not record.get("canonicalOutcomeArtifactsAbsentAtFreeze"):
        errors.append("outcome absence was not recorded at freeze")
    ancestor = (
        subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                record["preregistrationCommit"],
                "HEAD",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
        ).returncode
        == 0
    )
    if not ancestor:
        errors.append("preregistration commit is not an ancestor of HEAD")
    evidence = []
    for evidence_id, item in config["frozenInputs"].items():
        path = Path(item["path"])
        actual = sha256(path) if path.is_file() else None
        valid = actual == item["sha256"]
        evidence.append(
            {
                "evidenceId": evidence_id,
                "path": str(path),
                "expectedSha256": item["sha256"],
                "actualSha256": actual,
                "valid": valid,
            }
        )
        if not valid:
            errors.append(f"frozen input mismatch: {evidence_id}")
    return {
        "valid": not errors,
        "errors": errors,
        "record": record,
        "frozenInputs": evidence,
        "preregistrationCommitIsAncestor": ancestor,
    }


def registry_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["frozenInputs"]["specificationRegistry"]["path"])
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    owner_rows = [
        {
            "parameter": item["parameter"],
            "value": item["value"],
            "resolutionStatus": item["resolutionStatus"],
            "unresolved": item["unresolved"],
            "ambiguityId": item["ambiguityId"],
            "ownerStep": item["ownerStep"],
        }
        for item in registry["parameters"]
        if item.get("ownerStep") == "S08"
    ]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "registryVersion": registry["registryVersion"],
        "parameterCount": len(registry["parameters"]),
        "unresolvedParameterCount": registry["executionGate"][
            "unresolvedParameterCount"
        ],
        "unexpandedBranchSetCount": registry["executionGate"][
            "unexpandedBranchSetCount"
        ],
        "executable": registry["executionGate"]["executable"],
        "noSilentDefaults": registry["executionGate"]["noSilentDefaults"],
        "s08OwnerParameters": owner_rows,
    }


def fixture_seed_bundle(config: dict[str, Any]):
    seed = config["fixtureSeedContract"]
    namespace = isolated_stream_namespace(
        experiment_id=seed["experimentId"],
        specification_id=seed["specificationId"],
        trajectory_id=seed["trajectoryId"],
        replicate_index=int(seed["replicateIndex"]),
    )
    request = SeedRequest(
        experiment_id=seed["experimentId"],
        specification_id=seed["specificationId"],
        trajectory_id=seed["trajectoryId"],
        replicate_index=int(seed["replicateIndex"]),
        engine_id=seed["engineId"],
        root_seed_hex=seed["rootSeedHex"],
        coupling_policy=CouplingPolicy.TRAJECTORY_ISOLATED,
        coupling_reason=None,
        stream_namespaces={purpose: namespace for purpose in CANONICAL_STREAM_PURPOSES},
    )
    return derive_seed_bundle(request)


def build_fixtures(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, Any]]:
    bundle = fixture_seed_bundle(config)
    generators = bundle.fresh_generators()
    estimator = generators[StreamPurpose.ESTIMATOR]
    order_generator = generators[StreamPurpose.MACHINE_LEARNING]
    fixtures: list[dict[str, Any]] = []
    permutations: dict[str, list[int]] = {}
    for definition in config["fixtures"]:
        fixture_id = definition["fixtureId"]
        if fixture_id == "E01-S08-FIXTURE-TWO-ATTRACTORS-v1.0.0":
            first = np.asarray(definition["attractorA"], dtype=np.float64)
            second = np.asarray(definition["attractorB"], dtype=np.float64)
            probabilities = [first] * int(definition["counts"]["attractorA"])
            probabilities.extend(
                (1.0 - float(weight)) * first + float(weight) * second
                for weight in definition["bridgeBWeights"]
            )
            probabilities.extend([second] * int(definition["counts"]["attractorB"]))
            masses = definition["massCycle"]
            states = []
            for index, probability in enumerate(probabilities):
                mass = int(masses[index % len(masses)])
                state = estimator.multinomial(mass - len(probability), probability) + 1
                states.append(state.tolist())
        else:
            states = definition["states"]
        observation_ids = [f"g{index:04d}" for index in range(1, len(states) + 1)]
        fixtures.append(
            {
                "fixtureId": fixture_id,
                "provenance": definition["provenance"],
                "observationIds": observation_ids,
                "states": states,
                "stateSha256": json_sha256(states),
            }
        )
        permutations[fixture_id] = order_generator.permutation(len(states)).tolist()
    seed_manifest = {
        "schema": "eidosoma.e01.s08_fixture_seed_manifest.v1",
        "researchStepId": "S08",
        "seedRequestId": config["fixtureSeedContract"]["seedRequestId"],
        "seedBundle": bundle.to_payload(),
        "consumedPurposes": config["fixtureSeedContract"]["consumedPurposes"],
        "clusteringRngPolicy": "RNG_FREE_DETERMINISTIC",
        "orderAuditPermutations": permutations,
    }
    return fixtures, permutations, seed_manifest


def verify_upstream_fixture_views(
    config: dict[str, Any], fixtures: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    fixture_map = {item["fixtureId"]: item for item in fixtures}
    example_path = Path(config["frozenInputs"]["exampleTrajectory"]["path"])
    envelope = deserialize_envelope(example_path.read_bytes(), require_canonical=True)
    states = [
        generation["growth"]["finalState"]
        for generation in envelope["payload"]["generations"]
    ]
    expected = fixture_map["E01-S08-FIXTURE-S06-GROWTH-FINAL-v1.0.0"]["states"]
    checks.append(
        _check(
            states == expected,
            "UPSTREAM_S06_GROWTH_FINAL_VIEW",
            {
                "actual": states,
                "expected": expected,
                "payloadSha256": envelope["payloadSha256"],
            },
        )
    )
    source = load_json(
        Path(
            "/artifacts/E01_forensic_replication_bundle/software/historical_reference/verified_small_cases.json"
        )
    )
    case = next(
        item for item in source["cases"] if item["caseId"] == "HC11_nondrift_technique1"
    )
    checks.append(
        _check(
            case["actual"]["isNonDrift"] == [True, False, False, False, False]
            and case["actual"]["localScores"] == [1.0, 0.9, 0.7, 0.6, 0.0],
            "UPSTREAM_S04_HC11_ORACLE_PRESENT",
            case["actual"],
        )
    )
    return checks


def reference_configurations(
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, ClusterConfiguration]]:
    raw = {item["familyId"]: item for item in config["familyConfigurations"]}
    clusters: dict[str, ClusterConfiguration] = {}
    for family in CLUSTER_FAMILIES:
        item = raw[family]
        metric = {"Y_C": "cosine", "Y_E": "euclidean", "Y_A": "aitchison"}[family]
        clusters[family] = ClusterConfiguration(
            configuration_id=item["configurationId"],
            family_id=family,
            family_name=item["familyName"],
            evidence_class=item["evidenceClass"],
            metric=metric,
            representation=item["representation"],
            threshold=float(item["threshold"]),
            comparator=item["comparator"],
            minimum_cluster_size=int(item["minimumClusterSize"]),
            temporal_scope=item["temporalInformationScope"],
            zero_policy=item["zeroPolicy"],
        )
    return raw, clusters


def online_configuration(configuration: ClusterConfiguration) -> ClusterConfiguration:
    return replace(
        configuration,
        configuration_id=configuration.configuration_id.replace("-RETRO-", "-ONLINE-"),
        temporal_scope="past_only_online",
    )


def label_fixture(
    fixture: dict[str, Any],
    raw_configs: dict[str, dict[str, Any]],
    cluster_configs: dict[str, ClusterConfiguration],
) -> dict[str, LabelTraceResult]:
    fixture_id = fixture["fixtureId"]
    states = fixture["states"]
    observation_ids = fixture["observationIds"]
    historical = raw_configs["Y_H"]
    results = {
        "Y_H": historical_technique1_labels(
            states,
            trajectory_id=fixture_id,
            observation_ids=observation_ids,
            configuration_id=historical["configurationId"],
            threshold=float(historical["threshold"]),
            evidence_class=historical["evidenceClass"],
        )
    }
    for family, configuration in cluster_configs.items():
        results[family] = cluster_labels(
            states,
            trajectory_id=fixture_id,
            observation_ids=observation_ids,
            configuration=configuration,
        )
        results[f"{family}_ONLINE"] = cluster_labels(
            states,
            trajectory_id=fixture_id,
            observation_ids=observation_ids,
            configuration=online_configuration(configuration),
        )
    return results


def label_rows(
    fixture_results: dict[str, dict[str, LabelTraceResult]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order = ("Y_H", "Y_C", "Y_E", "Y_A", "Y_C_ONLINE", "Y_E_ONLINE", "Y_A_ONLINE")
    for fixture_id in sorted(fixture_results):
        for key in order:
            for record in fixture_results[fixture_id][key].rows:
                rows.append(record.as_dict())
    return rows


def label_array_payload(
    config: dict[str, Any],
    fixtures: list[dict[str, Any]],
    fixture_results: dict[str, dict[str, LabelTraceResult]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "eidosoma.e01.s08_label_arrays.v1",
        "researchStepId": "S08",
        "configurationCollectionVersion": config["configurationCollectionVersion"],
        "scopeBoundary": config["scopeBoundary"],
        "fixtures": [],
    }
    fixture_lookup = {item["fixtureId"]: item for item in fixtures}
    for fixture_id in sorted(fixture_results):
        arrays = []
        for key in (
            "Y_H",
            "Y_C",
            "Y_E",
            "Y_A",
            "Y_C_ONLINE",
            "Y_E_ONLINE",
            "Y_A_ONLINE",
        ):
            result = fixture_results[fixture_id][key]
            arrays.append(
                {
                    "configurationId": result.configuration_id,
                    "familyId": result.family_id,
                    "resultStatus": result.result_status,
                    "resultReason": result.result_reason,
                    "observationIds": [row.observation_id for row in result.rows],
                    "labelStatuses": [row.label_status for row in result.rows],
                    "isReplicator": [row.is_replicator for row in result.rows],
                    "clusterIds": [row.cluster_id for row in result.rows],
                    "componentIds": [row.component_id for row in result.rows],
                    "ineligibilityReasons": [
                        row.ineligibility_reason for row in result.rows
                    ],
                }
            )
        payload["fixtures"].append(
            {
                "fixtureId": fixture_id,
                "stateSha256": fixture_lookup[fixture_id]["stateSha256"],
                "observationCount": len(fixture_lookup[fixture_id]["states"]),
                "arrays": arrays,
            }
        )
    return payload


def _aligned_binary(
    first: LabelTraceResult, second: LabelTraceResult
) -> tuple[list[bool], list[bool], int, int]:
    first_map = {row.observation_id: row for row in first.rows}
    second_map = {row.observation_id: row for row in second.rows}
    left: list[bool] = []
    right: list[bool] = []
    first_null = 0
    second_null = 0
    for observation_id in sorted(set(first_map) & set(second_map)):
        row_a = first_map[observation_id]
        row_b = second_map[observation_id]
        if row_a.is_replicator is None:
            first_null += 1
        if row_b.is_replicator is None:
            second_null += 1
        if row_a.is_replicator is not None and row_b.is_replicator is not None:
            left.append(row_a.is_replicator)
            right.append(row_b.is_replicator)
    return left, right, first_null, second_null


def comparison_record(
    fixture_id: str,
    first_family: str,
    second_family: str,
    first: LabelTraceResult,
    second: LabelTraceResult,
) -> dict[str, Any]:
    left, right, first_null, second_null = _aligned_binary(first, second)
    left_array = np.asarray(left, dtype=bool)
    right_array = np.asarray(right, dtype=bool)
    intersection = int(np.count_nonzero(left_array & right_array))
    union = int(np.count_nonzero(left_array | right_array))
    positive_a = int(np.count_nonzero(left_array))
    positive_b = int(np.count_nonzero(right_array))
    agreements = int(np.count_nonzero(left_array == right_array))
    disagreements = len(left) - agreements
    denominator_positive = positive_a + positive_b
    return {
        "fixtureId": fixture_id,
        "familyA": first_family,
        "familyB": second_family,
        "configurationA": first.configuration_id,
        "configurationB": second.configuration_id,
        "totalObservations": len(first.rows),
        "commonNonNull": len(left),
        "nullInA": first_null,
        "nullInB": second_null,
        "positiveA": positive_a,
        "positiveB": positive_b,
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else None,
        "positiveAgreement": (
            2.0 * intersection / denominator_positive if denominator_positive else None
        ),
        "overallAgreement": agreements / len(left) if left else None,
        "disagreementCount": disagreements,
        "disagreementRate": disagreements / len(left) if left else None,
        "aPositiveBNegative": int(np.count_nonzero(left_array & ~right_array)),
        "aNegativeBPositive": int(np.count_nonzero(~left_array & right_array)),
        "binaryAdjustedRandIndex": (
            float(adjusted_rand_score(left, right)) if len(left) >= 2 else None
        ),
        "nullMetricReason": (
            "FEWER_THAN_TWO_COMMON_LABELS"
            if len(left) < 2
            else ("NO_POSITIVES_IN_EITHER_FAMILY" if union == 0 else None)
        ),
    }


def pooled_comparison_record(
    fixture_results: dict[str, dict[str, LabelTraceResult]],
    first_family: str,
    second_family: str,
) -> dict[str, Any]:
    left: list[bool] = []
    right: list[bool] = []
    null_a = 0
    null_b = 0
    total = 0
    for fixture_id in sorted(fixture_results):
        first = fixture_results[fixture_id][first_family]
        second = fixture_results[fixture_id][second_family]
        current_left, current_right, current_null_a, current_null_b = _aligned_binary(
            first, second
        )
        left.extend(current_left)
        right.extend(current_right)
        null_a += current_null_a
        null_b += current_null_b
        total += len(first.rows)
    left_array = np.asarray(left, dtype=bool)
    right_array = np.asarray(right, dtype=bool)
    intersection = int(np.count_nonzero(left_array & right_array))
    union = int(np.count_nonzero(left_array | right_array))
    positives_a = int(np.count_nonzero(left_array))
    positives_b = int(np.count_nonzero(right_array))
    agreements = int(np.count_nonzero(left_array == right_array))
    return {
        "fixtureId": "POOLED_ALL_FIXTURES",
        "familyA": first_family,
        "familyB": second_family,
        "configurationA": "POOLED_REFERENCE_CONFIGURATIONS",
        "configurationB": "POOLED_REFERENCE_CONFIGURATIONS",
        "totalObservations": total,
        "commonNonNull": len(left),
        "nullInA": null_a,
        "nullInB": null_b,
        "positiveA": positives_a,
        "positiveB": positives_b,
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else None,
        "positiveAgreement": (
            2 * intersection / (positives_a + positives_b)
            if positives_a + positives_b
            else None
        ),
        "overallAgreement": agreements / len(left) if left else None,
        "disagreementCount": len(left) - agreements,
        "disagreementRate": (len(left) - agreements) / len(left) if left else None,
        "aPositiveBNegative": int(np.count_nonzero(left_array & ~right_array)),
        "aNegativeBPositive": int(np.count_nonzero(~left_array & right_array)),
        "binaryAdjustedRandIndex": (
            float(adjusted_rand_score(left, right)) if len(left) >= 2 else None
        ),
        "nullMetricReason": (
            "FEWER_THAN_TWO_COMMON_LABELS"
            if len(left) < 2
            else ("NO_POSITIVES_IN_EITHER_FAMILY" if union == 0 else None)
        ),
    }


def cluster_ari(
    fixture_results: dict[str, dict[str, LabelTraceResult]],
    first_family: str,
    second_family: str,
) -> tuple[float | None, int, int]:
    left: list[str] = []
    right: list[str] = []
    excluded = 0
    for fixture_id in sorted(fixture_results):
        first = {
            row.observation_id: row
            for row in fixture_results[fixture_id][first_family].rows
        }
        second = {
            row.observation_id: row
            for row in fixture_results[fixture_id][second_family].rows
        }
        for observation_id in sorted(first):
            row_a = first[observation_id]
            row_b = second[observation_id]
            if row_a.is_replicator is None or row_b.is_replicator is None:
                excluded += 1
                continue
            left.append(
                f"{fixture_id}::{row_a.cluster_id}"
                if row_a.cluster_id is not None
                else f"{fixture_id}::NOISE_SHARED_CLASS"
            )
            right.append(
                f"{fixture_id}::{row_b.cluster_id}"
                if row_b.cluster_id is not None
                else f"{fixture_id}::NOISE_SHARED_CLASS"
            )
    ari = float(adjusted_rand_score(left, right)) if len(left) >= 2 else None
    return ari, len(left), excluded


def matrix_rows(
    families: tuple[str, ...], values: dict[tuple[str, str], float | None]
) -> list[dict[str, Any]]:
    return [
        {"family": family, **{other: values[(family, other)] for other in families}}
        for family in families
    ]


def comparison_outputs(
    fixture_results: dict[str, dict[str, LabelTraceResult]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    long_rows: list[dict[str, Any]] = []
    run_disagreement: list[dict[str, Any]] = []
    for fixture_id in sorted(fixture_results):
        for family_a in REFERENCE_FAMILIES:
            for family_b in REFERENCE_FAMILIES:
                long_rows.append(
                    comparison_record(
                        fixture_id,
                        family_a,
                        family_b,
                        fixture_results[fixture_id][family_a],
                        fixture_results[fixture_id][family_b],
                    )
                )
        for left_index, family_a in enumerate(REFERENCE_FAMILIES):
            for family_b in REFERENCE_FAMILIES[left_index + 1 :]:
                run_disagreement.append(
                    comparison_record(
                        fixture_id,
                        family_a,
                        family_b,
                        fixture_results[fixture_id][family_a],
                        fixture_results[fixture_id][family_b],
                    )
                )
    pooled: dict[tuple[str, str], dict[str, Any]] = {}
    for family_a in REFERENCE_FAMILIES:
        for family_b in REFERENCE_FAMILIES:
            record = pooled_comparison_record(fixture_results, family_a, family_b)
            pooled[(family_a, family_b)] = record
            long_rows.append(record)
    jaccard = matrix_rows(
        REFERENCE_FAMILIES,
        {
            (left, right): pooled[(left, right)]["jaccard"]
            for left in REFERENCE_FAMILIES
            for right in REFERENCE_FAMILIES
        },
    )
    binary_ari = matrix_rows(
        REFERENCE_FAMILIES,
        {
            (left, right): pooled[(left, right)]["binaryAdjustedRandIndex"]
            for left in REFERENCE_FAMILIES
            for right in REFERENCE_FAMILIES
        },
    )
    cluster_values: dict[tuple[str, str], float | None] = {}
    cluster_metadata: list[dict[str, Any]] = []
    for family_a in CLUSTER_FAMILIES:
        for family_b in CLUSTER_FAMILIES:
            ari, common, excluded = cluster_ari(fixture_results, family_a, family_b)
            cluster_values[(family_a, family_b)] = ari
            cluster_metadata.append(
                {
                    "familyA": family_a,
                    "familyB": family_b,
                    "clusterAdjustedRandIndex": ari,
                    "commonNonNull": common,
                    "excludedIneligible": excluded,
                    "driftEncoding": "SHARED_NOISE_CLASS_WITHIN_FIXTURE",
                }
            )
    cluster_matrix = matrix_rows(CLUSTER_FAMILIES, cluster_values)
    return (
        long_rows,
        run_disagreement,
        jaccard,
        binary_ari,
        [{"matrix": cluster_matrix, "metadata": cluster_metadata}],
    )


def observation_disagreement_rows(
    fixture_results: dict[str, dict[str, LabelTraceResult]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture_id in sorted(fixture_results):
        maps = {
            family: {
                row.observation_id: row
                for row in fixture_results[fixture_id][family].rows
            }
            for family in REFERENCE_FAMILIES
        }
        observation_ids = [
            row.observation_id for row in fixture_results[fixture_id]["Y_H"].rows
        ]
        for observation_id in observation_ids:
            labels = {
                family: maps[family][observation_id].is_replicator
                for family in REFERENCE_FAMILIES
            }
            statuses = {
                family: maps[family][observation_id].label_status
                for family in REFERENCE_FAMILIES
            }
            nonnull = [value for value in labels.values() if value is not None]
            rows.append(
                {
                    "fixtureId": fixture_id,
                    "observationId": observation_id,
                    **{
                        f"{family}Label": labels[family]
                        for family in REFERENCE_FAMILIES
                    },
                    **{
                        f"{family}Status": statuses[family]
                        for family in REFERENCE_FAMILIES
                    },
                    "nonNullFamilyCount": len(nonnull),
                    "positiveFamilyCount": sum(nonnull),
                    "allNonNullFamiliesAgree": len(set(nonnull)) <= 1
                    if nonnull
                    else None,
                    "hasExplicitIneligibility": any(
                        value is None for value in labels.values()
                    ),
                }
            )
    return rows


def temporal_scope_rows(
    fixture_results: dict[str, dict[str, LabelTraceResult]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture_id in sorted(fixture_results):
        for family in CLUSTER_FAMILIES:
            comparison = comparison_record(
                fixture_id,
                family,
                f"{family}_ONLINE",
                fixture_results[fixture_id][family],
                fixture_results[fixture_id][f"{family}_ONLINE"],
            )
            rows.append(
                {
                    "fixtureId": fixture_id,
                    "familyId": family,
                    "retrospectiveConfigurationId": fixture_results[fixture_id][
                        family
                    ].configuration_id,
                    "pastOnlyConfigurationId": fixture_results[fixture_id][
                        f"{family}_ONLINE"
                    ].configuration_id,
                    "commonNonNull": comparison["commonNonNull"],
                    "flipCount": comparison["disagreementCount"],
                    "flipRate": comparison["disagreementRate"],
                    "retrospectivePositivePastOnlyNegative": comparison[
                        "aPositiveBNegative"
                    ],
                    "retrospectiveNegativePastOnlyPositive": comparison[
                        "aNegativeBPositive"
                    ],
                    "binaryAdjustedRandIndex": comparison["binaryAdjustedRandIndex"],
                    "interpretation": "RETROSPECTIVE_DEPENDENCE_DIAGNOSTIC_NOT_BRANCH_RESOLUTION",
                }
            )
    return rows


def threshold_tag(value: float) -> str:
    return format(float(value), ".6g").replace("-", "m").replace(".", "p")


def label_summary(
    result: LabelTraceResult, reference: LabelTraceResult
) -> dict[str, Any]:
    labels = [row.is_replicator for row in result.rows]
    labeled = [value for value in labels if value is not None]
    left, right, _, _ = _aligned_binary(result, reference)
    flips = sum(first != second for first, second in zip(left, right, strict=True))
    components = {row.cluster_id for row in result.rows if row.cluster_id is not None}
    return {
        "totalObservations": len(labels),
        "labeledObservations": len(labeled),
        "ineligibleObservations": len(labels) - len(labeled),
        "replicatorObservations": sum(labeled),
        "replicatorFractionAmongLabeled": sum(labeled) / len(labeled)
        if labeled
        else None,
        "replicatorClusterCount": len(components),
        "commonWithReference": len(left),
        "flipCountVersusReference": flips,
        "binaryAdjustedRandVersusReference": (
            float(adjusted_rand_score(left, right)) if len(left) >= 2 else None
        ),
    }


def sensitivity_outputs(
    config: dict[str, Any],
    fixtures: list[dict[str, Any]],
    raw_configs: dict[str, dict[str, Any]],
    cluster_configs: dict[str, ClusterConfiguration],
    fixture_results: dict[str, dict[str, LabelTraceResult]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    configurations: dict[str, dict[str, Any]] = {}
    for family in REFERENCE_FAMILIES:
        item = raw_configs[family]
        configurations[item["configurationId"]] = {
            "configurationId": item["configurationId"],
            "familyId": family,
            "role": "REFERENCE_LABEL_FAMILY",
            "evidenceClass": item["evidenceClass"],
            "metric": item["metric"],
            "threshold": item["threshold"],
            "minimumClusterSize": item.get("minimumClusterSize"),
            "temporalInformationScope": item["temporalInformationScope"],
            "representation": item["representation"],
            "zeroPolicy": item["zeroPolicy"],
        }
    for family, cluster in cluster_configs.items():
        online = online_configuration(cluster)
        configurations[online.configuration_id] = {
            "configurationId": online.configuration_id,
            "familyId": family,
            "role": "PAST_ONLY_TEMPORAL_SCOPE_COMPANION",
            "evidenceClass": online.evidence_class,
            "metric": online.metric,
            "threshold": online.threshold,
            "minimumClusterSize": online.minimum_cluster_size,
            "temporalInformationScope": online.temporal_scope,
            "representation": online.representation,
            "zeroPolicy": online.zero_policy,
        }

    threshold_grid = config["thresholdSensitivity"]
    family_grids = {
        "Y_H": threshold_grid["historicalHStrictGreaterThan"],
        "Y_C": threshold_grid["cosineHStrictGreaterThan"],
        "Y_E": threshold_grid["euclideanDistanceStrictLessThan"],
        "Y_A": threshold_grid["aitchisonDistanceStrictLessThan"],
    }
    for fixture in fixtures:
        fixture_id = fixture["fixtureId"]
        reference_results = fixture_results[fixture_id]
        for family, grid in family_grids.items():
            for threshold in grid:
                tag = threshold_tag(float(threshold))
                if family == "Y_H":
                    configuration_id = f"E01-S08-YH-T1-THRESH-{tag}-v1.0.0"
                    result = historical_technique1_labels(
                        fixture["states"],
                        trajectory_id=fixture_id,
                        observation_ids=fixture["observationIds"],
                        configuration_id=configuration_id,
                        threshold=float(threshold),
                        evidence_class=raw_configs[family]["evidenceClass"],
                    )
                    minimum_size = None
                    metric_name = "historical_H_local_score"
                    representation = raw_configs[family]["representation"]
                    zero_policy = raw_configs[family]["zeroPolicy"]
                else:
                    base = cluster_configs[family]
                    configuration_id = f"E01-S08-{family}-THRESH-{tag}-MIN{base.minimum_cluster_size}-RETRO-v1.0.0"
                    candidate = replace(
                        base,
                        configuration_id=configuration_id,
                        threshold=float(threshold),
                    )
                    result = cluster_labels(
                        fixture["states"],
                        trajectory_id=fixture_id,
                        observation_ids=fixture["observationIds"],
                        configuration=candidate,
                    )
                    minimum_size = base.minimum_cluster_size
                    metric_name = base.metric
                    representation = base.representation
                    zero_policy = base.zero_policy
                configurations.setdefault(
                    configuration_id,
                    {
                        "configurationId": configuration_id,
                        "familyId": family,
                        "role": "THRESHOLD_SENSITIVITY",
                        "evidenceClass": raw_configs[family]["evidenceClass"],
                        "metric": metric_name,
                        "threshold": float(threshold),
                        "minimumClusterSize": minimum_size,
                        "temporalInformationScope": raw_configs[family][
                            "temporalInformationScope"
                        ],
                        "representation": representation,
                        "zeroPolicy": zero_policy,
                    },
                )
                rows.append(
                    {
                        "fixtureId": fixture_id,
                        "familyId": family,
                        "configurationId": configuration_id,
                        "sensitivityType": "threshold",
                        "threshold": float(threshold),
                        "minimumClusterSize": minimum_size,
                        **label_summary(result, reference_results[family]),
                    }
                )
        for family in CLUSTER_FAMILIES:
            base = cluster_configs[family]
            for minimum_size in config["commonClusteringContract"][
                "minimumClusterSizeSensitivity"
            ]:
                configuration_id = (
                    f"E01-S08-{family}-THRESH-{threshold_tag(base.threshold)}-"
                    f"MIN{minimum_size}-RETRO-PERSISTENCE-v1.0.0"
                )
                candidate = replace(
                    base,
                    configuration_id=configuration_id,
                    minimum_cluster_size=int(minimum_size),
                )
                result = cluster_labels(
                    fixture["states"],
                    trajectory_id=fixture_id,
                    observation_ids=fixture["observationIds"],
                    configuration=candidate,
                )
                configurations.setdefault(
                    configuration_id,
                    {
                        "configurationId": configuration_id,
                        "familyId": family,
                        "role": "MINIMUM_CLUSTER_SIZE_SENSITIVITY",
                        "evidenceClass": base.evidence_class,
                        "metric": base.metric,
                        "threshold": base.threshold,
                        "minimumClusterSize": int(minimum_size),
                        "temporalInformationScope": base.temporal_scope,
                        "representation": base.representation,
                        "zeroPolicy": base.zero_policy,
                    },
                )
                rows.append(
                    {
                        "fixtureId": fixture_id,
                        "familyId": family,
                        "configurationId": configuration_id,
                        "sensitivityType": "minimum_cluster_size",
                        "threshold": base.threshold,
                        "minimumClusterSize": int(minimum_size),
                        **label_summary(result, reference_results[family]),
                    }
                )
    diagnostic = config["historicalTechnique2Diagnostics"]
    for drift_size in diagnostic["driftSizes"]:
        configuration_id = (
            f"E01-S08-YH-T2-HGT{threshold_tag(float(diagnostic['threshold']))}-"
            f"D{drift_size}-v1.0.0"
        )
        configurations[configuration_id] = {
            "configurationId": configuration_id,
            "familyId": "Y_H",
            "role": "OPTIONAL_HISTORICAL_TECHNIQUE2_DIAGNOSTIC",
            "evidenceClass": diagnostic["evidenceClass"],
            "metric": "historical_H_adjacent",
            "threshold": diagnostic["threshold"],
            "minimumClusterSize": None,
            "driftSize": drift_size,
            "temporalInformationScope": "source_consecutive_similarity_with_backward_shift",
            "representation": raw_configs["Y_H"]["representation"],
            "zeroPolicy": raw_configs["Y_H"]["zeroPolicy"],
        }
    return rows, [configurations[key] for key in sorted(configurations)]


def technique2_outputs(
    config: dict[str, Any], fixtures: list[dict[str, Any]]
) -> dict[str, Any]:
    diagnostic = config["historicalTechnique2Diagnostics"]
    records = []
    for fixture in fixtures:
        for drift_size in diagnostic["driftSizes"]:
            configuration_id = (
                f"E01-S08-YH-T2-HGT{threshold_tag(float(diagnostic['threshold']))}-"
                f"D{drift_size}-v1.0.0"
            )
            records.append(
                historical_technique2_diagnostic(
                    fixture["states"],
                    trajectory_id=fixture["fixtureId"],
                    configuration_id=configuration_id,
                    threshold=float(diagnostic["threshold"]),
                    drift_size=int(drift_size),
                )
            )
    return {
        "schema": "eidosoma.e01.s08_historical_technique2_diagnostics.v1",
        "researchStepId": "S08",
        "sourceEdgePolicy": diagnostic["sourceEdgePolicy"],
        "records": records,
        "summary": {
            "recordCount": len(records),
            "okCount": sum(record["status"] == "OK" for record in records),
            "sourceDomainErrorCount": sum(
                record["status"] == "ERROR_SOURCE_DOMAIN" for record in records
            ),
            "sourceRepairsApplied": sum(
                bool(record["sourceRepairApplied"]) for record in records
            ),
        },
    }


def recurrence_rows(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        recurrence = continuous_past_recurrence(fixture["states"])
        for observation_id, value in zip(
            fixture["observationIds"], recurrence, strict=True
        ):
            rows.append(
                {
                    "fixtureId": fixture["fixtureId"],
                    "observationId": observation_id,
                    "continuousPastRecurrenceR_g": value,
                    "status": "DEFINED"
                    if value is not None
                    else "UNDEFINED_NO_ELIGIBLE_PAST",
                    "binaryLabelRole": "NONE_CONTINUOUS_DIAGNOSTIC_ONLY",
                }
            )
    return rows


def label_schema() -> dict[str, Any]:
    identifier = {"type": "string", "minLength": 1}
    nullable_string = {"type": ["string", "null"]}
    nullable_boolean = {"type": ["boolean", "null"]}
    array_record = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "configurationId",
            "familyId",
            "resultStatus",
            "resultReason",
            "observationIds",
            "labelStatuses",
            "isReplicator",
            "clusterIds",
            "componentIds",
            "ineligibilityReasons",
        ],
        "properties": {
            "configurationId": identifier,
            "familyId": identifier,
            "resultStatus": identifier,
            "resultReason": nullable_string,
            "observationIds": {"type": "array", "minItems": 1, "items": identifier},
            "labelStatuses": {"type": "array", "minItems": 1, "items": identifier},
            "isReplicator": {"type": "array", "minItems": 1, "items": nullable_boolean},
            "clusterIds": {"type": "array", "minItems": 1, "items": nullable_string},
            "componentIds": {"type": "array", "minItems": 1, "items": nullable_string},
            "ineligibilityReasons": {
                "type": "array",
                "minItems": 1,
                "items": nullable_string,
            },
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:eidosoma:schema:e01:s08:label-arrays:v1",
        "title": "E01 S08 checksum-protected label arrays",
        "type": "object",
        "additionalProperties": False,
        "required": ["serializationVersion", "payloadSha256", "payload"],
        "properties": {
            "serializationVersion": {"const": "E01-canonical-json-v1.0.0"},
            "payloadSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "payload": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "schema",
                    "researchStepId",
                    "configurationCollectionVersion",
                    "scopeBoundary",
                    "fixtures",
                ],
                "properties": {
                    "schema": {"const": "eidosoma.e01.s08_label_arrays.v1"},
                    "researchStepId": {"const": "S08"},
                    "configurationCollectionVersion": identifier,
                    "scopeBoundary": identifier,
                    "fixtures": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "fixtureId",
                                "stateSha256",
                                "observationCount",
                                "arrays",
                            ],
                            "properties": {
                                "fixtureId": identifier,
                                "stateSha256": {
                                    "type": "string",
                                    "pattern": "^[0-9a-f]{64}$",
                                },
                                "observationCount": {"type": "integer", "minimum": 1},
                                "arrays": {
                                    "type": "array",
                                    "minItems": 7,
                                    "maxItems": 7,
                                    "items": array_record,
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def _label_map(result: LabelTraceResult) -> dict[str, tuple[bool | None, str | None]]:
    return {
        row.observation_id: (row.is_replicator, row.component_id) for row in result.rows
    }


def invariance_checks(
    fixtures: list[dict[str, Any]],
    permutations: dict[str, list[int]],
    raw_configs: dict[str, dict[str, Any]],
    cluster_configs: dict[str, ClusterConfiguration],
    fixture_results: dict[str, dict[str, LabelTraceResult]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    fixture = next(
        item
        for item in fixtures
        if item["fixtureId"] == "E01-S08-FIXTURE-TWO-ATTRACTORS-v1.0.0"
    )
    states = np.asarray(fixture["states"], dtype=np.float64)
    ids = fixture["observationIds"]
    scales = np.arange(1, states.shape[0] + 1, dtype=np.float64)[:, None]
    component_order = np.array([2, 0, 3, 1])
    historical = raw_configs["Y_H"]
    baseline_h = fixture_results[fixture["fixtureId"]]["Y_H"]
    scaled_h = historical_technique1_labels(
        states * scales,
        trajectory_id=fixture["fixtureId"],
        observation_ids=ids,
        configuration_id=historical["configurationId"],
        threshold=float(historical["threshold"]),
        evidence_class=historical["evidenceClass"],
    )
    permuted_components_h = historical_technique1_labels(
        states[:, component_order],
        trajectory_id=fixture["fixtureId"],
        observation_ids=ids,
        configuration_id=historical["configurationId"],
        threshold=float(historical["threshold"]),
        evidence_class=historical["evidenceClass"],
    )
    checks.append(
        _check(
            [row.is_replicator for row in baseline_h.rows]
            == [row.is_replicator for row in scaled_h.rows]
            == [row.is_replicator for row in permuted_components_h.rows],
            "HISTORICAL_SCALE_AND_COMPONENT_PERMUTATION_INVARIANCE",
            {"observationCount": len(ids)},
        )
    )
    order = np.asarray(permutations[fixture["fixtureId"]], dtype=np.int64)
    for family, configuration in cluster_configs.items():
        baseline = fixture_results[fixture["fixtureId"]][family]
        scaled = cluster_labels(
            states * scales,
            trajectory_id=fixture["fixtureId"],
            observation_ids=ids,
            configuration=configuration,
        )
        component_permuted = cluster_labels(
            states[:, component_order],
            trajectory_id=fixture["fixtureId"],
            observation_ids=ids,
            configuration=configuration,
        )
        order_permuted = cluster_labels(
            states[order],
            trajectory_id=fixture["fixtureId"],
            observation_ids=[ids[index] for index in order],
            configuration=configuration,
        )
        metric_baseline = metric_result(states, metric=configuration.metric)
        metric_scaled = metric_result(states * scales, metric=configuration.metric)
        metric_component_permuted = metric_result(
            states[:, component_order], metric=configuration.metric
        )
        finite = np.isfinite(metric_baseline.values)
        metric_invariant = bool(
            np.allclose(
                metric_baseline.values[finite],
                metric_scaled.values[finite],
                rtol=1e-13,
                atol=1e-13,
            )
            and np.allclose(
                metric_baseline.values[finite],
                metric_component_permuted.values[finite],
                rtol=1e-13,
                atol=1e-13,
            )
        )
        checks.append(
            _check(
                _label_map(baseline)
                == _label_map(scaled)
                == _label_map(component_permuted)
                and metric_invariant,
                f"{family}_SCALE_AND_COMPONENT_PERMUTATION_INVARIANCE",
                {"metric": configuration.metric, "observationCount": len(ids)},
            )
        )
        checks.append(
            _check(
                _label_map(baseline) == _label_map(order_permuted),
                f"{family}_RETROSPECTIVE_ORDER_PERMUTATION_INVARIANCE",
                {
                    "permutation": order.tolist(),
                    "clusteringRngPolicy": "RNG_FREE_DETERMINISTIC",
                },
            )
        )
        online = online_configuration(configuration)
        first_online = fixture_results[fixture["fixtureId"]][f"{family}_ONLINE"]
        replay_online = cluster_labels(
            states,
            trajectory_id=fixture["fixtureId"],
            observation_ids=ids,
            configuration=online,
        )
        checks.append(
            _check(
                [row.as_dict() for row in first_online.rows]
                == [row.as_dict() for row in replay_online.rows],
                f"{family}_PAST_ONLY_SEQUENCE_PRESERVING_REPLAY",
                {"observationOrderPermuted": False},
            )
        )
    return checks


def edge_case_checks(
    fixtures: list[dict[str, Any]],
    fixture_results: dict[str, dict[str, LabelTraceResult]],
    technique2: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    exact_similarity = strict_similarity_adjacency(
        [[1.0, 0.9], [0.9, 1.0]], threshold=0.9, eligible=[True, True]
    )
    exact_distance = strict_distance_adjacency(
        [[0.0, 0.1], [0.1, 0.0]], threshold=0.1, eligible=[True, True]
    )
    checks.append(
        _check(
            not exact_similarity.any() and not exact_distance.any(),
            "STRICT_THRESHOLD_BOUNDARIES",
            {"HAtThresholdAccepted": False, "distanceAtThresholdAccepted": False},
        )
    )
    hc11 = fixture_results["E01-S08-FIXTURE-S04-HC11-v1.0.0"]["Y_H"]
    hc11_labels = [row.is_replicator for row in hc11.rows]
    hc11_scores = [row.historical_local_score for row in hc11.rows]
    checks.append(
        _check(
            hc11_labels == [True, False, False, False, False]
            and np.allclose(hc11_scores, [1.0, 0.9, 0.7, 0.6, 0.0], rtol=0, atol=1e-15)
            and hc11.rows[-1].source_padding,
            "HISTORICAL_HC11_EXACT_ORACLE",
            {"labels": hc11_labels, "localScores": hc11_scores},
        )
    )
    zero_fixture = fixture_results["E01-S08-FIXTURE-ZERO-AND-EXTINCTION-v1.0.0"]
    aitchison_rows = zero_fixture["Y_A"].rows
    zero_states = next(
        item["states"]
        for item in fixtures
        if item["fixtureId"] == "E01-S08-FIXTURE-ZERO-AND-EXTINCTION-v1.0.0"
    )
    expected_ineligible = [any(value == 0 for value in state) for state in zero_states]
    observed_ineligible = [row.is_replicator is None for row in aitchison_rows]
    checks.append(
        _check(
            observed_ineligible == expected_ineligible
            and all(
                row.ineligibility_reason is not None
                for row in aitchison_rows
                if row.is_replicator is None
            ),
            "AITCHISON_ZERO_NO_REPLACEMENT_NO_DROP",
            {
                "expectedIneligible": expected_ineligible,
                "observedIneligible": observed_ineligible,
            },
        )
    )
    injected: list[dict[str, Any]] = []
    for injection_id, action, expected in (
        (
            "NEGATIVE_STATE_REJECTED",
            lambda: metric_result([[1.0, -1.0]], metric="euclidean"),
            LabelContractError,
        ),
        (
            "NONFINITE_STATE_REJECTED",
            lambda: metric_result([[1.0, np.inf]], metric="cosine"),
            LabelContractError,
        ),
    ):
        try:
            action()
            success = False
            error_type = None
        except expected as exc:
            success = True
            error_type = type(exc).__name__
        injected.append(
            {"injectionId": injection_id, "success": success, "errorType": error_type}
        )
    checks.append(
        _check(
            all(record["success"] for record in injected),
            "FAIL_CLOSED_INVALID_STATE_INJECTIONS",
            injected,
        )
    )
    checks.append(
        _check(
            technique2["summary"]["sourceRepairsApplied"] == 0
            and technique2["summary"]["sourceDomainErrorCount"] > 0,
            "HISTORICAL_TECHNIQUE2_SOURCE_EDGE_PRESERVED",
            technique2["summary"],
        )
    )
    return checks, {
        "schema": "eidosoma.e01.s08_edge_case_validation.v1",
        "researchStepId": "S08",
        "checks": checks,
        "failureInjections": injected,
    }


def plot_disagreement(
    path: Path, fixture_results: dict[str, dict[str, LabelTraceResult]]
) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    fixtures = sorted(fixture_results)
    figure, axes = plt.subplots(
        len(fixtures), 1, figsize=(12, 2.0 * len(fixtures)), squeeze=False
    )
    cmap = ListedColormap(["#9e9e9e", "#f7f7f7", "#2166ac"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    for axis, fixture_id in zip(axes[:, 0], fixtures, strict=True):
        matrix = []
        for family in REFERENCE_FAMILIES:
            matrix.append(
                [
                    -1 if row.is_replicator is None else int(row.is_replicator)
                    for row in fixture_results[fixture_id][family].rows
                ]
            )
        image = axis.imshow(
            matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="none"
        )
        _ = image
        axis.set_yticks(range(len(REFERENCE_FAMILIES)), REFERENCE_FAMILIES)
        axis.set_xticks(range(len(matrix[0])))
        axis.set_xticklabels(range(1, len(matrix[0]) + 1), fontsize=7)
        axis.set_ylabel("family")
        axis.set_title(fixture_id, fontsize=9, loc="left")
    axes[-1, 0].set_xlabel(
        "observation index (gray=ineligible, white=drift, blue=replicator)"
    )
    figure.suptitle("S08 reference-label disagreement map", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_threshold(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, family in zip(axes.ravel(), REFERENCE_FAMILIES, strict=True):
        family_rows = [
            row
            for row in rows
            if row["familyId"] == family and row["sensitivityType"] == "threshold"
        ]
        for fixture_id in sorted({row["fixtureId"] for row in family_rows}):
            selected = sorted(
                (row for row in family_rows if row["fixtureId"] == fixture_id),
                key=lambda row: row["threshold"],
            )
            axis.plot(
                [row["threshold"] for row in selected],
                [
                    np.nan
                    if row["replicatorFractionAmongLabeled"] is None
                    else row["replicatorFractionAmongLabeled"]
                    for row in selected
                ],
                marker="o",
                linewidth=1,
                markersize=3,
                label=fixture_id.replace("E01-S08-FIXTURE-", "").replace("-v1.0.0", ""),
            )
        axis.set_title(family)
        axis.set_xlabel("strict threshold")
        axis.set_ylabel("replicator fraction among labeled")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=6, loc="best")
    figure.suptitle("Frozen S08 threshold sensitivity (no post-outcome selection)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_overlap(path: Path, rows: list[dict[str, Any]]) -> None:
    values = np.array(
        [
            [
                np.nan if row[family] is None else row[family]
                for family in REFERENCE_FAMILIES
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(4), REFERENCE_FAMILIES)
    axis.set_yticks(range(4), REFERENCE_FAMILIES)
    for row in range(4):
        for column in range(4):
            text = (
                "NA"
                if not np.isfinite(values[row, column])
                else f"{values[row, column]:.2f}"
            )
            axis.text(column, row, text, ha="center", va="center", color="white")
    axis.set_title("Pooled positive-label Jaccard overlap")
    figure.colorbar(image, ax=axis, label="Jaccard")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def shared_contract(
    config: dict[str, Any],
    registry: dict[str, Any],
    configurations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "eidosoma.e01.s08_label_family_contract.v1",
        "researchStepId": "S08",
        "contractVersion": config["configurationCollectionVersion"],
        "status": "RECONSTRUCTION_VALIDATION_CONTRACT",
        "scopeBoundary": config["scopeBoundary"],
        "sourceTraceableHistoricalBranch": config["familyConfigurations"][0],
        "validationOnlyClusterBranches": config["familyConfigurations"][1:],
        "commonClusteringContract": config["commonClusteringContract"],
        "temporalScopeCompanions": config["temporalScopeCompanions"],
        "historicalTechnique2Diagnostics": config["historicalTechnique2Diagnostics"],
        "thresholdSensitivity": config["thresholdSensitivity"],
        "comparisonContract": config["comparisonContract"],
        "registryBoundary": {
            "registryVersion": registry["registryVersion"],
            "registrySha256": registry["sha256"],
            "executable": registry["executable"],
            "noSilentDefaults": registry["noSilentDefaults"],
            "s08OwnerParameters": registry["s08OwnerParameters"],
            "action": "PRESERVE_ALL_SENTINELS_AND_BRANCH_SETS",
        },
        "materializedConfigurationCount": len(configurations),
        "authorCodeIdentity": "UNAVAILABLE::NO_AUTHOR_CODE_RELEASE_FOUND",
        "legacyMatlabRngIdentity": "UNRESOLVED::LEGACY_MATLAB_RNG_ALGORITHM_AND_GLOBAL_STATE_ORDER",
        "zeroResolutionBoundary": "DEFERRED_TO_S09_NO_PSEUDOCOUNT_OR_REPLACEMENT_IN_S08",
        "claimBoundary": "FIXTURE_VALIDATION_ONLY_NOT_PAPER_CLAIM_ADJUDICATION",
    }


def artifact_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(set(paths)):
        if path.is_file():
            records.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return records


def build_manifest(
    artifact_root: Path,
    config: dict[str, Any],
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    step_dir = artifact_root / STEP_RELATIVE
    shared_dir = artifact_root / SHARED_RELATIVE
    manifest_path = step_dir / "artifact_manifest.json"
    outputs = [
        path
        for base in (step_dir, shared_dir)
        for path in base.rglob("*")
        if path.is_file() and path != manifest_path
    ]
    repository_files = [
        CONFIG_PATH,
        REPOSITORY_ROOT / "src/e01_replicator_labels/__init__.py",
        REPOSITORY_ROOT / "src/e01_replicator_labels/labels.py",
        REPOSITORY_ROOT / "scripts/e01/build_s08_label_artifacts.py",
        REPOSITORY_ROOT / "tests/e01/test_replicator_labels.py",
    ]
    return {
        "schema": "eidosoma.e01.s08_artifact_manifest.v1",
        "researchStepId": "S08",
        "configurationCollectionVersion": config["configurationCollectionVersion"],
        "repository": str(REPOSITORY_ROOT),
        "repositoryCommit": git_output("rev-parse", "HEAD"),
        "branch": git_output("branch", "--show-current"),
        "preregistrationCommit": preregistration["record"]["preregistrationCommit"],
        "preregistrationSha256": sha256(CONFIG_PATH),
        "repositoryFiles": artifact_records(repository_files),
        "artifacts": artifact_records(outputs),
        "reportPresent": (step_dir / "research_step_full_results.md").is_file(),
        "manifestSelfHashExcluded": True,
        "s09Absent": not (artifact_root / "research_steps/S09").exists(),
    }


def run(artifact_root: Path) -> dict[str, Any]:
    config = load_config()
    preregistration = freeze_or_verify_preregistration(artifact_root, config)
    if not preregistration["valid"]:
        raise RuntimeError(
            "S08 preregistration validation failed: "
            + "; ".join(preregistration["errors"])
        )
    step_dir = artifact_root / STEP_RELATIVE
    shared_dir = artifact_root / SHARED_RELATIVE
    step_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)

    registry_before = registry_snapshot(config)
    fixtures, permutations, seed_manifest = build_fixtures(config)
    replay_fixtures, replay_permutations, _ = build_fixtures(config)
    fixture_replay_exact = (
        fixtures == replay_fixtures and permutations == replay_permutations
    )
    raw_configs, cluster_configs = reference_configurations(config)
    fixture_results = {
        fixture["fixtureId"]: label_fixture(fixture, raw_configs, cluster_configs)
        for fixture in fixtures
    }
    replay_results = {
        fixture["fixtureId"]: label_fixture(fixture, raw_configs, cluster_configs)
        for fixture in replay_fixtures
    }
    arrays_payload = label_array_payload(config, fixtures, fixture_results)
    replay_arrays_payload = label_array_payload(config, replay_fixtures, replay_results)
    arrays_envelope = make_envelope(arrays_payload)
    arrays_bytes = serialize_envelope(arrays_envelope)
    replay_bytes = serialize_envelope(make_envelope(replay_arrays_payload))

    output_rows = label_rows(fixture_results)
    sensitivity_rows, materialized_configurations = sensitivity_outputs(
        config, fixtures, raw_configs, cluster_configs, fixture_results
    )
    technique2 = technique2_outputs(config, fixtures)
    recurrence = recurrence_rows(fixtures)
    (
        overlap_rows,
        run_disagreement,
        jaccard_matrix,
        binary_ari_matrix,
        cluster_bundle,
    ) = comparison_outputs(fixture_results)
    cluster_matrix = cluster_bundle[0]["matrix"]
    cluster_metadata = cluster_bundle[0]["metadata"]
    disagreement_rows = observation_disagreement_rows(fixture_results)
    temporal_rows = temporal_scope_rows(fixture_results)

    label_fields = [
        "trajectoryId",
        "observationId",
        "observationIndexOneBased",
        "configurationId",
        "familyId",
        "temporalScope",
        "evidenceClass",
        "labelStatus",
        "isReplicator",
        "clusterId",
        "componentId",
        "referenceObservationId",
        "metricToReference",
        "historicalIncomingH",
        "historicalLocalScore",
        "ineligibilityReason",
        "sourcePadding",
    ]
    write_csv(step_dir / "label_outputs.csv", output_rows, label_fields)
    (step_dir / "label_arrays.json").write_bytes(arrays_bytes)
    write_csv(
        step_dir / "continuous_recurrence.csv",
        recurrence,
        list(recurrence[0]),
    )
    overlap_fields = list(overlap_rows[0])
    write_csv(step_dir / "label_overlap_long.csv", overlap_rows, overlap_fields)
    write_csv(
        step_dir / "binary_jaccard_matrix.csv",
        jaccard_matrix,
        ["family", *REFERENCE_FAMILIES],
    )
    write_csv(
        step_dir / "binary_ari_matrix.csv",
        binary_ari_matrix,
        ["family", *REFERENCE_FAMILIES],
    )
    write_csv(
        step_dir / "cluster_ari_matrix.csv",
        cluster_matrix,
        ["family", *CLUSTER_FAMILIES],
    )
    write_csv(
        step_dir / "cluster_ari_denominators.csv",
        cluster_metadata,
        list(cluster_metadata[0]),
    )
    write_csv(
        step_dir / "run_level_disagreement.csv",
        run_disagreement,
        overlap_fields,
    )
    write_csv(
        step_dir / "disagreement_diagnostics.csv",
        disagreement_rows,
        list(disagreement_rows[0]),
    )
    write_csv(
        step_dir / "temporal_scope_diagnostics.csv",
        temporal_rows,
        list(temporal_rows[0]),
    )
    write_csv(
        step_dir / "threshold_sensitivity.csv",
        sensitivity_rows,
        list(sensitivity_rows[0]),
    )
    write_json(step_dir / "historical_technique2_diagnostics.json", technique2)
    write_json(
        step_dir / "fixture_catalog.json",
        {"researchStepId": "S08", "fixtures": fixtures},
    )
    write_json(step_dir / "seed_manifest.json", seed_manifest)

    schema = label_schema()
    write_json(shared_dir / "label_arrays_schema_v1.0.0.json", schema)
    write_yaml(
        shared_dir / "clustering_configurations_v1.0.1.yaml",
        {
            "schema": "eidosoma.e01.s08_materialized_label_configurations.v1",
            "researchStepId": "S08",
            "configurationCollectionVersion": config["configurationCollectionVersion"],
            "preregistrationSha256": sha256(CONFIG_PATH),
            "configurationCount": len(materialized_configurations),
            "configurations": materialized_configurations,
        },
    )
    write_yaml(
        shared_dir / "label_family_contract_v1.0.1.yaml",
        shared_contract(config, registry_before, materialized_configurations),
    )

    plot_disagreement(step_dir / "label_disagreement_map.png", fixture_results)
    plot_threshold(step_dir / "threshold_sensitivity.png", sensitivity_rows)
    plot_overlap(step_dir / "binary_label_overlap.png", jaccard_matrix)

    validation_checks: list[dict[str, Any]] = []
    validation_checks.extend(
        _check(item["valid"], f"FROZEN_INPUT_{item['evidenceId']}", item)
        for item in preregistration["frozenInputs"]
    )
    validation_checks.append(
        _check(
            preregistration["preregistrationCommitIsAncestor"],
            "PREREGISTRATION_COMMIT_ANCESTOR",
            preregistration["record"],
        )
    )
    validation_checks.extend(verify_upstream_fixture_views(config, fixtures))
    validation_checks.append(
        _check(
            fixture_replay_exact,
            "S06_DERIVED_FIXTURE_AND_PERMUTATION_REPLAY",
            {"fixtureCount": len(fixtures), "permutationCount": len(permutations)},
        )
    )
    validation_checks.append(
        _check(
            arrays_bytes == replay_bytes,
            "EXACT_LABEL_ARRAY_REGENERATION",
            {
                "payloadSha256": arrays_envelope["payloadSha256"],
                "serializedBytes": len(arrays_bytes),
            },
        )
    )
    validation_checks.extend(
        invariance_checks(
            fixtures,
            permutations,
            raw_configs,
            cluster_configs,
            fixture_results,
        )
    )
    edge_checks, edge_payload = edge_case_checks(fixtures, fixture_results, technique2)
    validation_checks.extend(edge_checks)
    write_json(step_dir / "edge_case_validation.json", edge_payload)

    total_observations = sum(len(fixture["states"]) for fixture in fixtures)
    reference_rows = [
        row
        for row in output_rows
        if row["configurationId"]
        in {raw_configs[family]["configurationId"] for family in REFERENCE_FAMILIES}
    ]
    validation_checks.append(
        _check(
            len(reference_rows) == total_observations * len(REFERENCE_FAMILIES)
            and len(output_rows) == total_observations * 7,
            "COMPLETE_LABEL_ROW_COVERAGE",
            {
                "observationCount": total_observations,
                "referenceRowCount": len(reference_rows),
                "expectedReferenceRowCount": total_observations * 4,
                "allCanonicalRowCount": len(output_rows),
                "expectedAllCanonicalRowCount": total_observations * 7,
            },
        )
    )
    row_keys = {
        (row["trajectoryId"], row["observationId"], row["configurationId"])
        for row in output_rows
    }
    validation_checks.append(
        _check(
            len(row_keys) == len(output_rows),
            "LABEL_ROW_IDENTITIES_UNIQUE",
            {"unique": len(row_keys), "rows": len(output_rows)},
        )
    )
    invalid_null_rows = [
        row
        for row in output_rows
        if row["isReplicator"] is None
        and not row["ineligibilityReason"]
        and row["labelStatus"] != "ERROR_SOURCE_DOMAIN"
    ]
    validation_checks.append(
        _check(
            not invalid_null_rows,
            "NULL_LABELS_HAVE_EXPLICIT_REASON",
            {"invalidNullRowCount": len(invalid_null_rows)},
        )
    )
    schema_errors = sorted(
        Draft202012Validator(schema).iter_errors(arrays_envelope),
        key=lambda error: tuple(str(item) for item in error.path),
    )
    length_errors = []
    for fixture in arrays_payload["fixtures"]:
        expected = fixture["observationCount"]
        for array in fixture["arrays"]:
            for field in (
                "observationIds",
                "labelStatuses",
                "isReplicator",
                "clusterIds",
                "componentIds",
                "ineligibilityReasons",
            ):
                if len(array[field]) != expected:
                    length_errors.append(
                        f"{fixture['fixtureId']}:{array['configurationId']}:{field}"
                    )
    round_trip = deserialize_envelope(arrays_bytes, require_canonical=True)
    validation_checks.append(
        _check(
            not schema_errors and not length_errors and round_trip == arrays_envelope,
            "LABEL_SCHEMA_CHECKSUM_AND_CANONICAL_ROUND_TRIP",
            {
                "schemaErrorCount": len(schema_errors),
                "lengthErrorCount": len(length_errors),
                "payloadSha256": arrays_envelope["payloadSha256"],
            },
        )
    )
    expected_sensitivity_rows = len(fixtures) * (
        sum(
            len(values)
            for values in (
                config["thresholdSensitivity"]["historicalHStrictGreaterThan"],
                config["thresholdSensitivity"]["cosineHStrictGreaterThan"],
                config["thresholdSensitivity"]["euclideanDistanceStrictLessThan"],
                config["thresholdSensitivity"]["aitchisonDistanceStrictLessThan"],
            )
        )
        + len(CLUSTER_FAMILIES)
        * len(config["commonClusteringContract"]["minimumClusterSizeSensitivity"])
    )
    validation_checks.append(
        _check(
            len(sensitivity_rows) == expected_sensitivity_rows,
            "FROZEN_SENSITIVITY_GRID_COMPLETE",
            {
                "actualRows": len(sensitivity_rows),
                "expectedRows": expected_sensitivity_rows,
            },
        )
    )
    validation_checks.append(
        _check(
            len(overlap_rows) == len(fixtures) * 16 + 16
            and len(run_disagreement) == len(fixtures) * 6,
            "OVERLAP_ARI_AND_DISAGREEMENT_DENOMINATORS_COMPLETE",
            {
                "overlapRows": len(overlap_rows),
                "runDisagreementRows": len(run_disagreement),
                "clusterAriDenominatorRows": len(cluster_metadata),
            },
        )
    )

    registry_after = registry_snapshot(config)
    registry_preserved = registry_before == registry_after
    registry_payload = {
        "schema": "eidosoma.e01.s08_registry_preservation.v1",
        "researchStepId": "S08",
        "success": registry_preserved,
        "before": registry_before,
        "after": registry_after,
        "action": "NO_REGISTRY_UPDATE_SOURCE_EVIDENCE_ABSENT",
        "validationConfigurationsResolveRegistrySentinels": False,
    }
    write_json(step_dir / "registry_preservation.json", registry_payload)
    validation_checks.append(
        _check(
            registry_preserved
            and len(registry_after["s08OwnerParameters"])
            == config["registryBoundary"]["ownerParameterCount"]
            and not registry_after["executable"]
            and registry_after["noSilentDefaults"],
            "REGISTRY_BYTE_AND_OWNER_SENTINEL_PRESERVATION",
            registry_after,
        )
    )
    validation_checks.append(
        _check(
            not (artifact_root / "research_steps/S09").exists(),
            "S09_NOT_BEGUN",
            {"s09Path": str(artifact_root / "research_steps/S09"), "exists": False},
        )
    )
    figure_paths = [
        step_dir / "label_disagreement_map.png",
        step_dir / "threshold_sensitivity.png",
        step_dir / "binary_label_overlap.png",
    ]
    validation_checks.append(
        _check(
            all(path.is_file() and path.stat().st_size > 0 for path in figure_paths),
            "DIAGNOSTIC_FIGURES_WRITTEN",
            {
                str(path): path.stat().st_size if path.exists() else None
                for path in figure_paths
            },
        )
    )

    pooled = {
        (row["familyA"], row["familyB"]): row
        for row in overlap_rows
        if row["fixtureId"] == "POOLED_ALL_FIXTURES"
    }
    pairwise_disagreements = [
        row for row in run_disagreement if row["familyA"] != row["familyB"]
    ]
    temporal_flip_total = sum(row["flipCount"] for row in temporal_rows)
    all_success = all(item["success"] for item in validation_checks)
    validation = {
        "schema": "eidosoma.e01.s08_validation_summary.v1",
        "researchStepId": "S08",
        "stepNumber": 8,
        "success": all_success,
        "status": "complete" if all_success else "validation_failed",
        "outcomeClassification": "supportive"
        if all_success
        else "constraining/contradictory",
        "configurationCollectionVersion": config["configurationCollectionVersion"],
        "fixtureCount": len(fixtures),
        "observationCount": total_observations,
        "referenceLabelRowCount": len(reference_rows),
        "canonicalLabelRowCount": len(output_rows),
        "materializedConfigurationCount": len(materialized_configurations),
        "thresholdSensitivityRowCount": len(sensitivity_rows),
        "validationCheckCount": len(validation_checks),
        "passedValidationCheckCount": sum(
            item["success"] for item in validation_checks
        ),
        "failedValidationCheckCount": sum(
            not item["success"] for item in validation_checks
        ),
        "checks": validation_checks,
        "anchorResults": {
            "pooledPairwiseJaccard": {
                f"{left}:{right}": pooled[(left, right)]["jaccard"]
                for index, left in enumerate(REFERENCE_FAMILIES)
                for right in REFERENCE_FAMILIES[index + 1 :]
            },
            "pooledPairwiseBinaryAdjustedRand": {
                f"{left}:{right}": pooled[(left, right)]["binaryAdjustedRandIndex"]
                for index, left in enumerate(REFERENCE_FAMILIES)
                for right in REFERENCE_FAMILIES[index + 1 :]
            },
            "runPairMaximumDisagreementRate": max(
                row["disagreementRate"]
                for row in pairwise_disagreements
                if row["disagreementRate"] is not None
            ),
            "retrospectivePastOnlyFlipTotal": temporal_flip_total,
            "aitchisonExplicitIneligibleRows": sum(
                row["familyId"] == "Y_A" and row["isReplicator"] is None
                for row in reference_rows
            ),
            "historicalTechnique2SourceDomainErrors": technique2["summary"][
                "sourceDomainErrorCount"
            ],
        },
        "caveatsOrBlockers": [
            "Author clustering code and thresholds remain unavailable.",
            "Euclidean and Aitchison reference anchors are validation-only configurations.",
            "Retrospective labels can use future observations and are not prospective outcomes.",
            "Aitchison zero support is explicitly deferred to S09; no replacement was applied.",
            "Historical technique 2 retains its pinned source-domain failure without repair.",
        ],
        "recommendedNextAction": "Hand control back; begin S09 only after separate authorization.",
    }
    write_json(step_dir / "validation_summary.json", validation)

    required = [
        step_dir / "label_outputs.csv",
        step_dir / "label_arrays.json",
        step_dir / "label_overlap_long.csv",
        step_dir / "binary_jaccard_matrix.csv",
        step_dir / "binary_ari_matrix.csv",
        step_dir / "cluster_ari_matrix.csv",
        step_dir / "run_level_disagreement.csv",
        step_dir / "threshold_sensitivity.csv",
        step_dir / "validation_summary.json",
        step_dir / "registry_preservation.json",
        shared_dir / "label_family_contract_v1.0.1.yaml",
        shared_dir / "clustering_configurations_v1.0.1.yaml",
        shared_dir / "label_arrays_schema_v1.0.0.json",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise RuntimeError("One or more required S08 artifacts were not written.")
    manifest = build_manifest(artifact_root, config, preregistration)
    write_json(step_dir / "artifact_manifest.json", manifest)
    if not all_success:
        failed = [item["checkId"] for item in validation_checks if not item["success"]]
        raise RuntimeError("S08 validation failed: " + ", ".join(failed))
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    validation = run(args.artifacts_dir.resolve())
    print(json.dumps(validation, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
