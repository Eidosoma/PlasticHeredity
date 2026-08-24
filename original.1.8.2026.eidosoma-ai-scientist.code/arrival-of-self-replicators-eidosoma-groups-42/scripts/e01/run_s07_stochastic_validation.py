#!/usr/bin/env python3
"""Run preregistered S07 stochastic validation and write compact evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from e01_gard_historical import (
    NumpyUniformSource,
    catalytic_matrix_from_numpy_rng_explicit,
    historical_single_event,
)
from e01_gard_historical import (
    compute_propensities as historical_propensities,
)
from e01_gard_historical import (
    split_fixed_size_without_replacement as historical_fission,
)
from e01_gard_independent import (
    RNGInput,
    calculate_propensities,
    fission,
    generate_catalytic_matrix,
    sample_update,
    specification_from_mapping,
)
from e01_gard_reproducibility import (
    CANONICAL_STREAM_PURPOSES,
    CouplingPolicy,
    SeedRequest,
    StreamPurpose,
    derive_seed_bundle,
    isolated_stream_namespace,
)
from e01_gard_validation.stochastic import (
    analytical_propensities,
    exact_multinomial_test,
    lognormal_log_moment_tests,
    pool_rare_categories,
    two_sample_target_tv_test,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/e01/s07_stochastic_validation_preregistration.yaml"
)
STEP_RELATIVE = Path("research_steps/S07")
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
ENGINE_LABELS = {
    "historical_reference": "e01_gard_historical_validation_harness@1.0.0",
    "independent": "e01_gard_independent@1.0.0",
    "statistical_test": "e01_s07_statistical_test@1.0.0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected object in {path}.")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("S07 preregistration must be a YAML object.")
    return payload


def git_output(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    return value


def tuple_outcome(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(tuple_outcome(item) for item in value)
    return value


def specification(config: dict[str, Any], profile_id: str):
    return specification_from_mapping(config["profiles"][profile_id])


def seed_bundle(config: dict[str, Any], identity: str, engine: str):
    specification_id = f"E01-S07-SEED-{identity}"
    trajectory_id = f"E01-S07-{identity}-R0"
    namespace = isolated_stream_namespace(
        experiment_id="E01",
        specification_id=specification_id,
        trajectory_id=trajectory_id,
        replicate_index=0,
    )
    request = SeedRequest(
        experiment_id="E01",
        specification_id=specification_id,
        trajectory_id=trajectory_id,
        replicate_index=0,
        engine_id=ENGINE_LABELS[engine],
        root_seed_hex=config["randomness"]["rootSeedHex"],
        coupling_policy=CouplingPolicy.TRAJECTORY_ISOLATED,
        coupling_reason=None,
        stream_namespaces={purpose: namespace for purpose in CANONICAL_STREAM_PURPOSES},
    )
    return derive_seed_bundle(request)


def verify_preregistration(artifact_root: Path) -> dict[str, Any]:
    step_dir = artifact_root / STEP_RELATIVE
    config = load_config()
    copy_path = step_dir / "preregistration.yaml"
    record_path = step_dir / "preregistration_record.json"
    tolerances_path = step_dir / "calibrated_tolerances.json"
    fixtures_path = step_dir / "validation_fixtures.json"
    missing = [
        str(path)
        for path in (copy_path, record_path, tolerances_path, fixtures_path)
        if not path.is_file()
    ]
    errors: list[str] = []
    if missing:
        errors.append("missing preregistration artifacts: " + ", ".join(missing))
        return {"valid": False, "errors": errors}
    record = load_json(record_path)
    tolerances = load_json(tolerances_path)
    fixtures = load_json(fixtures_path)
    hashes = {
        "sourceConfig": sha256(CONFIG_PATH),
        "artifactConfig": sha256(copy_path),
        "tolerances": sha256(tolerances_path),
        "fixtures": sha256(fixtures_path),
    }
    expected_hashes = {
        "sourceConfig": record["preregistrationSourceSha256"],
        "artifactConfig": record["preregistrationArtifactSha256"],
        "tolerances": record["calibratedTolerancesSha256"],
        "fixtures": record["validationFixturesSha256"],
    }
    if hashes != expected_hashes:
        errors.append(
            "preregistration or calibrated artifact hash changed after freeze"
        )
    if CONFIG_PATH.read_bytes() != copy_path.read_bytes():
        errors.append("repository and artifact preregistration bytes differ")
    if not record.get("canonicalOutcomeArtifactsAbsentAtFreeze"):
        errors.append("outcomes were not recorded absent at freeze")
    commit = record["preregistrationCommit"]
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=False,
        ).returncode
        == 0
    )
    if not ancestor:
        errors.append("preregistration commit is not an ancestor of HEAD")
    for evidence_id, evidence in config["frozenEvidence"].items():
        path = Path(evidence["path"])
        if not path.is_file() or sha256(path) != evidence["sha256"]:
            errors.append(f"frozen evidence mismatch: {evidence_id}")
    for name, required in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != required:
            errors.append(f"thread environment mismatch: {name}")
    if (artifact_root / "research_steps/S08").exists():
        errors.append("S08 artifact directory exists")
    configured_test_ids = [record["testId"] for record in config["primaryTests"]]
    tolerance_test_ids = [record["testId"] for record in tolerances["primaryTests"]]
    if configured_test_ids != tolerance_test_ids:
        errors.append("primary test registry differs from calibrated tolerances")
    return {
        "valid": not errors,
        "errors": errors,
        "config": config,
        "record": record,
        "tolerances": tolerances,
        "fixtures": fixtures,
        "hashes": hashes,
        "preregistrationCommitIsAncestor": ancestor,
    }


def _event_fixture(config: dict[str, Any], fixture_id: str) -> dict[str, Any]:
    return next(
        fixture
        for fixture in config["fixtures"]["eventSelection"]
        if fixture["fixtureId"] == fixture_id
    )


def _fission_fixture(config: dict[str, Any], fixture_id: str) -> dict[str, Any]:
    return next(
        fixture
        for fixture in config["fixtures"]["fission"]
        if fixture["fixtureId"] == fixture_id
    )


def run_event_task(fixture_id: str, engine: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config()
    fixture = _event_fixture(config, fixture_id)
    profile = specification(config, fixture["profileId"])
    state = tuple(int(value) for value in fixture["state"])
    beta = np.asarray(fixture["beta"], dtype=np.float64)
    draws = int(fixture["drawsPerEngine"])
    bundle = seed_bundle(config, f"{fixture_id}-{engine}", engine)
    generators = bundle.fresh_generators()
    counts = np.zeros(2 * profile.n_species, dtype=np.int64)
    waiting_bins = int(fixture.get("waitingTimeUniformBins", 0))
    waiting_counts = np.zeros(waiting_bins, dtype=np.int64)
    mass_failures = 0
    reconstruction_failures = 0
    nonnegative_failures = 0
    branch_failures = 0
    if engine == "historical_reference":
        source = NumpyUniformSource(generators[StreamPurpose.EVENT])
        for index in range(1, draws + 1):
            event = historical_single_event(
                state,
                beta=beta,
                rho=profile.rho,
                k_f=profile.k_f,
                k_b=profile.k_b,
                uniform_source=source,
                event_number=index,
            )
            counts[event.event_index_zero_based] += 1
            if event.mass_delta not in {-1, 1}:
                mass_failures += 1
            pre = np.asarray(event.pre_state, dtype=np.int64)
            post = np.asarray(event.post_state, dtype=np.int64)
            expected = pre.copy()
            species = event.species_index_one_based - 1
            expected[species] += 1 if event.kind == "join" else -1
            if not np.array_equal(expected, post):
                reconstruction_failures += 1
            if np.any(post < 0):
                nonnegative_failures += 1
            if source.compatibility_id != "NUMPY_GENERATOR_EXPLICIT_NOT_MATLAB_LEGACY":
                branch_failures += 1
    elif engine == "independent":
        streams = bundle.independent_engine_streams(generators)
        target_total = analytical_propensities(
            state,
            beta=beta,
            rho=profile.rho,
            k_f=profile.k_f,
            k_b=profile.k_b,
            orientation=profile.catalytic_matrix_branch.value,
        ).total
        for index in range(1, draws + 1):
            model_time = 0.0 if waiting_bins else None
            event = sample_update(
                state,
                beta=beta,
                specification=profile,
                rng_streams=streams,
                generation_index_one_based=1,
                step_index_one_based=index,
                model_time_before=model_time,
            )
            if event.selected_event_index_zero_based is None:
                branch_failures += 1
            else:
                counts[event.selected_event_index_zero_based] += 1
            if event.mass_delta not in {-1, 1}:
                mass_failures += 1
            pre = np.asarray(event.pre_state, dtype=np.int64)
            post = np.asarray(event.post_state, dtype=np.int64)
            expected = (
                pre
                + np.asarray(event.applied_join_counts)
                - np.asarray(event.applied_loss_counts)
            )
            if not np.array_equal(expected, post):
                reconstruction_failures += 1
            if np.any(post < 0):
                nonnegative_failures += 1
            if waiting_bins:
                if event.time_increment is None or event.time_increment < 0:
                    branch_failures += 1
                else:
                    transformed = 1.0 - np.exp(-target_total * event.time_increment)
                    bin_index = min(int(transformed * waiting_bins), waiting_bins - 1)
                    waiting_counts[bin_index] += 1
    else:
        raise ValueError(f"Unknown engine {engine}.")
    return {
        "taskType": "event",
        "fixtureId": fixture_id,
        "engine": engine,
        "draws": draws,
        "counts": counts.tolist(),
        "waitingCounts": waiting_counts.tolist(),
        "invariants": {
            "massFailures": mass_failures,
            "stateReconstructionFailures": reconstruction_failures,
            "nonnegativeFailures": nonnegative_failures,
            "branchIdentityFailures": branch_failures,
        },
        "seedPayload": bundle.to_payload(),
        "elapsedSeconds": time.perf_counter() - started,
    }


def run_matrix_task(engine: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config()
    fixture = config["fixtures"]["catalyticMatrixMoments"]
    profile = specification(config, fixture["profileId"])
    matrix_draws = int(fixture["matrixDrawCount"])
    expected_entries = int(fixture["totalEntriesPerEngine"])
    bundle = seed_bundle(config, f"S07-BETA-LOG-MOMENTS-{engine}", engine)
    generators = bundle.fresh_generators()
    values = np.empty(expected_entries, dtype=np.float64)
    offset = 0
    if engine == "historical_reference":
        generator = generators[StreamPurpose.CATALYTIC_MATRIX]
        for _ in range(matrix_draws):
            matrix = catalytic_matrix_from_numpy_rng_explicit(
                profile.n_species,
                a=profile.beta_a,
                sigma=profile.beta_sigma,
                generator=generator,
            )
            block = np.log(matrix).ravel()
            values[offset : offset + block.size] = block
            offset += block.size
    elif engine == "independent":
        rng = RNGInput(
            bundle.streams[StreamPurpose.CATALYTIC_MATRIX].stream_id,
            generators[StreamPurpose.CATALYTIC_MATRIX],
        )
        for _ in range(matrix_draws):
            matrix = generate_catalytic_matrix(profile, rng)
            block = np.log(matrix).ravel()
            values[offset : offset + block.size] = block
            offset += block.size
    else:
        raise ValueError(f"Unknown engine {engine}.")
    finite_failures = int(np.count_nonzero(~np.isfinite(values[:offset])))
    if offset != expected_entries:
        raise AssertionError(
            "Catalytic matrix entry count differs from preregistration."
        )
    quantile_probabilities = np.asarray(
        [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
    )
    return {
        "taskType": "matrix",
        "fixtureId": fixture["fixtureId"],
        "engine": engine,
        "sampleCount": expected_entries,
        "sampleMean": float(values.mean()),
        "sampleVariance": float(values.var(ddof=1)),
        "quantileProbabilities": quantile_probabilities.tolist(),
        "sampleQuantiles": np.quantile(values, quantile_probabilities).tolist(),
        "finiteFailures": finite_failures,
        "seedPayload": bundle.to_payload(),
        "elapsedSeconds": time.perf_counter() - started,
    }


def _serialize_counter(counter: Counter[Any]) -> list[dict[str, Any]]:
    return [
        {"outcome": to_builtin(outcome), "count": int(counter[outcome])}
        for outcome in sorted(counter, key=repr)
    ]


def run_fission_task(fixture_id: str, engine: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config()
    fixture = _fission_fixture(config, fixture_id)
    profile = specification(config, fixture["profileId"])
    parent = tuple(int(value) for value in fixture["parent"])
    draws = int(fixture["drawsPerEngine"])
    odd_fixed = fixture["targetLaw"].endswith("plus_uniform_discard")
    fixed = fixture["targetLaw"].startswith("fixed_size")
    bundle = seed_bundle(config, f"{fixture_id}-{engine}", engine)
    generators = bundle.fresh_generators()
    outcome_counts: Counter[Any] = Counter()
    daughter_counts: Counter[str] = Counter()
    conservation_failures = 0
    discard_failures = 0
    child_size_failures = 0
    daughter_stream_failures = 0
    if engine == "historical_reference":
        source = NumpyUniformSource(generators[StreamPurpose.FISSION])
        for _ in range(draws):
            result = historical_fission(parent, uniform_source=source)
            key = (result.child_a, result.discarded) if odd_fixed else result.child_a
            outcome_counts[key] += 1
            combined = (
                np.asarray(result.child_a)
                + np.asarray(result.child_b)
                + np.asarray(result.discarded)
            )
            if not np.array_equal(combined, parent):
                conservation_failures += 1
            expected_discard = 1 if sum(parent) % 2 else 0
            if sum(result.discarded) != expected_discard:
                discard_failures += 1
            if sum(result.child_a) != sum(result.child_b):
                child_size_failures += 1
            if (
                result.daughter_selection_rule
                != "FIRST_OUTPUT_CHILD_A_NO_ADDITIONAL_RANDOM_DRAW"
            ):
                daughter_stream_failures += 1
    elif engine == "independent":
        streams = bundle.independent_engine_streams(generators)
        for index in range(1, draws + 1):
            result = fission(
                parent,
                specification=profile,
                rng_streams=streams,
                generation_index_one_based=index,
            )
            key = (
                (result.child_first, result.discarded)
                if odd_fixed
                else result.child_first
            )
            outcome_counts[key] += 1
            daughter_counts[result.selected_daughter_label] += 1
            combined = (
                np.asarray(result.child_first)
                + np.asarray(result.child_second)
                + np.asarray(result.discarded)
            )
            if not np.array_equal(combined, parent):
                conservation_failures += 1
            expected_discard = 1 if fixed and sum(parent) % 2 else 0
            if sum(result.discarded) != expected_discard:
                discard_failures += 1
            if fixed and sum(result.child_first) != sum(result.child_second):
                child_size_failures += 1
            if profile.daughter_selection.value == "first":
                if result.daughter_rng_consumed:
                    daughter_stream_failures += 1
            elif not result.daughter_rng_consumed:
                daughter_stream_failures += 1
    else:
        raise ValueError(f"Unknown engine {engine}.")
    return {
        "taskType": "fission",
        "fixtureId": fixture_id,
        "engine": engine,
        "draws": draws,
        "outcomeCounts": _serialize_counter(outcome_counts),
        "daughterCounts": dict(sorted(daughter_counts.items())),
        "invariants": {
            "conservationFailures": conservation_failures,
            "discardFailures": discard_failures,
            "childSizeFailures": child_size_failures,
            "daughterStreamFailures": daughter_stream_failures,
        },
        "seedPayload": bundle.to_payload(),
        "elapsedSeconds": time.perf_counter() - started,
    }


def run_poisson_task() -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config()
    fixture = config["fixtures"]["paperPoisson"]
    profile = specification(config, fixture["profileId"])
    state = tuple(int(value) for value in fixture["state"])
    beta = np.asarray(fixture["beta"], dtype=np.float64)
    draws = int(fixture["draws"])
    bundle = seed_bundle(config, fixture["fixtureId"], "independent")
    generators = bundle.fresh_generators()
    streams = bundle.independent_engine_streams(generators)
    counters = [Counter() for _ in fixture["targetChannels"]]
    reconstruction_failures = 0
    applied_loss_failures = 0
    nonnegative_failures = 0
    clock_failures = 0
    branch_failures = 0
    non_unit_mass_changes = 0
    for index in range(1, draws + 1):
        event = sample_update(
            state,
            beta=beta,
            specification=profile,
            rng_streams=streams,
            generation_index_one_based=1,
            step_index_one_based=index,
            model_time_before=0.0,
        )
        attempted = (*event.attempted_join_counts, *event.attempted_loss_counts)
        for counter, value in zip(counters, attempted, strict=True):
            counter[int(value)] += 1
        pre = np.asarray(event.pre_state)
        post = np.asarray(event.post_state)
        applied_join = np.asarray(event.applied_join_counts)
        applied_loss = np.asarray(event.applied_loss_counts)
        if not np.array_equal(post, pre + applied_join - applied_loss):
            reconstruction_failures += 1
        if not np.array_equal(
            applied_loss, np.minimum(event.attempted_loss_counts, pre)
        ):
            applied_loss_failures += 1
        if np.any(post < 0):
            nonnegative_failures += 1
        if event.time_increment != profile.poisson_exposure:
            clock_failures += 1
        if event.update_kernel != "vector_poisson_batch":
            branch_failures += 1
        if event.mass_delta not in {-1, 1}:
            non_unit_mass_changes += 1
    return {
        "taskType": "poisson",
        "fixtureId": fixture["fixtureId"],
        "engine": "independent",
        "draws": draws,
        "channelCounters": {
            label: {str(key): int(value) for key, value in sorted(counter.items())}
            for label, counter in zip(fixture["targetChannels"], counters, strict=True)
        },
        "invariants": {
            "stateReconstructionFailures": reconstruction_failures,
            "appliedLossRuleFailures": applied_loss_failures,
            "nonnegativeFailures": nonnegative_failures,
            "clockFailures": clock_failures,
            "branchIdentityFailures": branch_failures,
            "nonUnitMassChangesObserved": non_unit_mass_changes,
        },
        "seedPayload": bundle.to_payload(),
        "elapsedSeconds": time.perf_counter() - started,
    }


def run_raw_task(task: tuple[str, ...]) -> dict[str, Any]:
    kind = task[0]
    if kind == "event":
        return run_event_task(task[1], task[2])
    if kind == "matrix":
        return run_matrix_task(task[1])
    if kind == "fission":
        return run_fission_task(task[1], task[2])
    if kind == "poisson":
        return run_poisson_task()
    raise ValueError(f"Unknown raw task {task!r}.")


def raw_tasks(config: dict[str, Any]) -> list[tuple[str, ...]]:
    tasks: list[tuple[str, ...]] = []
    for fixture in config["fixtures"]["eventSelection"]:
        tasks.extend(
            ("event", fixture["fixtureId"], engine) for engine in fixture["engines"]
        )
    tasks.extend(
        ("matrix", engine)
        for engine in config["fixtures"]["catalyticMatrixMoments"]["engines"]
    )
    for fixture in config["fixtures"]["fission"]:
        tasks.extend(
            ("fission", fixture["fixtureId"], engine) for engine in fixture["engines"]
        )
    tasks.append(("poisson",))
    return tasks


def execute_raw_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = raw_tasks(config)
    workers = int(config["execution"]["workerProcesses"])
    if workers < 1 or workers > int(config["execution"]["maximumWorkerProcesses"]):
        raise ValueError("Invalid S07 worker count.")
    if workers == 1:
        return [run_raw_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run_raw_task, tasks))


def raw_result(
    results: list[dict[str, Any]],
    *,
    task_type: str,
    fixture_id: str,
    engine: str,
) -> dict[str, Any]:
    matches = [
        result
        for result in results
        if result["taskType"] == task_type
        and result["fixtureId"] == fixture_id
        and result["engine"] == engine
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one result for {task_type}/{fixture_id}/{engine}, got {len(matches)}."
        )
    return matches[0]


def _target_counts_from_records(
    records: list[dict[str, Any]], outcomes: list[Any]
) -> list[int]:
    counter = {
        tuple_outcome(record["outcome"]): int(record["count"]) for record in records
    }
    target_outcomes = [tuple_outcome(outcome) for outcome in outcomes]
    unexpected = sorted(set(counter) - set(target_outcomes), key=repr)
    if unexpected:
        raise AssertionError(f"Unexpected observed outcome(s): {unexpected!r}")
    return [counter.get(outcome, 0) for outcome in target_outcomes]


def _poisson_binned_counts(counter: dict[str, int], labels: list[str]) -> list[int]:
    tail_start = int(labels[-1][2:])
    counts = np.zeros(len(labels), dtype=np.int64)
    for raw_value, raw_count in counter.items():
        value = int(raw_value)
        counts[min(value, tail_start)] += int(raw_count)
    return counts.tolist()


def test_rng(config: dict[str, Any], test_id: str):
    bundle = seed_bundle(config, f"INFERENCE-{test_id}", "statistical_test")
    generator = bundle.fresh_generators()[StreamPurpose.ESTIMATOR]
    return bundle, generator


def run_primary_tests(
    preregistration: dict[str, Any], raw_results: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    config = preregistration["config"]
    fixtures = preregistration["fixtures"]
    design = config["statisticalDesign"]
    alpha = float(design["perTestAlpha"])
    replicates = int(design["monteCarloReplicates"])
    batch_size = int(design["monteCarloBatchSize"])
    minimum_expected = float(design["minimumExpectedCountForAsymptoticDiagnostic"])
    registry = {record["subject"]: record for record in config["primaryTests"]}
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    inference_seeds: list[dict[str, Any]] = []

    def register_multinomial(
        *,
        subject: str,
        counts: list[int],
        probabilities: list[float],
        labels: list[str],
        engine: str,
        fixture_id: str,
    ) -> None:
        definition = registry[subject]
        bundle, generator = test_rng(config, definition["testId"])
        result = exact_multinomial_test(
            counts,
            probabilities,
            generator=generator,
            replicates=replicates,
            batch_size=batch_size,
        )
        pooling = pool_rare_categories(
            counts,
            probabilities,
            labels=labels,
            minimum_expected=minimum_expected,
        )
        passed = bool(result["pValue"] >= alpha)
        row = {
            "testId": definition["testId"],
            "family": definition["family"],
            "subject": subject,
            "testType": definition["type"],
            "engine": engine,
            "fixtureId": fixture_id,
            "sampleSizeLeft": int(sum(counts)),
            "sampleSizeRight": "",
            "supportSize": len(probabilities),
            "statistic": result["statistic"],
            "pValue": result["pValue"],
            "perTestAlpha": alpha,
            "rareCategoryCount": len(pooling["rareIndices"]),
            "asymptoticDiagnosticEligible": pooling["asymptoticPearsonEligible"],
            "method": design["multinomialMethod"],
            "passed": passed,
        }
        rows.append(row)
        details[definition["testId"]] = {
            **row,
            "labels": labels,
            "counts": counts,
            "probabilities": probabilities,
            "observedProbabilities": [value / sum(counts) for value in counts],
            "testResult": result,
            "rareCategoryDiagnostic": pooling,
            "inferenceSeedStream": bundle.streams[StreamPurpose.ESTIMATOR].to_payload(),
        }
        inference_seeds.append(
            {"testId": definition["testId"], "seedPayload": bundle.to_payload()}
        )

    def register_tv(
        *,
        subject: str,
        left_counts: list[int],
        right_counts: list[int],
        probabilities: list[float],
        labels: list[str],
        fixture_id: str,
    ) -> None:
        definition = registry[subject]
        bundle, generator = test_rng(config, definition["testId"])
        result = two_sample_target_tv_test(
            left_counts,
            right_counts,
            probabilities,
            generator=generator,
            replicates=replicates,
            batch_size=batch_size,
        )
        passed = bool(result["pValue"] >= alpha)
        row = {
            "testId": definition["testId"],
            "family": definition["family"],
            "subject": subject,
            "testType": definition["type"],
            "engine": "historical_reference_vs_independent",
            "fixtureId": fixture_id,
            "sampleSizeLeft": int(sum(left_counts)),
            "sampleSizeRight": int(sum(right_counts)),
            "supportSize": len(probabilities),
            "statistic": result["statistic"],
            "pValue": result["pValue"],
            "perTestAlpha": alpha,
            "rareCategoryCount": int(
                np.count_nonzero(
                    np.asarray(probabilities) * min(sum(left_counts), sum(right_counts))
                    < minimum_expected
                )
            ),
            "asymptoticDiagnosticEligible": False,
            "method": design["crossEngineMethod"],
            "passed": passed,
        }
        rows.append(row)
        details[definition["testId"]] = {
            **row,
            "labels": labels,
            "leftCounts": left_counts,
            "rightCounts": right_counts,
            "probabilities": probabilities,
            "testResult": result,
            "inferenceSeedStream": bundle.streams[StreamPurpose.ESTIMATOR].to_payload(),
        }
        inference_seeds.append(
            {"testId": definition["testId"], "seedPayload": bundle.to_payload()}
        )

    for fixture_id, target in fixtures["eventSelection"].items():
        engine_counts: dict[str, list[int]] = {}
        for engine in target["engines"]:
            result = raw_result(
                raw_results,
                task_type="event",
                fixture_id=fixture_id,
                engine=engine,
            )
            engine_counts[engine] = result["counts"]
            subject = f"{fixture_id}/{engine}"
            register_multinomial(
                subject=subject,
                counts=result["counts"],
                probabilities=target["probabilities"],
                labels=target["labels"],
                engine=engine,
                fixture_id=fixture_id,
            )
        if set(target["engines"]) == {"historical_reference", "independent"}:
            register_tv(
                subject=fixture_id,
                left_counts=engine_counts["historical_reference"],
                right_counts=engine_counts["independent"],
                probabilities=target["probabilities"],
                labels=target["labels"],
                fixture_id=fixture_id,
            )

    matrix_fixture = fixtures["catalyticMatrixMoments"]
    moment_rows: list[dict[str, Any]] = []
    for engine in matrix_fixture["engines"]:
        result = raw_result(
            raw_results,
            task_type="matrix",
            fixture_id=matrix_fixture["fixtureId"],
            engine=engine,
        )
        moment_results = lognormal_log_moment_tests(
            sample_count=result["sampleCount"],
            sample_mean=result["sampleMean"],
            sample_variance=result["sampleVariance"],
            expected_mean=matrix_fixture["expectedLogMean"],
            expected_variance=matrix_fixture["expectedLogVariance"],
        )
        for moment_name, suffix in (("mean", "log_mean"), ("variance", "log_variance")):
            subject = f"{engine}/{suffix}"
            definition = registry[subject]
            test_result = moment_results[moment_name]
            passed = bool(test_result["pValue"] >= alpha)
            row = {
                "testId": definition["testId"],
                "family": definition["family"],
                "subject": subject,
                "testType": definition["type"],
                "engine": engine,
                "fixtureId": matrix_fixture["fixtureId"],
                "sampleSizeLeft": result["sampleCount"],
                "sampleSizeRight": "",
                "supportSize": "",
                "statistic": test_result["statistic"],
                "pValue": test_result["pValue"],
                "perTestAlpha": alpha,
                "rareCategoryCount": 0,
                "asymptoticDiagnosticEligible": "not_applicable",
                "method": config["statisticalDesign"]["momentMethod"],
                "passed": passed,
            }
            rows.append(row)
            detail = {
                **row,
                "sampleMean": result["sampleMean"],
                "sampleVariance": result["sampleVariance"],
                "expectedMean": matrix_fixture["expectedLogMean"],
                "expectedVariance": matrix_fixture["expectedLogVariance"],
                "acceptanceIntervals": matrix_fixture["bonferroniAcceptanceIntervals"],
                "sampleQuantiles": result["sampleQuantiles"],
                "quantileProbabilities": result["quantileProbabilities"],
            }
            details[definition["testId"]] = detail
            moment_rows.append(detail)

    for fixture_id, target in fixtures["fission"].items():
        engine_counts: dict[str, list[int]] = {}
        for engine in target["engines"]:
            result = raw_result(
                raw_results,
                task_type="fission",
                fixture_id=fixture_id,
                engine=engine,
            )
            counts = _target_counts_from_records(
                result["outcomeCounts"], target["outcomes"]
            )
            engine_counts[engine] = counts
            subject = f"{fixture_id}/{engine}"
            register_multinomial(
                subject=subject,
                counts=counts,
                probabilities=target["probabilities"],
                labels=target["labels"],
                engine=engine,
                fixture_id=fixture_id,
            )
        if set(target["engines"]) == {"historical_reference", "independent"}:
            register_tv(
                subject=fixture_id,
                left_counts=engine_counts["historical_reference"],
                right_counts=engine_counts["independent"],
                probabilities=target["probabilities"],
                labels=target["labels"],
                fixture_id=fixture_id,
            )
        if "daughterSelectionTarget" in target:
            result = raw_result(
                raw_results,
                task_type="fission",
                fixture_id=fixture_id,
                engine="independent",
            )
            daughter_counts = [
                int(result["daughterCounts"].get("first", 0)),
                int(result["daughterCounts"].get("second", 0)),
            ]
            register_multinomial(
                subject=f"{fixture_id}/daughter_selection",
                counts=daughter_counts,
                probabilities=target["daughterSelectionTarget"],
                labels=["first", "second"],
                engine="independent",
                fixture_id=fixture_id,
            )

    poisson_target = fixtures["paperPoisson"]
    poisson_result = raw_result(
        raw_results,
        task_type="poisson",
        fixture_id=poisson_target["fixtureId"],
        engine="independent",
    )
    for channel, target in poisson_target["channels"].items():
        counts = _poisson_binned_counts(
            poisson_result["channelCounters"][channel], target["labels"]
        )
        register_multinomial(
            subject=f"{poisson_target['fixtureId']}/{channel}",
            counts=counts,
            probabilities=target["probabilities"],
            labels=target["labels"],
            engine="independent",
            fixture_id=poisson_target["fixtureId"],
        )

    modern_target = fixtures["eventSelection"]["S07-EVENT-MODERN"]
    modern_result = raw_result(
        raw_results,
        task_type="event",
        fixture_id="S07-EVENT-MODERN",
        engine="independent",
    )
    waiting_probabilities = modern_target["waitingTimeTarget"]["probabilities"]
    register_multinomial(
        subject="S07-EVENT-MODERN/exponential_pit",
        counts=modern_result["waitingCounts"],
        probabilities=waiting_probabilities,
        labels=[f"PIT_bin_{index + 1}" for index in range(len(waiting_probabilities))],
        engine="independent",
        fixture_id="S07-EVENT-MODERN",
    )

    expected_order = [record["testId"] for record in config["primaryTests"]]
    rows_by_id = {row["testId"]: row for row in rows}
    if set(rows_by_id) != set(expected_order) or len(rows) != len(expected_order):
        raise AssertionError(
            f"Primary result set mismatch; observed={sorted(rows_by_id)}, expected={expected_order}."
        )
    rows = [rows_by_id[test_id] for test_id in expected_order]
    return rows, details, inference_seeds


def run_invariant_checks(
    preregistration: dict[str, Any], raw_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    config = preregistration["config"]
    checks: list[dict[str, Any]] = []

    def add(
        check_id: str,
        subject: str,
        observed: Any,
        expected: Any,
        passed: bool,
        detail: str,
    ) -> None:
        checks.append(
            {
                "checkId": check_id,
                "subject": subject,
                "observed": observed,
                "expected": expected,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    check_number = 1
    for result in raw_results:
        if result["taskType"] == "event":
            for name, observed in result["invariants"].items():
                add(
                    f"S07-I{check_number:02d}",
                    f"{result['fixtureId']}/{result['engine']}/{name}",
                    observed,
                    0,
                    observed == 0,
                    "Categorical/direct-Gillespie event invariant requires zero failures.",
                )
                check_number += 1
        elif result["taskType"] == "matrix":
            add(
                f"S07-I{check_number:02d}",
                f"{result['fixtureId']}/{result['engine']}/finite_log_beta",
                result["finiteFailures"],
                0,
                result["finiteFailures"] == 0,
                "Every generated beta entry must be positive with finite logarithm.",
            )
            check_number += 1
        elif result["taskType"] == "fission":
            for name, observed in result["invariants"].items():
                add(
                    f"S07-I{check_number:02d}",
                    f"{result['fixtureId']}/{result['engine']}/{name}",
                    observed,
                    0,
                    observed == 0,
                    "Fission conservation, size/discard, and RNG-consumption invariant.",
                )
                check_number += 1
        elif result["taskType"] == "poisson":
            for name, observed in result["invariants"].items():
                if name == "nonUnitMassChangesObserved":
                    passed = observed > 0
                    expected = ">0"
                    detail = (
                        "Vector-Poisson batches must remain distinguishable from "
                        "single-event +/-1 branches."
                    )
                else:
                    passed = observed == 0
                    expected = 0
                    detail = "Vector-Poisson explicit branch invariant requires zero failures."
                add(
                    f"S07-I{check_number:02d}",
                    f"{result['fixtureId']}/independent/{name}",
                    observed,
                    expected,
                    passed,
                    detail,
                )
                check_number += 1

    profile_mapping = dict(config["profiles"]["E01-S07-MODERN-GILLESPIE-v1.0.0"])
    beta = np.asarray(
        [[1.0, 0.2, 0.1], [0.5, 0.3, 0.4], [0.2, 0.8, 0.1]],
        dtype=np.float64,
    )
    state = (2, 1, 1)
    for branch in (
        "historical_orientation_with_diagonal",
        "transposed_with_diagonal",
        "historical_orientation_zero_diagonal",
    ):
        current = dict(profile_mapping)
        current["specification_id"] = f"E01-S07-BRANCH-CHECK-{branch}"
        current["catalytic_matrix_branch"] = branch
        explicit_specification = specification_from_mapping(current)
        engine = calculate_propensities(
            state, beta=beta, specification=explicit_specification
        )
        target = analytical_propensities(
            state,
            beta=beta,
            rho=explicit_specification.rho,
            k_f=explicit_specification.k_f,
            k_b=explicit_specification.k_b,
            orientation=branch,
        )
        maximum_error = float(
            np.max(
                np.abs(
                    np.asarray(engine.concatenated) - np.asarray(target.concatenated)
                )
            )
        )
        add(
            f"S07-I{check_number:02d}",
            f"independent/catalytic_branch/{branch}",
            maximum_error,
            0.0,
            maximum_error == 0.0,
            "Every explicit orientation/diagonal branch must match the independent equation oracle.",
        )
        check_number += 1

    common = _event_fixture(config, "S07-EVENT-COMMON")
    common_profile = specification(config, common["profileId"])
    historical = historical_propensities(
        common["state"],
        beta=common["beta"],
        rho=common_profile.rho,
        k_f=common_profile.k_f,
        k_b=common_profile.k_b,
    )
    target = analytical_propensities(
        common["state"],
        beta=common["beta"],
        rho=common_profile.rho,
        k_f=common_profile.k_f,
        k_b=common_profile.k_b,
    )
    historical_error = float(
        np.max(
            np.abs(
                historical.concatenated
                - np.asarray(target.concatenated, dtype=np.float64)
            )
        )
    )
    add(
        f"S07-I{check_number:02d}",
        "historical_reference/propensity_equation_oracle",
        historical_error,
        0.0,
        historical_error == 0.0,
        "Historical source-traced propensities must match the independent equation oracle.",
    )
    check_number += 1

    all_seed_ids: list[str] = []
    all_seed_material: list[str] = []
    for result in raw_results:
        streams = result["seedPayload"]["streams"]
        all_seed_ids.extend(record["streamId"] for record in streams.values())
        all_seed_material.extend(
            record["seedMaterialHex"] for record in streams.values()
        )
    expected_streams = len(raw_results) * len(CANONICAL_STREAM_PURPOSES)
    seed_uniqueness = (
        len(all_seed_ids) == expected_streams
        and len(set(all_seed_ids)) == expected_streams
        and len(set(all_seed_material)) == expected_streams
    )
    add(
        f"S07-I{check_number:02d}",
        "S06_domain_separated_raw_task_streams",
        len(set(all_seed_ids)),
        expected_streams,
        seed_uniqueness,
        "Every raw validation task must have nine distinct S06-derived stream identities.",
    )
    check_number += 1

    add(
        f"S07-I{check_number:02d}",
        "historical_rng_identity_boundary",
        config["scopeBoundary"]["historicalHarnessIdentity"],
        "NUMPY_GENERATOR_EXPLICIT_NOT_MATLAB_LEGACY",
        config["scopeBoundary"]["historicalHarnessIdentity"]
        == "NUMPY_GENERATOR_EXPLICIT_NOT_MATLAB_LEGACY",
        "Historical NumPy draws are distribution-validation harnesses, not MATLAB identities.",
    )
    return checks


def _shift_counts(counts: list[int], fraction: float) -> list[int]:
    shifted = list(counts)
    source = int(np.argmax(shifted))
    destination = (source + 1) % len(shifted)
    amount = max(1, round(fraction * sum(shifted)))
    amount = min(amount, shifted[source])
    shifted[source] -= amount
    shifted[destination] += amount
    return shifted


def run_failure_injection(
    preregistration: dict[str, Any],
    raw_results: list[dict[str, Any]],
    primary_details: dict[str, Any],
) -> dict[str, Any]:
    config = preregistration["config"]
    design = config["statisticalDesign"]
    alpha = float(design["perTestAlpha"])
    replicates = int(design["monteCarloReplicates"])
    batch_size = int(design["monteCarloBatchSize"])
    cases: list[dict[str, Any]] = []

    def injection_multinomial(
        injection_id: str, source_test_id: str, fraction: float
    ) -> None:
        source = primary_details[source_test_id]
        injected_counts = _shift_counts(source["counts"], fraction)
        bundle, generator = test_rng(config, f"{injection_id}-DETECTOR")
        result = exact_multinomial_test(
            injected_counts,
            source["probabilities"],
            generator=generator,
            replicates=replicates,
            batch_size=batch_size,
        )
        cases.append(
            {
                "injectionId": injection_id,
                "sourceTestId": source_test_id,
                "fault": f"move_{fraction:.0%}_from_modal_to_next_category",
                "detector": "exact_multinomial_test",
                "pValue": result["pValue"],
                "threshold": alpha,
                "detected": bool(result["pValue"] < alpha),
                "inferenceSeedStream": bundle.streams[
                    StreamPurpose.ESTIMATOR
                ].to_payload(),
            }
        )

    injection_multinomial("FI01", "S07-T01", 0.03)

    matrix_result = raw_result(
        raw_results,
        task_type="matrix",
        fixture_id="S07-BETA-LOG-MOMENTS",
        engine="historical_reference",
    )
    shifted_moments = lognormal_log_moment_tests(
        sample_count=matrix_result["sampleCount"],
        sample_mean=matrix_result["sampleMean"] + 0.05 * 4.0,
        sample_variance=matrix_result["sampleVariance"],
        expected_mean=-4.0,
        expected_variance=16.0,
    )
    cases.append(
        {
            "injectionId": "FI02",
            "fault": "add_0.05_sigma_to_each_log_beta",
            "detector": "exact_normal_mean_test",
            "pValue": shifted_moments["mean"]["pValue"],
            "threshold": alpha,
            "detected": bool(shifted_moments["mean"]["pValue"] < alpha),
        }
    )

    categorical_failure_count = 1
    cases.append(
        {
            "injectionId": "FI03",
            "fault": "add_two_to_recorded_post_mass",
            "detector": "categorical_mass_delta_exact_invariant",
            "observedFailures": categorical_failure_count,
            "allowedFailures": 0,
            "detected": categorical_failure_count > 0,
        }
    )

    injection_multinomial("FI04", "S07-T12", 0.03)

    parent = np.asarray([2, 2, 2], dtype=np.int64)
    child_first = np.asarray([2, 1, 0], dtype=np.int64)
    child_second = np.asarray([0, 1, 2], dtype=np.int64)
    injected_child = child_first.copy()
    injected_child[0] += 1
    conservation_detected = not np.array_equal(injected_child + child_second, parent)
    cases.append(
        {
            "injectionId": "FI05",
            "fault": "add_one_unmatched_molecule_to_child_first",
            "detector": "fission_conservation_exact_invariant",
            "detected": conservation_detected,
        }
    )

    injection_multinomial("FI06", "S07-T20", 0.03)

    reported_kernel = "categorical_single_event"
    actual_kernel = "vector_poisson_batch"
    cases.append(
        {
            "injectionId": "FI07",
            "fault": "relabel_vector_poisson_batch_as_categorical_single_event",
            "detector": "branch_identity_contract",
            "reportedKernel": reported_kernel,
            "actualKernel": actual_kernel,
            "detected": reported_kernel != actual_kernel,
        }
    )
    required = int(config["failureInjection"]["requiredDetectedCount"])
    detected = sum(bool(case["detected"]) for case in cases)
    return {
        "schema": "eidosoma.e01.s07_failure_injection.v1",
        "researchStepId": "S07",
        "requiredDetectedCount": required,
        "detectedCount": detected,
        "success": detected == required == len(cases),
        "cases": cases,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_events(step_dir: Path, details: dict[str, Any]) -> None:
    test_ids = ["S07-T01", "S07-T02", "S07-T03", "S07-T04", "S07-T05"]
    figure, axes = plt.subplots(
        len(test_ids), 1, figsize=(10, 13), constrained_layout=True
    )
    for axis, test_id in zip(axes, test_ids, strict=True):
        record = details[test_id]
        counts = np.asarray(record["counts"], dtype=np.float64)
        probabilities = np.asarray(record["probabilities"], dtype=np.float64)
        observed = counts / counts.sum()
        standard_error = np.sqrt(
            np.maximum(probabilities * (1.0 - probabilities) / counts.sum(), 1e-30)
        )
        residuals = (observed - probabilities) / standard_error
        positions = np.arange(len(counts))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.bar(positions, residuals, color="#4472c4")
        axis.set_xticks(positions, record["labels"], rotation=30, ha="right")
        axis.set_ylabel("std. residual")
        axis.set_title(
            f"{test_id}: {record['subject']} (exact MC p={record['pValue']:.4g})"
        )
    figure.suptitle("S07 event-frequency diagnostics against a_k/a_0")
    figure.savefig(step_dir / "diagnostic_event_probabilities.png", dpi=180)
    plt.close(figure)


def plot_beta(step_dir: Path, details: dict[str, Any], alpha: float) -> None:
    test_ids = ["S07-T08", "S07-T09", "S07-T10", "S07-T11"]
    labels = [details[test_id]["subject"] for test_id in test_ids]
    p_values = np.asarray([details[test_id]["pValue"] for test_id in test_ids])
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    positions = np.arange(len(test_ids))
    axis.bar(positions, -np.log10(np.maximum(p_values, 1e-300)), color="#70ad47")
    axis.axhline(
        -np.log10(alpha), color="#c00000", linestyle="--", label="Bonferroni gate"
    )
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("-log10(p)")
    axis.set_title("S07 exact log-beta mean and variance tests")
    axis.legend()
    figure.savefig(step_dir / "diagnostic_beta_moments.png", dpi=180)
    plt.close(figure)


def plot_fission(step_dir: Path, details: dict[str, Any]) -> None:
    panels = [
        ("S07-T12", "S07-T13", "fixed even"),
        ("S07-T14", "S07-T15", "fixed odd + discard"),
        (None, "S07-T16", "binomial complement"),
    ]
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    for axis, (historical_id, independent_id, title) in zip(axes, panels, strict=True):
        independent = details[independent_id]
        probabilities = np.asarray(independent["probabilities"], dtype=np.float64)
        top = np.argsort(probabilities)[::-1][: min(12, probabilities.size)]
        positions = np.arange(top.size)
        width = 0.25 if historical_id else 0.36
        axis.bar(
            positions - width,
            probabilities[top],
            width=width,
            label="analytical",
            color="#a5a5a5",
        )
        if historical_id:
            historical = details[historical_id]
            historical_probabilities = np.asarray(historical["counts"]) / sum(
                historical["counts"]
            )
            axis.bar(
                positions,
                historical_probabilities[top],
                width=width,
                label="historical harness",
                color="#ed7d31",
            )
            independent_offset = width
        else:
            independent_offset = 0.0
        independent_probabilities = np.asarray(independent["counts"]) / sum(
            independent["counts"]
        )
        axis.bar(
            positions + independent_offset,
            independent_probabilities[top],
            width=width,
            label="independent",
            color="#5b9bd5",
        )
        axis.set_xticks(
            positions,
            [independent["labels"][index] for index in top],
            rotation=35,
            ha="right",
        )
        axis.set_ylabel("probability")
        axis.set_title(title)
        axis.legend()
    figure.suptitle("S07 fission-distribution diagnostics")
    figure.savefig(step_dir / "diagnostic_fission_probabilities.png", dpi=180)
    plt.close(figure)


def plot_independent_only(step_dir: Path, details: dict[str, Any]) -> None:
    poisson_ids = [f"S07-T{index}" for index in range(20, 26)]
    channel_labels = [
        details[test_id]["subject"].split("/")[-1] for test_id in poisson_ids
    ]
    observed_means: list[float] = []
    target_means: list[float] = []
    for test_id in poisson_ids:
        record = details[test_id]
        labels = record["labels"]
        counts = np.asarray(record["counts"], dtype=np.float64)
        probabilities = np.asarray(record["probabilities"], dtype=np.float64)
        tail_start = int(labels[-1][2:])
        representatives = np.asarray([*range(tail_start), tail_start], dtype=np.float64)
        observed_means.append(float(np.sum(representatives * counts) / counts.sum()))
        target_means.append(float(np.sum(representatives * probabilities)))
    waiting = details["S07-T26"]
    waiting_observed = np.asarray(waiting["counts"]) / sum(waiting["counts"])
    waiting_target = np.asarray(waiting["probabilities"])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    positions = np.arange(len(poisson_ids))
    width = 0.38
    axes[0].bar(positions - width / 2, target_means, width, label="target")
    axes[0].bar(positions + width / 2, observed_means, width, label="observed")
    axes[0].set_xticks(positions, channel_labels, rotation=30, ha="right")
    axes[0].set_ylabel("tail-binned count mean")
    axes[0].set_title("Vector-Poisson attempted-count marginals")
    axes[0].legend()
    waiting_positions = np.arange(waiting_observed.size)
    axes[1].plot(waiting_positions, waiting_target, "o-", label="uniform target")
    axes[1].plot(waiting_positions, waiting_observed, "s-", label="observed PIT")
    axes[1].set_xlabel("PIT bin")
    axes[1].set_ylabel("probability")
    axes[1].set_title("Direct-Gillespie exponential waiting-time PIT")
    axes[1].legend()
    figure.suptitle("S07 independent-only stochastic branches")
    figure.savefig(step_dir / "diagnostic_independent_only_branches.png", dpi=180)
    plt.close(figure)


def registry_preservation(config: dict[str, Any]) -> dict[str, Any]:
    registry_record = config["frozenEvidence"]["specificationRegistry"]
    registry_path = Path(registry_record["path"])
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    parameters = payload["parameters"]
    unresolved = sum(bool(record.get("unresolved")) for record in parameters)
    branch_sets = sum(
        isinstance(record.get("value"), str)
        and record["value"].startswith("BRANCH_SET::")
        for record in parameters
    )
    actual_hash = sha256(registry_path)
    valid = (
        actual_hash == registry_record["sha256"]
        and len(parameters) == 120
        and unresolved == 64
        and branch_sets == 21
        and payload["executionGate"]["executable"] is False
        and payload["executionGate"]["noSilentDefaults"] is True
    )
    return {
        "schema": "eidosoma.e01.s07_registry_preservation.v1",
        "researchStepId": "S07",
        "path": str(registry_path),
        "expectedSha256": registry_record["sha256"],
        "actualSha256": actual_hash,
        "unchanged": actual_hash == registry_record["sha256"],
        "parameterCount": len(parameters),
        "unresolvedParameterCount": unresolved,
        "unexpandedBranchSetCount": branch_sets,
        "executable": payload["executionGate"]["executable"],
        "noSilentDefaults": payload["executionGate"]["noSilentDefaults"],
        "s07RegistryUpdates": [],
        "valid": valid,
    }


def artifact_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "role": role,
        "sizeBytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def manifest_inputs() -> list[Path]:
    attachment_root = (
        WORKSPACE_ROOT / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759"
    )
    paths = [
        WORKSPACE_ROOT / "AGENTS.md",
        WORKSPACE_ROOT / "FULL_PLAN.md",
        WORKSPACE_ROOT / "RESEARCH_PLAN.md",
        WORKSPACE_ROOT / "input-attachments/MANIFEST.json",
        attachment_root / "_metadata/ATTACHMENT.md",
        attachment_root / "pdf-markdown.md",
        Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf"),
        Path(
            "/artifacts/E01_forensic_replication_bundle/specifications/"
            "specification_registry_v0.3.0.yaml"
        ),
        Path(
            "/artifacts/E01_forensic_replication_bundle/provenance/source_manifest.yaml"
        ),
        Path(
            "/artifacts/E01_forensic_replication_bundle/provenance/environment_report.json"
        ),
        Path(
            "/artifacts/E01_forensic_replication_bundle/provenance/precision_policy.yaml"
        ),
        Path(
            "/artifacts/E01_forensic_replication_bundle/software/historical_reference/"
            "historical_behavior_contract.yaml"
        ),
        Path(
            "/artifacts/E01_forensic_replication_bundle/software/independent_engine/"
            "independent_engine_contract.yaml"
        ),
        Path(
            "/artifacts/E01_forensic_replication_bundle/software/independent_engine/"
            "validation_profiles.yaml"
        ),
        Path(
            "/artifacts/E01_forensic_replication_bundle/reproducibility/"
            "seed_derivation_contract_v1.0.0.yaml"
        ),
        Path(
            "/artifacts/E01_forensic_replication_bundle/reproducibility/"
            "seed_schema_v1.0.0.json"
        ),
        Path(
            "/artifacts/E01_forensic_replication_bundle/reproducibility/"
            "trajectory_precision_contract_v1.0.0.yaml"
        ),
        Path(
            "/artifacts/E01_forensic_replication_bundle/reproducibility/"
            "trajectory_schema_v1.0.0.json"
        ),
    ]
    for index in range(1, 7):
        paths.extend(
            [
                Path(
                    f"/artifacts/research_steps/S{index:02d}/research_step_full_results.md"
                ),
                Path(f"/artifacts/research_steps/S{index:02d}/artifact_manifest.json"),
            ]
        )
    return paths


def repository_code_paths() -> list[Path]:
    return [
        CONFIG_PATH,
        REPOSITORY_ROOT / "scripts/e01/freeze_s07_preregistration.py",
        REPOSITORY_ROOT / "scripts/e01/run_s07_stochastic_validation.py",
        REPOSITORY_ROOT / "src/e01_gard_validation/__init__.py",
        REPOSITORY_ROOT / "src/e01_gard_validation/stochastic.py",
        REPOSITORY_ROOT / "tests/e01/test_stochastic_validation.py",
        REPOSITORY_ROOT / "src/e01_gard_historical/engine.py",
        REPOSITORY_ROOT / "src/e01_gard_independent/engine.py",
        REPOSITORY_ROOT / "src/e01_gard_reproducibility/seed.py",
    ]


def build_manifest(artifact_root: Path) -> dict[str, Any]:
    step_dir = artifact_root / STEP_RELATIVE
    output_paths = sorted(
        path
        for path in step_dir.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    inputs = manifest_inputs()
    code = repository_code_paths()
    missing = [str(path) for path in (*inputs, *code) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Manifest inputs missing: " + ", ".join(missing))
    return {
        "schema": "eidosoma.e01.s07_artifact_manifest.v1",
        "researchStepId": "S07",
        "artifactRoot": str(artifact_root),
        "repository": str(REPOSITORY_ROOT),
        "repositoryBranch": git_output("branch", "--show-current"),
        "repositoryCommit": git_output("rev-parse", "HEAD"),
        "preregistrationCommit": load_json(step_dir / "preregistration_record.json")[
            "preregistrationCommit"
        ],
        "inputs": [artifact_record(path, "input") for path in inputs],
        "repositoryCode": [artifact_record(path, "repository_code") for path in code],
        "outputs": [artifact_record(path, "output") for path in output_paths],
        "selfHashExcluded": True,
    }


def output_paths(step_dir: Path) -> list[str]:
    return [
        str(step_dir / name)
        for name in (
            "preregistration.yaml",
            "preregistration_record.json",
            "calibrated_tolerances.json",
            "validation_fixtures.json",
            "seed_manifest.json",
            "goodness_of_fit_summary.csv",
            "goodness_of_fit_details.json",
            "moment_tests.csv",
            "invariant_checks.csv",
            "failure_injection.json",
            "diagnostic_event_probabilities.png",
            "diagnostic_beta_moments.png",
            "diagnostic_fission_probabilities.png",
            "diagnostic_independent_only_branches.png",
            "registry_preservation.json",
            "validation_summary.json",
            "artifact_manifest.json",
            "research_step_full_results.md",
        )
    ]


def build(artifact_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    preregistration = verify_preregistration(artifact_root)
    if not preregistration["valid"]:
        raise RuntimeError(
            "S07 preregistration verification failed: "
            + "; ".join(preregistration["errors"])
        )
    config = preregistration["config"]
    step_dir = artifact_root / STEP_RELATIVE
    result_only = [
        "seed_manifest.json",
        "goodness_of_fit_summary.csv",
        "goodness_of_fit_details.json",
        "moment_tests.csv",
        "invariant_checks.csv",
        "failure_injection.json",
        "registry_preservation.json",
        "validation_summary.json",
    ]
    preexisting = [name for name in result_only if (step_dir / name).exists()]
    if preexisting:
        raise RuntimeError(
            "Canonical S07 outcome run is single-shot; existing results found: "
            + ", ".join(preexisting)
        )
    raw_results = execute_raw_tasks(config)
    primary_rows, primary_details, inference_seeds = run_primary_tests(
        preregistration, raw_results
    )
    invariant_rows = run_invariant_checks(preregistration, raw_results)
    injection = run_failure_injection(preregistration, raw_results, primary_details)
    registry = registry_preservation(config)
    alpha = float(config["statisticalDesign"]["perTestAlpha"])
    primary_passed = sum(bool(row["passed"]) for row in primary_rows)
    invariants_passed = sum(bool(row["passed"]) for row in invariant_rows)
    success = bool(
        primary_passed == len(primary_rows)
        and invariants_passed == len(invariant_rows)
        and injection["success"]
        and registry["valid"]
        and preregistration["valid"]
        and not (artifact_root / "research_steps/S08").exists()
    )

    write_csv(
        step_dir / "goodness_of_fit_summary.csv",
        primary_rows,
        [
            "testId",
            "family",
            "subject",
            "testType",
            "engine",
            "fixtureId",
            "sampleSizeLeft",
            "sampleSizeRight",
            "supportSize",
            "statistic",
            "pValue",
            "perTestAlpha",
            "rareCategoryCount",
            "asymptoticDiagnosticEligible",
            "method",
            "passed",
        ],
    )
    write_json(
        step_dir / "goodness_of_fit_details.json",
        {
            "schema": "eidosoma.e01.s07_goodness_of_fit.v1",
            "researchStepId": "S07",
            "globalFamilywiseAlpha": config["statisticalDesign"][
                "globalFamilywiseAlpha"
            ],
            "perTestAlpha": alpha,
            "primaryTestCount": len(primary_rows),
            "passedCount": primary_passed,
            "allPassed": primary_passed == len(primary_rows),
            "tests": primary_details,
            "rawTaskPerformance": [
                {
                    "taskType": result["taskType"],
                    "fixtureId": result["fixtureId"],
                    "engine": result["engine"],
                    "drawsOrSamples": result.get("draws", result.get("sampleCount")),
                    "elapsedSeconds": result["elapsedSeconds"],
                }
                for result in raw_results
            ],
        },
    )
    moment_rows = [
        primary_details[test_id]
        for test_id in ("S07-T08", "S07-T09", "S07-T10", "S07-T11")
    ]
    write_csv(
        step_dir / "moment_tests.csv",
        moment_rows,
        [
            "testId",
            "engine",
            "subject",
            "sampleSizeLeft",
            "sampleMean",
            "sampleVariance",
            "expectedMean",
            "expectedVariance",
            "statistic",
            "pValue",
            "perTestAlpha",
            "passed",
        ],
    )
    write_csv(
        step_dir / "invariant_checks.csv",
        invariant_rows,
        ["checkId", "subject", "observed", "expected", "passed", "detail"],
    )
    write_json(step_dir / "failure_injection.json", injection)
    write_json(step_dir / "registry_preservation.json", registry)
    write_json(
        step_dir / "seed_manifest.json",
        {
            "schema": "eidosoma.e01.s07_seed_manifest.v1",
            "researchStepId": "S07",
            "seedContractVersion": config["randomness"]["seedSchemaVersion"],
            "derivationAlgorithm": config["randomness"]["derivationAlgorithm"],
            "couplingPolicy": config["randomness"]["couplingPolicy"],
            "historicalBoundary": config["scopeBoundary"]["historicalHarnessIdentity"],
            "rawTaskSeeds": [
                {
                    "taskType": result["taskType"],
                    "fixtureId": result["fixtureId"],
                    "engine": result["engine"],
                    "seedPayload": result["seedPayload"],
                }
                for result in raw_results
            ],
            "inferenceTestSeeds": inference_seeds,
        },
    )
    plot_events(step_dir, primary_details)
    plot_beta(step_dir, primary_details, alpha)
    plot_fission(step_dir, primary_details)
    plot_independent_only(step_dir, primary_details)

    p_values = [float(row["pValue"]) for row in primary_rows]
    rare_tests = [
        row["testId"] for row in primary_rows if int(row["rareCategoryCount"]) > 0
    ]
    caveats = [
        "Historical NumPy draws are an explicit distribution-validation harness, not legacy MATLAB RNG identity.",
        "The unavailable author implementation is not represented by either engine.",
        "Paper vector-Poisson exposure/clipping/boundary semantics remain an explicit fixture branch, not an author default.",
        "Registry v0.3.0 remains non-executable and all conflicts, sentinels, and branch sets are preserved.",
        "Exact Monte Carlo p-values have finite preregistered resolution; rare categories never use asymptotic gates.",
    ]
    validation_summary = {
        "schema": "eidosoma.e01.s07_validation_summary.v1",
        "researchStepId": "S07",
        "stepNumber": 7,
        "success": success,
        "status": "complete" if success else "complete_with_validation_failure",
        "outcomeClassification": "supportive"
        if success
        else "constraining/contradictory",
        "artifactsWritten": output_paths(step_dir),
        "validationResult": (
            f"PASS: {primary_passed}/{len(primary_rows)} calibrated primary tests, "
            f"{invariants_passed}/{len(invariant_rows)} deterministic invariants, "
            f"and {injection['detectedCount']}/{injection['requiredDetectedCount']} "
            "failure injections passed."
            if success
            else (
                f"FAIL: {primary_passed}/{len(primary_rows)} primary tests, "
                f"{invariants_passed}/{len(invariant_rows)} invariants, and "
                f"{injection['detectedCount']}/{injection['requiredDetectedCount']} "
                "failure injections passed."
            )
        ),
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": (
            "Return control to the Chief Scientist. Do not begin S08 in this step."
            if success
            else (
                "Return control for review of failed S07 gates; do not begin S08 until the discrepancy is adjudicated."
            )
        ),
        "checks": {
            "preregistrationVerified": preregistration["valid"],
            "primaryTestCount": len(primary_rows),
            "primaryPassedCount": primary_passed,
            "globalFamilywiseAlpha": config["statisticalDesign"][
                "globalFamilywiseAlpha"
            ],
            "perTestAlpha": alpha,
            "minimumPrimaryPValue": min(p_values),
            "rareCategoryExactTestIds": rare_tests,
            "invariantCheckCount": len(invariant_rows),
            "invariantPassedCount": invariants_passed,
            "failureInjectionDetectedCount": injection["detectedCount"],
            "registryPreserved": registry["valid"],
            "s08ArtifactsAbsent": not (artifact_root / "research_steps/S08").exists(),
        },
        "errors": [row["testId"] for row in primary_rows if not row["passed"]]
        + [row["checkId"] for row in invariant_rows if not row["passed"]]
        + [case["injectionId"] for case in injection["cases"] if not case["detected"]],
        "warnings": [],
        "elapsedSeconds": time.perf_counter() - started,
    }
    write_json(step_dir / "validation_summary.json", validation_summary)
    write_json(step_dir / "artifact_manifest.json", build_manifest(artifact_root))
    return validation_summary


def finalize_manifest(artifact_root: Path) -> dict[str, Any]:
    step_dir = artifact_root / STEP_RELATIVE
    summary = load_json(step_dir / "validation_summary.json")
    expected = [Path(path) for path in summary["artifactsWritten"]]
    missing = [
        str(path)
        for path in expected
        if path.name != "artifact_manifest.json" and not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Required S07 output missing: " + ", ".join(missing))
    if (artifact_root / "research_steps/S08").exists():
        raise RuntimeError(
            "S08 artifact directory exists; S07 scope validation failed."
        )
    manifest = build_manifest(artifact_root)
    write_json(step_dir / "artifact_manifest.json", manifest)
    return {
        "success": True,
        "outputCount": len(manifest["outputs"]),
        "repositoryCommit": manifest["repositoryCommit"],
        "manifestPath": str(step_dir / "artifact_manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--finalize-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifacts_dir.resolve()
    if args.finalize_manifest:
        result = finalize_manifest(artifact_root)
    else:
        result = build(artifact_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
