"""Registration, execution, reporting, status, and verification workflows."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import io
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .cohorts import allocate_stage1, allocate_stage2, construct_fresh_pair_pool
from .contract import (
    DEFAULT_ARTIFACTS,
    FIXED_PRIMARY,
    SCHEMA_VERSION,
    STAGE1_CONDITIONS,
    STAGE1_CONTRACT,
    STAGE1_NAMESPACE,
    STAGE1_PROFILE,
    STAGE2_CONTRACT,
    STAGE2_NAMESPACE,
    STAGE2_PROFILE,
    ReaderConfiguration,
    as_jsonable_configuration,
    atomic_write_json,
    atomic_write_text,
    implementation_manifest,
    load_json,
    parse_historical_pair_id,
    read_checkpoint,
    reader_configurations,
    seal_registration,
    sha256_file,
    sha256_json,
    verify_registration,
    write_checkpoint,
)
from .engine import (
    CarrierPair,
    corrupt_carrier_signs,
    decode_state_hex,
    deterministic_board,
    encode_state_hex,
    parent_statistics,
    permute_carrier,
    pooled_reference,
    simulate_daughter,
    texture2x2,
    transform_board,
    transform_carrier,
    write_spatial_latch,
    write_carrier,
)
from .inference import (
    AssignmentAccumulator,
    adjudicate_stage1,
    adjudicate_stage2,
    aggregate_assignment,
)
from .snapshot import verify_snapshot


def _artifact_root(value: Path | None) -> Path:
    return (value or DEFAULT_ARTIFACTS).resolve()


def _input_context(artifacts: Path) -> dict[str, Any]:
    input_root = artifacts / "input"
    manifest = verify_snapshot(input_root)
    donors = load_json(input_root / "DONORS.json")["donors"]
    return {
        "manifest": manifest,
        "donors": donors,
        "donor_index": {donor["donor_id"]: donor for donor in donors},
        "hypothesis": load_json(input_root / "HYPOTHESIS.json"),
        "launch_resets": load_json(input_root / "LAUNCH_RESETS.json"),
        "legacy": load_json(input_root / "LEGACY.json"),
        "exclusions": load_json(input_root / "HISTORICAL_PAIR_EXCLUSIONS.json"),
    }


def _pair_pool(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    return construct_fresh_pair_pool(
        context["donors"], context["exclusions"]["pair_ids"]
    )


def validate_cleanroom(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    context = _input_context(artifacts)
    pool = _pair_pool(context)
    historical_ids = context["exclusions"]["pair_ids"]
    historical_donors = {
        donor
        for pair_id in historical_ids
        for donor in parse_historical_pair_id(pair_id)
    }
    fresh_donors = {
        donor
        for pair in pool
        for donor in (pair["a_donor_id"], pair["b_donor_id"])
    }
    if historical_donors & fresh_donors:
        raise AssertionError("historically exposed donor leaked into fresh pair pool")
    allocations = allocate_stage1(pool, STAGE1_NAMESPACE)
    used = [pair["pair_id"] for cohort in allocations.values() for pair in cohort]
    stage2_preview = allocate_stage2(pool, used, STAGE2_NAMESPACE)
    report = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "snapshot_digest": context["manifest"]["snapshot_digest"],
        "frozen_donors": len(context["donors"]),
        "historical_pair_exclusions": len(historical_ids),
        "historical_donor_exclusions": len(historical_donors),
        "fresh_nonreusing_pairs": len(pool),
        "stage1_counts": {key: len(value) for key, value in allocations.items()},
        "stage2_capacity": {key: len(value) for key, value in stage2_preview.items()},
        "density_caliper_max": max(pair["density_difference"] for pair in pool),
        "source_runtime_firewall": "scientific modules consume artifacts/input only",
        "new_experiments_added": False,
    }
    atomic_write_json(artifacts / "VALIDATION.json", report)
    return report


def _configuration_from_dict(value: Mapping[str, Any]) -> ReaderConfiguration:
    return ReaderConfiguration(
        family=str(value["family"]),
        write_window=int(value["write_window"]),
        strength=float(value["strength"]),
        read_duration=int(value["read_duration"]),
    )


def parity_check(artifacts_root: Path | None = None) -> dict[str, Any]:
    """Recalculate retained deterministic fixtures; never create evidence."""

    artifacts = _artifact_root(artifacts_root)
    context = _input_context(artifacts)
    donors = context["donors"]
    donor_index = context["donor_index"]
    legacy = context["legacy"]
    roundtrip_failures = 0
    anchor_error = 0.0
    for donor in donors:
        for key in (
            "initial_state_hex",
            "ancestor_state_hex",
            "anchor_state_hex",
            "donor_state_hex",
            "offspring_state_hex",
        ):
            value = donor[key]
            roundtrip_failures += int(encode_state_hex(decode_state_hex(value)) != value)
        observed = texture2x2(decode_state_hex(donor["anchor_state_hex"]))
        anchor_error = max(
            anchor_error,
            float(np.max(np.abs(observed - np.asarray(donor["anchor_terminal2x2"])))),
        )

    calibration_ids = legacy["stage1"]["cohorts"]["cohorts"]["calibration"]
    reference_errors: dict[str, dict[str, float]] = {}
    references: dict[int, dict[str, np.ndarray]] = {}
    for window in (16, 32):
        histories = []
        for pair_id in calibration_ids:
            for donor_id in parse_historical_pair_id(pair_id):
                histories.append(
                    parent_statistics(
                        decode_state_hex(donor_index[donor_id]["donor_state_hex"]), window
                    )
                )
        calculated = pooled_reference(histories)
        references[window] = calculated
        expected = legacy["stage1"]["calibration"]["reference"][str(window)]
        reference_errors[str(window)] = {
            "motif_probability": float(
                np.max(
                    np.abs(
                        calculated["motif_probability"]
                        - np.asarray(expected["motif_probability"])
                    )
                )
            ),
            "context_probability": float(
                np.max(
                    np.abs(
                        calculated["context_probability"]
                        - np.asarray(expected["context_probability"])
                    )
                )
            ),
        }

    fixture = legacy["parity_fixture"]
    left_id, right_id = parse_historical_pair_id(fixture["pair_id"])
    carrier_errors: dict[str, float] = {}
    for configuration_id, expected_mean in fixture["carrier_mean_abs"].items():
        config_data = legacy["stage1"]["design"]["configurations"]
        config = _configuration_from_dict(
            next(item for item in config_data if item["configuration_id"] == configuration_id)
        )
        carriers = []
        for donor_id in (left_id, right_id):
            statistics = parent_statistics(
                decode_state_hex(donor_index[donor_id]["donor_state_hex"]),
                config.write_window,
            )
            carriers.append(
                write_carrier(statistics, references[config.write_window], config.family)
            )
        observed_mean = float(np.mean([np.mean(np.abs(carrier)) for carrier in carriers]))
        carrier_errors[configuration_id] = abs(observed_mean - float(expected_mean))

    stage1_decision = legacy["stage1"]["stage_decision"]
    stage2_decision = legacy["stage2"]["stage_decision"]
    passed = bool(
        roundtrip_failures == 0
        and anchor_error <= 1e-12
        and max(
            error
            for window in reference_errors.values()
            for error in window.values()
        )
        <= 1e-12
        and max(carrier_errors.values()) <= 1e-6
        and stage1_decision["verdict"] == "ROBUST_LOCAL_MOTIF_CONTROLLABILITY"
        and stage2_decision["verdict"]
        == "DENSITY_ROBUST_GENERAL_MOTIF_CHANNEL"
        and stage1_decision["selected_stage2_input"]["configuration_id"]
        == FIXED_PRIMARY.configuration_id
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "evidential_status": "NON_EVIDENTIAL_IMPLEMENTATION_PARITY",
        "passed": passed,
        "state_hex_roundtrip_failures": roundtrip_failures,
        "anchor_terminal2x2_max_abs_error": anchor_error,
        "calibration_reference_max_abs_errors": reference_errors,
        "writer_carrier_mean_abs_errors": carrier_errors,
        "legacy_stage1_verdict": stage1_decision["verdict"],
        "legacy_stage2_verdict": stage2_decision["verdict"],
        "reader_trajectory_parity": (
            "not claimed: retained artifacts do not contain raw daughter states or a "
            "complete semantic-RNG specification"
        ),
    }
    atomic_write_json(artifacts / "parity" / "REPORT.json", report)
    if not passed:
        raise AssertionError("deterministic parity checks failed")
    return report


def register_stage1(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    validation = validate_cleanroom(artifacts)
    context = _input_context(artifacts)
    pool = _pair_pool(context)
    cohorts = allocate_stage1(pool, STAGE1_NAMESPACE)
    legacy_nominees = [
        {
            **nominee["configuration"],
            "selected_checkpoint": int(nominee["selected_checkpoint"]),
        }
        for nominee in context["legacy"]["stage1"]["selection"]["nominees"]
    ]
    if FIXED_PRIMARY.configuration_id not in {
        nominee["configuration_id"] for nominee in legacy_nominees
    }:
        raise ValueError("legacy fixed primary is absent from retained nominee data")
    pair_pool_document = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_digest": context["manifest"]["snapshot_digest"],
        "pairing_policy": (
            "outcome-blind same-launch A/B greedy minimum-density-distance pairing; "
            "0.02 caliper; deterministic SHA-256 tie breaks; no donor reuse"
        ),
        "pairs": pool,
    }
    pair_pool_path = artifacts / "cohorts" / "FRESH_PAIR_POOL.json"
    if pair_pool_path.exists():
        if load_json(pair_pool_path) != pair_pool_document:
            raise ValueError("existing fresh pair pool differs from deterministic reconstruction")
    else:
        atomic_write_json(pair_pool_path, pair_pool_document)
    registration = seal_registration(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": 1,
            "state": "REGISTERED_NO_FRESH_OUTCOMES",
            "namespace": STAGE1_NAMESPACE,
            "snapshot_digest": context["manifest"]["snapshot_digest"],
            "pair_pool_sha256": sha256_file(pair_pool_path),
            "implementation_manifest": implementation_manifest(),
            "profile": STAGE1_PROFILE,
            "contract": STAGE1_CONTRACT,
            "conditions": STAGE1_CONDITIONS,
            "all_configurations": [
                as_jsonable_configuration(config) for config in reader_configurations()
            ],
            "confirmatory_primary": as_jsonable_configuration(FIXED_PRIMARY),
            "validation_nominees": legacy_nominees,
            "selection_policy": (
                "the retained Stage-1 winner is the sole confirmatory primary; "
                "fresh discovery cannot substitute another configuration"
            ),
            "cohorts": cohorts,
            "historical_pair_exclusion_count": len(context["exclusions"]["pair_ids"]),
            "fresh_pair_pool_count": len(pool),
            "stage2_reserve_verified": validation["stage2_capacity"],
            "automatic_stage2_launch": False,
            "added_experiments": [],
        }
    )
    path = artifacts / "stage1" / "REGISTRATION.json"
    if path.exists():
        existing = load_json(path)
        verify_registration(existing)
        if existing != registration:
            raise ValueError("Stage-1 registration already exists with a different design")
        return existing
    atomic_write_json(path, registration)
    _write_status(artifacts)
    return registration


def _load_registration(artifacts: Path, stage: int) -> dict[str, Any]:
    path = artifacts / f"stage{stage}" / "REGISTRATION.json"
    registration = load_json(path)
    verify_registration(registration)
    if registration["implementation_manifest"] != implementation_manifest():
        raise ValueError(
            f"Stage-{stage} implementation changed after registration; fresh outcomes blocked"
        )
    context = _input_context(artifacts)
    if registration["snapshot_digest"] != context["manifest"]["snapshot_digest"]:
        raise ValueError(f"Stage-{stage} snapshot binding mismatch")
    return registration


def _reference_to_json(reference: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: np.asarray(value).tolist() for key, value in reference.items()}


def run_stage1_calibration(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    registration = _load_registration(artifacts, 1)
    path = artifacts / "stage1" / "CALIBRATION.json"
    if path.exists():
        return read_checkpoint(path, registration["design_digest"])
    context = _input_context(artifacts)
    donor_index = context["donor_index"]
    references: dict[str, Any] = {}
    for window in registration["profile"]["write_windows"]:
        histories = []
        for pair in registration["cohorts"]["calibration"]:
            for key in ("a_donor_id", "b_donor_id"):
                histories.append(
                    parent_statistics(
                        decode_state_hex(donor_index[pair[key]]["donor_state_hex"]),
                        int(window),
                    )
                )
        references[str(window)] = _reference_to_json(pooled_reference(histories))
    payload = {
        "phase": "label_blind_calibration",
        "pair_count": len(registration["cohorts"]["calibration"]),
        "label_access": False,
        "reference": references,
    }
    write_checkpoint(path, registration["design_digest"], payload)
    _write_status(artifacts)
    return payload


def _load_reference(artifacts: Path, registration: Mapping[str, Any]) -> dict[int, Any]:
    payload = read_checkpoint(
        artifacts / "stage1" / "CALIBRATION.json", registration["design_digest"]
    )
    return {
        int(window): {
            key: np.asarray(values, dtype=np.float64) for key, values in reference.items()
        }
        for window, reference in payload["reference"].items()
    }


def _carrier_pair(
    pair: Mapping[str, Any],
    donor_index: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, np.ndarray],
    family: str,
    window: int,
) -> CarrierPair:
    carriers = []
    for donor_id in (pair["a_donor_id"], pair["b_donor_id"]):
        statistics = parent_statistics(
            decode_state_hex(donor_index[donor_id]["donor_state_hex"]), window
        )
        carriers.append(write_carrier(statistics, reference, family))
    return CarrierPair(carriers[0], carriers[1])


def _donor_subset(
    donor_index: Mapping[str, Mapping[str, Any]], *pairs: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    donor_ids = {
        pair[key]
        for pair in pairs
        for key in ("a_donor_id", "b_donor_id")
    }
    return {donor_id: donor_index[donor_id] for donor_id in donor_ids}


def _target_arrays(hypothesis: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    targets = hypothesis["targets"]
    primary = {label: np.asarray(targets["primary"][label]) for label in ("A", "B")}
    terminal_source = targets.get("primary_terminal", targets["primary"])
    terminal = {label: np.asarray(terminal_source[label]) for label in ("A", "B")}
    return primary, terminal


def _evaluate_condition(
    resets: Mapping[str, np.ndarray],
    carriers: CarrierPair,
    configuration: ReaderConfiguration,
    targets_primary: Mapping[str, np.ndarray],
    targets_terminal: Mapping[str, np.ndarray],
    namespace: str,
    seed_prefix: tuple[object, ...],
    replicates: int,
    *,
    process_noise: float = 0.0,
    read_enabled: bool = True,
    spatial_latches: Mapping[str, np.ndarray] | None = None,
    spatial_latch_strength: float | None = None,
    observation_transform: str = "identity",
    collect_diagnostics: bool = False,
) -> dict[str, Any]:
    checkpoints = tuple(int(value) for value in STAGE1_CONTRACT["checkpoints"])
    primary = {checkpoint: AssignmentAccumulator() for checkpoint in checkpoints}
    terminal = {checkpoint: AssignmentAccumulator() for checkpoint in checkpoints}
    survival = {checkpoint: 0 for checkpoint in checkpoints}
    diagnostic_values: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"A": [], "B": []}
        for name in ("occupancy", "components", "autocorrelation", "low_frequency_power")
    }
    for replicate in range(replicates):
        for label, carrier in (("A", carriers.a), ("B", carriers.b)):
            trajectory = simulate_daughter(
                resets[label],
                carrier,
                configuration,
                namespace,
                (*seed_prefix, replicate),
                process_noise=process_noise,
                read_enabled=read_enabled,
                spatial_latch=None if spatial_latches is None else spatial_latches[label],
                spatial_latch_strength=spatial_latch_strength,
                observation_transform=observation_transform,
                collect_diagnostics=collect_diagnostics,
            )
            for checkpoint, observation in trajectory.items():
                from .engine import assign_form

                primary[checkpoint].add(
                    label, assign_form(observation["primary"], targets_primary)
                )
                terminal[checkpoint].add(
                    label, assign_form(observation["terminal"], targets_terminal)
                )
                survival[checkpoint] += int(observation["alive"])
                if "diagnostics" in observation:
                    for name, value in observation["diagnostics"].items():
                        diagnostic_values[name][label].append(
                            np.atleast_1d(np.asarray(value, dtype=np.float64))
                        )
    result = {
        "checkpoints": {
            str(checkpoint): {
                "primary": primary[checkpoint].finish(),
                "terminal": terminal[checkpoint].finish(),
                "survival": survival[checkpoint] / (2.0 * replicates),
            }
            for checkpoint in checkpoints
        },
        "replicates": replicates,
        "reset_asserted_identical": bool(np.array_equal(resets["A"], resets["B"])),
    }
    if collect_diagnostics:
        result["diagnostics"] = {
            name: {
                "a_mean": np.mean(values["A"], axis=0).tolist(),
                "b_mean": np.mean(values["B"], axis=0).tolist(),
                "b_minus_a": (
                    np.mean(values["B"], axis=0) - np.mean(values["A"], axis=0)
                ).tolist(),
            }
            for name, values in diagnostic_values.items()
        }
    return result


def _screen_worker(argument: Mapping[str, Any]) -> dict[str, Any]:
    pair = argument["pair"]
    donor_index = argument["donor_index"]
    references = {
        int(window): {key: np.asarray(value) for key, value in table.items()}
        for window, table in argument["references"].items()
    }
    primary = {key: np.asarray(value) for key, value in argument["targets_primary"].items()}
    terminal = {key: np.asarray(value) for key, value in argument["targets_terminal"].items()}
    reset = deterministic_board(STAGE1_NAMESPACE, pair["pair_id"], "neutral-reset")
    results: dict[str, Any] = {}
    carrier_cache: dict[tuple[str, int], CarrierPair] = {}
    for config_data in argument["configurations"]:
        config = _configuration_from_dict(config_data)
        cache_key = (config.family, config.write_window)
        if cache_key not in carrier_cache:
            carrier_cache[cache_key] = _carrier_pair(
                pair,
                donor_index,
                references[config.write_window],
                config.family,
                config.write_window,
            )
        results[config.configuration_id] = {
            "configuration": as_jsonable_configuration(config),
            "conditions": {
                "intact": _evaluate_condition(
                    {"A": reset, "B": reset},
                    carrier_cache[cache_key],
                    config,
                    primary,
                    terminal,
                    STAGE1_NAMESPACE,
                    (pair["pair_id"], "screen", "native"),
                    int(argument["replicates"]),
                )
            },
        }
    return {
        "phase": "screen",
        "pair_id": pair["pair_id"],
        "configurations": results,
    }


def _validation_worker(argument: Mapping[str, Any]) -> dict[str, Any]:
    pair = argument["pair"]
    unrelated = argument["unrelated"]
    donor_index = argument["donor_index"]
    references = {
        int(window): {key: np.asarray(value) for key, value in table.items()}
        for window, table in argument["references"].items()
    }
    primary = {key: np.asarray(value) for key, value in argument["targets_primary"].items()}
    terminal = {key: np.asarray(value) for key, value in argument["targets_terminal"].items()}
    latch_config = argument["spatial_latch_benchmark"]
    reset = deterministic_board(STAGE1_NAMESPACE, pair["pair_id"], "neutral-reset")
    result: dict[str, Any] = {}
    for config_data in argument["configurations"]:
        config = _configuration_from_dict(config_data)
        intact = _carrier_pair(
            pair,
            donor_index,
            references[config.write_window],
            config.family,
            config.write_window,
        )
        unrelated_carriers = _carrier_pair(
            unrelated,
            donor_index,
            references[config.write_window],
            config.family,
            config.write_window,
        )
        zero = CarrierPair(np.zeros_like(intact.a), np.zeros_like(intact.b))
        shuffled = CarrierPair(
            permute_carrier(intact.a, STAGE1_NAMESPACE, pair["pair_id"], "shuffle"),
            permute_carrier(intact.b, STAGE1_NAMESPACE, pair["pair_id"], "shuffle"),
        )
        opposite = CarrierPair(intact.b, intact.a)
        corrupted = CarrierPair(
            corrupt_carrier_signs(
                intact.a, STAGE1_CONTRACT["carrier_corruption"], STAGE1_NAMESPACE,
                pair["pair_id"], "corruption"
            ),
            corrupt_carrier_signs(
                intact.b, STAGE1_CONTRACT["carrier_corruption"], STAGE1_NAMESPACE,
                pair["pair_id"], "corruption"
            ),
        )
        parent_boards = {
            "A": decode_state_hex(donor_index[pair["a_donor_id"]]["donor_state_hex"]),
            "B": decode_state_hex(donor_index[pair["b_donor_id"]]["donor_state_hex"]),
        }
        incomplete = {}
        for label in ("A", "B"):
            board = reset.copy().reshape(-1)
            board[:64] = parent_boards[label].reshape(-1)[:64]
            incomplete[label] = board.reshape(reset.shape)
        common = {
            "configuration": as_jsonable_configuration(config),
            "conditions": {},
        }
        condition_specs = {
            "intact": (
                {"A": reset, "B": reset}, intact, {"collect_diagnostics": True}
            ),
            "zero": ({"A": reset, "B": reset}, zero, {}),
            "read_disabled": (
                {"A": reset, "B": reset}, intact, {"read_enabled": False}
            ),
            "shuffle": ({"A": reset, "B": reset}, shuffled, {}),
            "opposite_history": ({"A": reset, "B": reset}, opposite, {}),
            "unrelated_same_form": (
                {"A": reset, "B": reset}, unrelated_carriers, {}
            ),
            "process_noise": (
                {"A": reset, "B": reset},
                intact,
                {"process_noise": STAGE1_CONTRACT["process_noise"]},
            ),
            "carrier_sign_corruption": (
                {"A": reset, "B": reset}, corrupted, {}
            ),
            "spatial_latch_benchmark": (
                {"A": reset, "B": reset},
                zero,
                {
                    "spatial_latches": {
                        label: write_spatial_latch(
                            board,
                            window=16,
                            upper=float(latch_config["upper"]),
                            lower=float(latch_config["lower"]),
                            retention=float(latch_config["decay"]),
                        )
                        for label, board in parent_boards.items()
                    },
                    "spatial_latch_strength": float(latch_config["kappa"]),
                },
            ),
            "incomplete_visible64_reset": (incomplete, zero, {"read_enabled": False}),
        }
        for condition, (resets, carriers, options) in condition_specs.items():
            evaluation_config = (
                ReaderConfiguration(
                    config.family,
                    config.write_window,
                    config.strength,
                    8,
                )
                if condition == "spatial_latch_benchmark"
                else config
            )
            common["conditions"][condition] = _evaluate_condition(
                resets,
                carriers,
                evaluation_config,
                primary,
                terminal,
                STAGE1_NAMESPACE,
                (pair["pair_id"], "validation", "native"),
                int(argument["replicates"]),
                **options,
            )
        result[config.configuration_id] = common
    return {
        "phase": "validation",
        "pair_id": pair["pair_id"],
        "unrelated_pair_id": unrelated["pair_id"],
        "configurations": result,
    }


def _run_checkpoint_tasks(
    arguments: Sequence[Mapping[str, Any]],
    worker: Any,
    checkpoint_dir: Path,
    binding: str,
    workers: int,
    artifacts: Path,
) -> int:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = checkpoint_dir.parent / "PROGRESS.json"
    pending = [
        argument
        for argument in arguments
        if not (checkpoint_dir / f"{argument['pair']['pair_id']}.json").exists()
    ]
    for argument in arguments:
        path = checkpoint_dir / f"{argument['pair']['pair_id']}.json"
        if path.exists():
            payload = read_checkpoint(path, binding)
            if payload.get("pair_id") != argument["pair"]["pair_id"]:
                raise ValueError(f"checkpoint pair identity mismatch in {path}")
    if not pending:
        atomic_write_json(
            progress_path,
            {
                "state": "complete",
                "completed": len(arguments),
                "total": len(arguments),
                "eta_seconds": 0.0,
                "last_update_unix": time.time(),
            },
        )
        return 0
    already_complete = len(arguments) - len(pending)
    started_monotonic = time.monotonic()
    started_unix = time.time()
    atomic_write_json(
        progress_path,
        {
            "state": "running",
            "completed": already_complete,
            "total": len(arguments),
            "eta_seconds": None,
            "session_started_unix": started_unix,
            "last_update_unix": started_unix,
        },
    )

    def record_progress(session_completed: int) -> None:
        elapsed = max(time.monotonic() - started_monotonic, 1e-9)
        rate = session_completed / elapsed
        completed_total = already_complete + session_completed
        remaining = len(arguments) - completed_total
        atomic_write_json(
            progress_path,
            {
                "state": "complete" if remaining == 0 else "running",
                "completed": completed_total,
                "total": len(arguments),
                "pairs_per_second": rate,
                "eta_seconds": remaining / rate if rate > 0 else None,
                "session_started_unix": started_unix,
                "last_update_unix": time.time(),
            },
        )

    def record_interruption(session_completed: int, error: BaseException) -> None:
        atomic_write_json(
            progress_path,
            {
                "state": "interrupted",
                "completed": already_complete + session_completed,
                "total": len(arguments),
                "eta_seconds": None,
                "session_started_unix": started_unix,
                "last_update_unix": time.time(),
                "error_type": type(error).__name__,
                "resume_safe": True,
            },
        )

    completed = 0
    if workers == 1:
        try:
            iterator = ((argument, worker(argument)) for argument in pending)
            for argument, payload in iterator:
                write_checkpoint(
                    checkpoint_dir / f"{argument['pair']['pair_id']}.json", binding, payload
                )
                completed += 1
                record_progress(completed)
                _write_status(artifacts)
        except BaseException as error:
            record_interruption(completed, error)
            _write_status(artifacts)
            raise
        return completed
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker, argument): argument for argument in pending}
            for future in as_completed(futures):
                argument = futures[future]
                payload = future.result()
                write_checkpoint(
                    checkpoint_dir / f"{argument['pair']['pair_id']}.json", binding, payload
                )
                completed += 1
                record_progress(completed)
                _write_status(artifacts)
    except BaseException as error:
        record_interruption(completed, error)
        _write_status(artifacts)
        raise
    return completed


def run_stage1(
    artifacts_root: Path | None = None,
    *,
    phase: str = "all",
    workers: int = 8,
) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    if workers < 1:
        raise ValueError("workers must be positive")
    registration = _load_registration(artifacts, 1)
    if phase not in {"calibration", "screen", "validation", "all"}:
        raise ValueError("phase must be calibration, screen, validation, or all")
    run_stage1_calibration(artifacts)
    if phase == "calibration":
        return status(artifacts)
    context = _input_context(artifacts)
    references = {
        str(window): {key: value.tolist() for key, value in reference.items()}
        for window, reference in _load_reference(artifacts, registration).items()
    }
    primary, terminal = _target_arrays(context["hypothesis"])
    common = {
        "references": references,
        "targets_primary": {key: value.tolist() for key, value in primary.items()},
        "targets_terminal": {key: value.tolist() for key, value in terminal.items()},
        "spatial_latch_benchmark": context["hypothesis"]["spatial_latch_benchmark"],
    }
    if phase in {"screen", "all"}:
        arguments = [
            {
                **common,
                "pair": pair,
                "donor_index": _donor_subset(context["donor_index"], pair),
                "configurations": registration["all_configurations"],
                "replicates": registration["profile"]["screen_replicates"],
            }
            for pair in registration["cohorts"]["discovery"]
        ]
        _run_checkpoint_tasks(
            arguments,
            _screen_worker,
            artifacts / "stage1" / "screen" / "checkpoints",
            registration["design_digest"],
            workers,
            artifacts,
        )
    if phase in {"validation", "all"}:
        screen_dir = artifacts / "stage1" / "screen" / "checkpoints"
        _, missing_screen = _checkpoint_payloads(
            screen_dir,
            registration["cohorts"]["discovery"],
            registration["design_digest"],
        )
        if missing_screen:
            raise RuntimeError("full fresh discovery screen must complete before validation")
        cohort = registration["cohorts"]["validation"]
        arguments = [
            {
                **common,
                "pair": pair,
                "unrelated": cohort[(index + 1) % len(cohort)],
                "donor_index": _donor_subset(
                    context["donor_index"], pair, cohort[(index + 1) % len(cohort)]
                ),
                "configurations": registration["validation_nominees"],
                "replicates": registration["profile"]["validation_replicates"],
            }
            for index, pair in enumerate(cohort)
        ]
        _run_checkpoint_tasks(
            arguments,
            _validation_worker,
            artifacts / "stage1" / "validation" / "checkpoints",
            registration["design_digest"],
            workers,
            artifacts,
        )
    _write_status(artifacts)
    return status(artifacts)


def _checkpoint_payloads(
    directory: Path,
    cohort: Sequence[Mapping[str, Any]],
    binding: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    payloads: list[dict[str, Any]] = []
    missing: list[str] = []
    for pair in cohort:
        path = directory / f"{pair['pair_id']}.json"
        if not path.exists():
            missing.append(pair["pair_id"])
        else:
            payloads.append(read_checkpoint(path, binding))
    return payloads, missing


def _screen_summary(
    payloads: Sequence[Mapping[str, Any]], configurations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for config in configurations:
        configuration_id = config["configuration_id"]
        checkpoints: dict[str, Any] = {}
        for checkpoint in STAGE1_CONTRACT["checkpoints"]:
            rows = [
                payload["configurations"][configuration_id]["conditions"]["intact"]
                ["checkpoints"][str(checkpoint)]["primary"]
                for payload in payloads
            ]
            terminals = [
                payload["configurations"][configuration_id]["conditions"]["intact"]
                ["checkpoints"][str(checkpoint)]["terminal"]
                for payload in payloads
            ]
            survivals = [
                payload["configurations"][configuration_id]["conditions"]["intact"]
                ["checkpoints"][str(checkpoint)]["survival"]
                for payload in payloads
            ]
            aggregate = aggregate_assignment(rows)
            aggregate["terminal_crossover"] = aggregate_assignment(terminals)["crossover"]
            aggregate["survival"] = float(np.mean(survivals))
            checkpoints[str(checkpoint)] = aggregate
        selected_checkpoint = max(
            STAGE1_CONTRACT["checkpoints"],
            key=lambda checkpoint: (
                checkpoints[str(checkpoint)]["crossover"],
                checkpoints[str(checkpoint)]["survival"],
                checkpoints[str(checkpoint)]["terminal_crossover"],
                checkpoints[str(checkpoint)]["fraction_pairs_positive"],
                -checkpoint,
            ),
        )
        summary[configuration_id] = {
            "configuration": dict(config),
            "selected_checkpoint": selected_checkpoint,
            "checkpoints": checkpoints,
        }
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    buffer = io.StringIO()
    fieldnames = list(rows[0])
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _seal_report(artifacts: Path, stage: int, filenames: Sequence[str]) -> None:
    stage_root = artifacts / f"stage{stage}"
    files = {name: sha256_file(stage_root / name) for name in filenames}
    atomic_write_json(
        stage_root / "MANIFEST.json",
        {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "files": files,
            "seal_digest": sha256_json(files),
        },
    )


def _stage1_raw_rows(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for configuration_id, configuration in payload["configurations"].items():
            for condition, outcome in configuration["conditions"].items():
                for checkpoint, checkpoint_result in outcome["checkpoints"].items():
                    for observer in ("primary", "terminal"):
                        metric = checkpoint_result[observer]
                        rows.append(
                            {
                                "pair_id": payload["pair_id"],
                                "phase": payload["phase"],
                                "configuration_id": configuration_id,
                                "condition": condition,
                                "checkpoint": checkpoint,
                                "observer": observer,
                                "p_a_given_a": metric["p_a_given_a"],
                                "p_a_given_b": metric["p_a_given_b"],
                                "p_b_given_a": metric["p_b_given_a"],
                                "p_b_given_b": metric["p_b_given_b"],
                                "direction_a": metric["direction_a"],
                                "direction_b": metric["direction_b"],
                                "crossover": metric["crossover"],
                                "correct": metric["correct"],
                                "resolved": metric["resolved"],
                                "survival": checkpoint_result["survival"],
                            }
                        )
    return rows


def stage1_report(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    registration = _load_registration(artifacts, 1)
    screen, missing_screen = _checkpoint_payloads(
        artifacts / "stage1" / "screen" / "checkpoints",
        registration["cohorts"]["discovery"],
        registration["design_digest"],
    )
    validation, missing_validation = _checkpoint_payloads(
        artifacts / "stage1" / "validation" / "checkpoints",
        registration["cohorts"]["validation"],
        registration["design_digest"],
    )
    complete = not missing_screen and not missing_validation
    if not complete:
        result = {
            "schema_version": SCHEMA_VERSION,
            "stage": 1,
            "complete": False,
            "verdict": "INCOMPLETE",
            "missing_screen_pair_ids": missing_screen,
            "missing_validation_pair_ids": missing_validation,
            "confirmatory_primary": FIXED_PRIMARY.configuration_id,
        }
        atomic_write_json(artifacts / "stage1" / "RESULTS.json", result)
        atomic_write_text(
            artifacts / "stage1" / "REPORT.md",
            "# Fresh motif-channel Stage 1\n\n"
            "The registered run is incomplete. No scientific gate can pass.\n",
        )
        _write_status(artifacts)
        return result
    screen_results = _screen_summary(screen, registration["all_configurations"])
    adjudications = {
        nominee["configuration_id"]: adjudicate_stage1(
            validation,
            nominee["configuration_id"],
            complete=True,
            namespace=registration["namespace"],
            resamples=int(registration["profile"]["bootstrap_resamples"]),
            checkpoint=int(nominee["selected_checkpoint"]),
        )
        for nominee in registration["validation_nominees"]
    }
    primary = adjudications[FIXED_PRIMARY.configuration_id]
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": 1,
        "complete": True,
        "design_digest": registration["design_digest"],
        "snapshot_digest": registration["snapshot_digest"],
        "fresh_pair_counts": {
            key: len(value) for key, value in registration["cohorts"].items()
        },
        "screen": screen_results,
        "validation": adjudications,
        "confirmatory_primary": FIXED_PRIMARY.configuration_id,
        "adjudication": primary,
        "verdict": primary["verdict"],
        "claim_boundary": STAGE1_CONTRACT["claim_boundary"],
    }
    _write_csv(artifacts / "stage1" / "RAW_SCREEN.csv", _stage1_raw_rows(screen))
    _write_csv(
        artifacts / "stage1" / "RAW_VALIDATION.csv", _stage1_raw_rows(validation)
    )
    atomic_write_json(artifacts / "stage1" / "RESULTS.json", result)
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": 1,
        "verdict": primary["verdict"],
        "review_required": True,
        "automatic_launch": False,
        "stage2_registration_allowed": bool(primary["robust"]),
        "selected_stage2_input": as_jsonable_configuration(FIXED_PRIMARY),
        "claim_boundary": "Stage 1 cannot establish Plastic Heredity",
        "results_sha256": sha256_file(artifacts / "stage1" / "RESULTS.json"),
    }
    atomic_write_json(artifacts / "stage1" / "STAGE_DECISION.json", decision)
    atomic_write_json(
        artifacts / "stage1" / "QUEUE.json",
        {
            "stage2": "AVAILABLE_FOR_EXPLICIT_REGISTRATION"
            if primary["robust"]
            else "HALTED",
            "automatic_launch": False,
            "stage3": "OUT_OF_SCOPE",
        },
    )
    intact = primary["conditions"]["intact"]
    report = (
        "# Fresh motif-channel Stage 1\n\n"
        f"Verdict: **{primary['verdict']}**.\n\n"
        f"The preregistered primary was `{FIXED_PRIMARY.configuration_id}`. "
        f"Its sweep-64 intact crossover was {intact['crossover']:.6f} "
        f"(familywise CI {intact['ci']}); survival was {primary['survival']:.6f}.\n\n"
        "All parent donors were frozen but excluded from every retained historical "
        "outcome pair. Daughter randomness and outcomes are new. The result is a "
        "one-generation controllability test, not evidence of Plastic Heredity.\n"
    )
    lay = (
        "# Lay summary\n\n"
        "We gave identical new cellular-automaton daughters different local motif "
        "memories written by A-like or B-like parents. The registered test asks "
        "whether those memories reliably steer the daughters toward the matching "
        "form. Even a positive result covers one generation only; it does not show "
        "a self-renewing inherited carrier.\n"
    )
    atomic_write_text(artifacts / "stage1" / "REPORT.md", report)
    atomic_write_text(artifacts / "stage1" / "LAY_SUMMARY.md", lay)
    _seal_report(
        artifacts,
        1,
        (
            "REGISTRATION.json",
            "CALIBRATION.json",
            "RESULTS.json",
            "STAGE_DECISION.json",
            "QUEUE.json",
            "REPORT.md",
            "LAY_SUMMARY.md",
            "RAW_SCREEN.csv",
            "RAW_VALIDATION.csv",
        ),
    )
    _write_status(artifacts)
    return result


def _count_valid_checkpoints(directory: Path, binding: str | None) -> int:
    if binding is None or not directory.exists():
        return 0
    count = 0
    for path in directory.glob("*.json"):
        read_checkpoint(path, binding)
        count += 1
    return count


def status(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    input_state = "missing"
    snapshot_digest = None
    try:
        manifest = verify_snapshot(artifacts / "input")
        input_state = "verified"
        snapshot_digest = manifest["snapshot_digest"]
    except FileNotFoundError:
        pass
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input_snapshot": input_state,
        "snapshot_digest": snapshot_digest,
        "stage1": {"state": "not_registered"},
        "stage2": {"state": "not_registered"},
        "automatic_launch": False,
    }
    for stage in (1, 2):
        registration_path = artifacts / f"stage{stage}" / "REGISTRATION.json"
        if not registration_path.exists():
            continue
        registration = load_json(registration_path)
        verify_registration(registration)
        binding = registration["design_digest"]
        if stage == 1:
            screen = _count_valid_checkpoints(
                artifacts / "stage1" / "screen" / "checkpoints", binding
            )
            validation = _count_valid_checkpoints(
                artifacts / "stage1" / "validation" / "checkpoints", binding
            )
            calibration = int((artifacts / "stage1" / "CALIBRATION.json").exists())
            expected_screen = len(registration["cohorts"]["discovery"])
            expected_validation = len(registration["cohorts"]["validation"])
            state = (
                "complete"
                if screen == expected_screen and validation == expected_validation
                else "in_progress"
                if screen or validation
                else "registered"
            )
            result["stage1"] = {
                "state": state,
                "calibration": f"{calibration}/1",
                "screen_pairs": f"{screen}/{expected_screen}",
                "validation_pairs": f"{validation}/{expected_validation}",
                "verdict": (
                    load_json(artifacts / "stage1" / "STAGE_DECISION.json")["verdict"]
                    if (artifacts / "stage1" / "STAGE_DECISION.json").exists()
                    else None
                ),
                "screen_progress": (
                    load_json(artifacts / "stage1" / "screen" / "PROGRESS.json")
                    if (artifacts / "stage1" / "screen" / "PROGRESS.json").exists()
                    else None
                ),
                "validation_progress": (
                    load_json(artifacts / "stage1" / "validation" / "PROGRESS.json")
                    if (artifacts / "stage1" / "validation" / "PROGRESS.json").exists()
                    else None
                ),
            }
        else:
            audit = int((artifacts / "stage2" / "WRITER_AUDIT.json").exists())
            completed = _count_valid_checkpoints(
                artifacts / "stage2" / "generalization" / "checkpoints", binding
            )
            expected = len(registration["cohorts"]["outcome"])
            state = (
                "complete"
                if completed == expected
                else "in_progress"
                if audit or completed
                else "registered"
            )
            result["stage2"] = {
                "state": state,
                "writer_audit": f"{audit}/1",
                "outcome_pairs": f"{completed}/{expected}",
                "verdict": (
                    load_json(artifacts / "stage2" / "STAGE_DECISION.json")["verdict"]
                    if (artifacts / "stage2" / "STAGE_DECISION.json").exists()
                    else None
                ),
                "progress": (
                    load_json(
                        artifacts / "stage2" / "generalization" / "PROGRESS.json"
                    )
                    if (
                        artifacts / "stage2" / "generalization" / "PROGRESS.json"
                    ).exists()
                    else None
                ),
            }
    return result


def _write_status(artifacts: Path) -> None:
    atomic_write_json(artifacts / "STATUS.json", status(artifacts))


def register_stage2(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    stage1_registration = _load_registration(artifacts, 1)
    decision_path = artifacts / "stage1" / "STAGE_DECISION.json"
    results_path = artifacts / "stage1" / "RESULTS.json"
    if not decision_path.exists() or not results_path.exists():
        raise RuntimeError("a complete reviewed Stage-1 report is required")
    decision = load_json(decision_path)
    if decision.get("stage2_registration_allowed") is not True:
        raise RuntimeError("Stage-1 robust gate did not authorize Stage-2 registration")
    context = _input_context(artifacts)
    pool_document = load_json(artifacts / "cohorts" / "FRESH_PAIR_POOL.json")
    pool = pool_document["pairs"]
    used_stage1 = [
        pair["pair_id"]
        for cohort in stage1_registration["cohorts"].values()
        for pair in cohort
    ]
    cohorts = allocate_stage2(pool, used_stage1, STAGE2_NAMESPACE)
    registration = seal_registration(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": 2,
            "state": "REGISTERED_NO_STAGE2_OUTCOMES",
            "namespace": STAGE2_NAMESPACE,
            "snapshot_digest": context["manifest"]["snapshot_digest"],
            "pair_pool_sha256": sha256_file(
                artifacts / "cohorts" / "FRESH_PAIR_POOL.json"
            ),
            "stage1_results_sha256": sha256_file(results_path),
            "stage1_decision_sha256": sha256_file(decision_path),
            "implementation_manifest": implementation_manifest(),
            "profile": STAGE2_PROFILE,
            "contract": STAGE2_CONTRACT,
            "configuration": as_jsonable_configuration(FIXED_PRIMARY),
            "cohorts": cohorts,
            "development_policy": (
                "two pairs quarantined before reference outcomes; this clean-room "
                "implementation runs no development daughter experiment"
            ),
            "reader_policy": "retained Stage-1 winner imported without tuning",
            "automatic_stage3_launch": False,
            "stage3": "OUT_OF_SCOPE",
            "added_experiments": [],
        }
    )
    path = artifacts / "stage2" / "REGISTRATION.json"
    if path.exists():
        existing = load_json(path)
        verify_registration(existing)
        if existing != registration:
            raise ValueError("Stage-2 registration already exists with a different design")
        return existing
    atomic_write_json(path, registration)
    _write_status(artifacts)
    return registration


def _motif_frequency(initial: np.ndarray, window: int = 32) -> np.ndarray:
    counts = parent_statistics(initial, window)["motif"].astype(np.float64)
    return counts / counts.sum()


def run_writer_audit(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    registration = _load_registration(artifacts, 2)
    path = artifacts / "stage2" / "WRITER_AUDIT.json"
    if path.exists():
        return read_checkpoint(path, registration["design_digest"])
    stage1_registration = _load_registration(artifacts, 1)
    reference = _load_reference(artifacts, stage1_registration)[32]
    context = _input_context(artifacts)
    donor_index = context["donor_index"]
    records: list[dict[str, Any]] = []
    symmetry_error = 0.0
    for pair in registration["cohorts"]["writer_audit"]:
        for label, key in (("A", "a_donor_id"), ("B", "b_donor_id")):
            board = decode_state_hex(donor_index[pair[key]]["donor_state_hex"])
            frequency = _motif_frequency(board)
            carrier = write_carrier(
                parent_statistics(board, 32), reference, "motif_energy512"
            )
            records.append(
                {
                    "pair_id": pair["pair_id"],
                    "label": label,
                    "carrier": carrier,
                }
            )
            for transform in ("translate_3_5", "rot90", "reflect_x"):
                transformed = _motif_frequency(transform_board(board, transform))
                expected = (
                    frequency
                    if transform.startswith("translate")
                    else transform_carrier(frequency, transform)
                )
                symmetry_error = max(
                    symmetry_error, float(np.max(np.abs(transformed - expected)))
                )
    correct = 0
    for record in records:
        training = [
            candidate
            for candidate in records
            if candidate["pair_id"] != record["pair_id"]
        ]
        centroids = {
            label: np.mean(
                [candidate["carrier"] for candidate in training if candidate["label"] == label],
                axis=0,
            )
            for label in ("A", "B")
        }
        vector = record["carrier"]
        scores = {
            label: (
                float(
                    np.dot(vector, centroid)
                    / (np.linalg.norm(vector) * np.linalg.norm(centroid))
                )
                if np.linalg.norm(vector) > 0 and np.linalg.norm(centroid) > 0
                else 0.0
            )
            for label, centroid in centroids.items()
        }
        predicted = max(scores, key=lambda label: (scores[label], label == "A"))
        correct += int(predicted == record["label"])
    accuracy = correct / len(records)
    payload = {
        "phase": "writer_audit",
        "pair_count": len(registration["cohorts"]["writer_audit"]),
        "history_count": len(records),
        "max_abs_symmetry_error": symmetry_error,
        "leave_one_pair_out_accuracy": accuracy,
        "symmetry_passed": symmetry_error <= STAGE2_CONTRACT["symmetry_tolerance"],
        "classification_passed": accuracy >= STAGE2_CONTRACT["writer_accuracy_gate"],
        "passed": bool(
            symmetry_error <= STAGE2_CONTRACT["symmetry_tolerance"]
            and accuracy >= STAGE2_CONTRACT["writer_accuracy_gate"]
        ),
        "adjudication_only": True,
    }
    write_checkpoint(path, registration["design_digest"], payload)
    _write_status(artifacts)
    return payload


def _environment_reset(
    environment: str,
    pair_id: str,
    launch_resets: Mapping[str, str],
) -> tuple[np.ndarray, str]:
    native = deterministic_board(STAGE2_NAMESPACE, pair_id, "native-reset")
    if environment == "native":
        return native, "identity"
    if environment.startswith("launch"):
        return decode_state_hex(launch_resets[environment]), "identity"
    if environment == "native_translate_3_5":
        return transform_board(native, "translate_3_5"), "translate_3_5"
    if environment == "native_rot90":
        return transform_board(native, "rot90"), "rot90"
    if environment == "native_reflect_x":
        return transform_board(native, "reflect_x"), "reflect_x"
    density = {
        "random_density_10": 0.10,
        "random_density_30": 0.30,
        "random_density_50": 0.50,
    }.get(environment)
    if density is not None:
        return (
            deterministic_board(
                STAGE2_NAMESPACE, pair_id, environment, density=density
            ),
            "identity",
        )
    raise ValueError(f"unknown environment: {environment}")


def _covariant_pair(carriers: CarrierPair, transform: str) -> CarrierPair:
    if transform in {"rot90", "reflect_x"}:
        return CarrierPair(
            transform_carrier(carriers.a, transform),
            transform_carrier(carriers.b, transform),
        )
    return carriers


def _stage2_worker(argument: Mapping[str, Any]) -> dict[str, Any]:
    pair = argument["pair"]
    unrelated = argument["unrelated"]
    donor_index = argument["donor_index"]
    reference = {key: np.asarray(value) for key, value in argument["reference"].items()}
    primary = {key: np.asarray(value) for key, value in argument["targets_primary"].items()}
    terminal = {key: np.asarray(value) for key, value in argument["targets_terminal"].items()}
    configuration = _configuration_from_dict(argument["configuration"])
    intact_native = _carrier_pair(
        pair, donor_index, reference, "motif_energy512", 32
    )
    unrelated_native = _carrier_pair(
        unrelated, donor_index, reference, "motif_energy512", 32
    )
    environments: dict[str, Any] = {}
    all_environments = list(argument["primary_environments"]) + list(
        argument["stress_environments"]
    )
    for environment in all_environments:
        reset, observation_transform = _environment_reset(
            environment, pair["pair_id"], argument["launch_resets"]
        )
        transform = (
            "rot90"
            if environment == "native_rot90"
            else "reflect_x"
            if environment == "native_reflect_x"
            else "identity"
        )
        intact = _covariant_pair(intact_native, transform)
        unrelated_carriers = _covariant_pair(unrelated_native, transform)
        zero = CarrierPair(np.zeros_like(intact.a), np.zeros_like(intact.b))
        opposite = CarrierPair(intact.b, intact.a)
        midpoint_value = 0.5 * (intact.a + intact.b)
        midpoint = CarrierPair(midpoint_value, midpoint_value)
        shuffled = CarrierPair(
            permute_carrier(
                intact.a, STAGE2_NAMESPACE, pair["pair_id"], environment, "shuffle"
            ),
            permute_carrier(
                intact.b, STAGE2_NAMESPACE, pair["pair_id"], environment, "shuffle"
            ),
        )
        matched_random = CarrierPair(
            permute_carrier(
                intact.a,
                STAGE2_NAMESPACE,
                pair["pair_id"],
                environment,
                "matched-random-A",
            ),
            permute_carrier(
                intact.b,
                STAGE2_NAMESPACE,
                pair["pair_id"],
                environment,
                "matched-random-B",
            ),
        )
        condition_carriers = {
            "intact": intact,
            "zero": zero,
            "read_disabled": intact,
            "shuffle": shuffled,
            "matched_random": matched_random,
            "opposite_history": opposite,
            "unrelated_pair": unrelated_carriers,
            "midpoint": midpoint,
        }
        conditions = (
            argument["stress_conditions"]
            if environment in argument["stress_environments"]
            else argument["core_conditions"]
        )
        environment_result = {"conditions": {}}
        for condition in conditions:
            environment_result["conditions"][condition] = _evaluate_condition(
                {"A": reset, "B": reset},
                condition_carriers[condition],
                configuration,
                primary,
                terminal,
                STAGE2_NAMESPACE,
                (pair["pair_id"], "generalization", environment),
                int(argument["replicates"]),
                read_enabled=condition != "read_disabled",
                observation_transform=observation_transform,
                collect_diagnostics=condition == "intact",
            )
        if environment == "native":
            corrupted = CarrierPair(
                corrupt_carrier_signs(
                    intact.a,
                    STAGE2_CONTRACT["carrier_corruption"],
                    STAGE2_NAMESPACE,
                    pair["pair_id"],
                    "native-corruption",
                ),
                corrupt_carrier_signs(
                    intact.b,
                    STAGE2_CONTRACT["carrier_corruption"],
                    STAGE2_NAMESPACE,
                    pair["pair_id"],
                    "native-corruption",
                ),
            )
            environment_result["conditions"]["process_noise"] = _evaluate_condition(
                {"A": reset, "B": reset},
                intact,
                configuration,
                primary,
                terminal,
                STAGE2_NAMESPACE,
                (pair["pair_id"], "generalization", environment),
                int(argument["replicates"]),
                process_noise=STAGE2_CONTRACT["process_noise"],
            )
            environment_result["conditions"][
                "carrier_sign_corruption"
            ] = _evaluate_condition(
                {"A": reset, "B": reset},
                corrupted,
                configuration,
                primary,
                terminal,
                STAGE2_NAMESPACE,
                (pair["pair_id"], "generalization", environment),
                int(argument["replicates"]),
            )
            for dose in argument["dose_contrasts"]:
                dose = float(dose)
                dose_pair = CarrierPair(
                    midpoint_value + dose * (intact.a - midpoint_value),
                    midpoint_value + dose * (intact.b - midpoint_value),
                )
                environment_result["conditions"][
                    f"dose_{dose:.2f}"
                ] = _evaluate_condition(
                    {"A": reset, "B": reset},
                    dose_pair,
                    configuration,
                    primary,
                    terminal,
                    STAGE2_NAMESPACE,
                    (pair["pair_id"], "generalization", environment),
                    int(argument["replicates"]),
                )
        environments[environment] = environment_result
    return {
        "phase": "stage2_generalization",
        "pair_id": pair["pair_id"],
        "unrelated_pair_id": unrelated["pair_id"],
        "configuration": as_jsonable_configuration(configuration),
        "environments": environments,
    }


def run_stage2(
    artifacts_root: Path | None = None,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    if workers < 1:
        raise ValueError("workers must be positive")
    registration = _load_registration(artifacts, 2)
    run_writer_audit(artifacts)
    context = _input_context(artifacts)
    stage1_registration = _load_registration(artifacts, 1)
    reference = _load_reference(artifacts, stage1_registration)[32]
    primary, terminal = _target_arrays(context["hypothesis"])
    cohort = registration["cohorts"]["outcome"]
    arguments = [
        {
            "pair": pair,
            "unrelated": cohort[(index + 1) % len(cohort)],
            "donor_index": _donor_subset(
                context["donor_index"], pair, cohort[(index + 1) % len(cohort)]
            ),
            "reference": {key: value.tolist() for key, value in reference.items()},
            "targets_primary": {key: value.tolist() for key, value in primary.items()},
            "targets_terminal": {key: value.tolist() for key, value in terminal.items()},
            "launch_resets": context["launch_resets"],
            "configuration": registration["configuration"],
            "replicates": registration["profile"]["replicates"],
            "primary_environments": registration["profile"]["primary_environments"],
            "core_conditions": registration["profile"]["core_conditions"],
            "stress_environments": registration["profile"]["stress_environments"],
            "stress_conditions": registration["profile"]["stress_conditions"],
            "dose_contrasts": registration["profile"]["dose_contrasts"],
        }
        for index, pair in enumerate(cohort)
    ]
    _run_checkpoint_tasks(
        arguments,
        _stage2_worker,
        artifacts / "stage2" / "generalization" / "checkpoints",
        registration["design_digest"],
        workers,
        artifacts,
    )
    _write_status(artifacts)
    return status(artifacts)


def stage2_report(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    registration = _load_registration(artifacts, 2)
    payloads, missing = _checkpoint_payloads(
        artifacts / "stage2" / "generalization" / "checkpoints",
        registration["cohorts"]["outcome"],
        registration["design_digest"],
    )
    audit_path = artifacts / "stage2" / "WRITER_AUDIT.json"
    audit = (
        read_checkpoint(audit_path, registration["design_digest"])
        if audit_path.exists()
        else {"passed": False, "state": "missing"}
    )
    complete = not missing and audit_path.exists()
    if complete:
        adjudication = adjudicate_stage2(
            payloads,
            audit,
            complete=True,
            namespace=registration["namespace"],
            resamples=int(registration["profile"]["bootstrap_resamples"]),
        )
    else:
        adjudication = {
            "complete": False,
            "verdict": "INCOMPLETE",
            "writer_audit": audit,
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": 2,
        "complete": complete,
        "design_digest": registration["design_digest"],
        "missing_pair_ids": missing,
        "configuration": registration["configuration"],
        "adjudication": adjudication,
        "verdict": adjudication["verdict"],
        "claim_boundary": STAGE2_CONTRACT["claim_boundary"],
    }
    if complete:
        raw_rows: list[dict[str, Any]] = []
        for payload in payloads:
            for environment, environment_result in payload["environments"].items():
                for condition, outcome in environment_result["conditions"].items():
                    for checkpoint, checkpoint_result in outcome["checkpoints"].items():
                        for observer in ("primary", "terminal"):
                            metric = checkpoint_result[observer]
                            raw_rows.append(
                                {
                                    "pair_id": payload["pair_id"],
                                    "environment": environment,
                                    "condition": condition,
                                    "checkpoint": checkpoint,
                                    "observer": observer,
                                    "p_a_given_a": metric["p_a_given_a"],
                                    "p_a_given_b": metric["p_a_given_b"],
                                    "p_b_given_a": metric["p_b_given_a"],
                                    "p_b_given_b": metric["p_b_given_b"],
                                    "direction_a": metric["direction_a"],
                                    "direction_b": metric["direction_b"],
                                    "crossover": metric["crossover"],
                                    "correct": metric["correct"],
                                    "resolved": metric["resolved"],
                                    "survival": checkpoint_result["survival"],
                                }
                            )
        _write_csv(artifacts / "stage2" / "RAW_OUTCOMES.csv", raw_rows)
    atomic_write_json(artifacts / "stage2" / "RESULTS.json", result)
    if complete:
        decision = {
            "schema_version": SCHEMA_VERSION,
            "stage": 2,
            "verdict": adjudication["verdict"],
            "review_required": True,
            "automatic_launch": False,
            "stage3": "OUT_OF_SCOPE",
            "claim_boundary": "Stage 2 cannot establish Plastic Heredity",
            "results_sha256": sha256_file(artifacts / "stage2" / "RESULTS.json"),
        }
        atomic_write_json(artifacts / "stage2" / "STAGE_DECISION.json", decision)
        atomic_write_json(
            artifacts / "stage2" / "QUEUE.json",
            {"automatic_launch": False, "stage3": "OUT_OF_SCOPE"},
        )
    report = (
        "# Fresh motif-channel Stage 2\n\n"
        f"Verdict: **{adjudication['verdict']}**.\n\n"
        + (
            "The fixed Stage-1 motif reader was tested without retuning across the "
            "registered reset, transformation, causal-control, noise, dose, and "
            "density panels. All outcomes use fresh donor pairs and fresh semantic "
            "random streams.\n\n"
            if complete
            else "The registered run is incomplete; no scientific gate can pass.\n\n"
        )
        + "This remains a one-generation form-channel test, not multigenerational "
        "Plastic Heredity. No Stage 3 is implemented or queued.\n"
    )
    atomic_write_text(artifacts / "stage2" / "REPORT.md", report)
    atomic_write_text(
        artifacts / "stage2" / "LAY_SUMMARY.md",
        "# Lay summary\n\nThe second stage asks whether the same local motif memory "
        "works for new parent pairs and several already-registered starting boards "
        "and transformations. It does not test whether daughters can rewrite and "
        "pass the memory onward.\n",
    )
    if complete:
        _seal_report(
            artifacts,
            2,
            (
                "REGISTRATION.json",
                "WRITER_AUDIT.json",
                "RESULTS.json",
                "STAGE_DECISION.json",
                "QUEUE.json",
                "REPORT.md",
                "LAY_SUMMARY.md",
                "RAW_OUTCOMES.csv",
            ),
        )
    _write_status(artifacts)
    return result


def verify_all(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    manifest = verify_snapshot(artifacts / "input")
    checks: dict[str, Any] = {
        "snapshot": True,
        "snapshot_digest": manifest["snapshot_digest"],
        "stage1_registration": False,
        "stage2_registration": False,
        "stage1_checkpoints": 0,
        "stage2_checkpoints": 0,
    }
    for stage in (1, 2):
        path = artifacts / f"stage{stage}" / "REGISTRATION.json"
        if not path.exists():
            continue
        registration = _load_registration(artifacts, stage)
        if registration["pair_pool_sha256"] != sha256_file(
            artifacts / "cohorts" / "FRESH_PAIR_POOL.json"
        ):
            raise ValueError(f"Stage-{stage} fresh pair-pool hash mismatch")
        if stage == 2:
            if registration["stage1_results_sha256"] != sha256_file(
                artifacts / "stage1" / "RESULTS.json"
            ):
                raise ValueError("Stage-2 binding to Stage-1 results is invalid")
            if registration["stage1_decision_sha256"] != sha256_file(
                artifacts / "stage1" / "STAGE_DECISION.json"
            ):
                raise ValueError("Stage-2 binding to Stage-1 decision is invalid")
        checks[f"stage{stage}_registration"] = True
        directories = (
            [
                artifacts / "stage1" / "CALIBRATION.json",
                artifacts / "stage1" / "screen" / "checkpoints",
                artifacts / "stage1" / "validation" / "checkpoints",
            ]
            if stage == 1
            else [
                artifacts / "stage2" / "WRITER_AUDIT.json",
                artifacts / "stage2" / "generalization" / "checkpoints",
            ]
        )
        count = 0
        for item in directories:
            if item.is_file():
                read_checkpoint(item, registration["design_digest"])
                count += 1
            elif item.is_dir():
                for checkpoint in item.glob("*.json"):
                    read_checkpoint(checkpoint, registration["design_digest"])
                    count += 1
        checks[f"stage{stage}_checkpoints"] = count
        decision_path = artifacts / f"stage{stage}" / "STAGE_DECISION.json"
        results_path = artifacts / f"stage{stage}" / "RESULTS.json"
        if decision_path.exists():
            decision = load_json(decision_path)
            if not results_path.exists() or decision["results_sha256"] != sha256_file(
                results_path
            ):
                raise ValueError(f"Stage-{stage} result/decision hash mismatch")
        report_manifest_path = artifacts / f"stage{stage}" / "MANIFEST.json"
        if report_manifest_path.exists():
            report_manifest = load_json(report_manifest_path)
            actual_files = {
                name: sha256_file(artifacts / f"stage{stage}" / name)
                for name in report_manifest["files"]
            }
            if actual_files != report_manifest["files"] or sha256_json(
                actual_files
            ) != report_manifest["seal_digest"]:
                raise ValueError(f"Stage-{stage} sealed report manifest mismatch")
    checks["valid"] = True
    atomic_write_json(artifacts / "VERIFY.json", checks)
    return checks
