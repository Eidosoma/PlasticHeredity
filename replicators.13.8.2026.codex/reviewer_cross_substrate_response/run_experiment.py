"""Staged CLI for the reviewer cross-substrate CA campaign."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Iterator, Sequence

TASK_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_ROOT.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd

from reviewer_cross_substrate_response.analysis import (
    confirmation_metrics,
    pilot_eligibility,
)
from reviewer_cross_substrate_response.campaign import (
    MODEL_NAMES,
    assign_matched_strangers,
    calibration_block,
    generate_landmarks,
    mechanics_jobs,
    mechanics_trial,
    parameter_payload,
    simulate_state_futures,
)
from reviewer_cross_substrate_response.core import (
    connected_components_torus,
    calibrated_threshold,
    canonical_digest,
    canonical_similarity,
    checksum_lines,
    derive_seed,
    exact_order_null_probability,
    json_ready,
    score_break_renewal,
    sha256_file,
)
from reviewer_cross_substrate_response.models import (
    EVOLOOP_SEED,
    EvoloopParameters,
    EvoloopRule,
    ProtocellParameters,
    mechanics_cells,
    place_pattern,
    profile_named,
    protocell_initial,
    protocell_sweep,
)


FORMAT = "reviewer-cross-substrate-ca-v1"
SOURCE_FILES = (
    "__init__.py",
    "core.py",
    "models.py",
    "campaign.py",
    "analysis.py",
    "run_experiment.py",
    "evoloop.table",
    "README.md",
    "HYPOTHESIS_PROVENANCE.md",
    "REVIEW_AND_PLAN.md",
    "REVIEWER_RESPONSE_DRAFT.md",
    "PREREGISTRATION.md",
    "PUBLIC_SOURCES.md",
    "run_developmental_pilot.sh",
    "tests/__init__.py",
    "tests/test_cross_substrate.py",
)

FUTURE_COLUMNS = (
    "model", "stage", "block_id", "parameter_key", "landmark", "branch",
    "future_id", "main_complete", "half", "event", "break_index", "renewal_start", "observed_boundaries",
    "inherited_count", "complete_horizon", "order_null_probability",
    "event_minus_order_null", "failure",
)
BOUNDARY_COLUMNS = (
    "model", "stage", "block_id", "parameter_key", "landmark", "branch",
    "half", "future_id", "boundary", "similarity", "stranger_similarity",
    "inherited", "parent_size", "child_size", "elapsed_updates",
    "observation_index",
)
CALIBRATION_COLUMNS = (
    "model", "block_id", "parameter_key", "boundary_index",
    "lineage_attempt", "parent_size", "child_size", "actual_similarity",
    "stranger_similarity",
)
CALIBRATION_CHECKPOINT_COLUMNS = (
    "model", "block_id", "parameter_key", "boundary_index",
    "lineage_attempt", "parent_size", "child_size", "actual_similarity",
    "observation_index",
)


def roots(profile_name: str) -> dict[str, Path]:
    base = TASK_ROOT / "artifacts" if profile_name == "full" else TASK_ROOT / "artifacts" / "smoke"
    return {
        "base": base,
        "protocol": base / "protocol",
        "work": base / "work",
        "output": base / "output",
        "verification": base / "verification",
    }


def _assert_local(path: Path) -> None:
    resolved = path.resolve()
    if resolved != TASK_ROOT and TASK_ROOT not in resolved.parents:
        raise ValueError(f"refusing write outside reviewer folder: {resolved}")


def _write_json(path: Path, value: Any) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, value: str) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _pack_rasters(rasters: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shapes = np.asarray([raster.shape for raster in rasters], dtype=np.int32)
    if not rasters:
        shapes = np.zeros((0, 2), dtype=np.int32)
    sizes = np.asarray([raster.size for raster in rasters], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(sizes)))
    data = (
        np.concatenate([np.asarray(raster, dtype=np.uint8).ravel() for raster in rasters])
        if rasters else np.zeros(0, dtype=np.uint8)
    )
    return data, offsets, shapes


def _write_crop_archive(path: Path, crop_rows: Sequence[dict[str, Any]]) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(crop_rows, key=lambda item: int(item["observation_index"]))
    expected = list(range(len(ordered)))
    observed = [int(item["observation_index"]) for item in ordered]
    if observed != expected:
        raise ValueError("crop observation indices must be contiguous")
    group_keys = sorted(
        {(int(item["landmark"]), int(item["boundary"])) for item in ordered}
    )
    group_lookup = {key: index for index, key in enumerate(group_keys)}
    observation_group = np.zeros(len(ordered), dtype=np.int16)
    observation_local = np.zeros(len(ordered), dtype=np.int32)
    payload: dict[str, np.ndarray] = {
        "group_keys": np.asarray(group_keys, dtype=np.int32).reshape((-1, 2)),
        "observation_group": observation_group,
        "observation_local": observation_local,
    }
    for key, group_id in group_lookup.items():
        members = [
            item
            for item in ordered
            if (int(item["landmark"]), int(item["boundary"])) == key
        ]
        for local_index, item in enumerate(members):
            observation_index = int(item["observation_index"])
            observation_group[observation_index] = group_id
            observation_local[observation_index] = local_index
        for prefix in ("parent", "child"):
            data, offsets, shapes = _pack_rasters(
                [np.asarray(item[f"{prefix}_crop"], dtype=np.uint8) for item in members]
            )
            payload[f"{prefix}_data_{group_id:03d}"] = data
            payload[f"{prefix}_offsets_{group_id:03d}"] = offsets
            payload[f"{prefix}_shapes_{group_id:03d}"] = shapes
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _archive_raster(
    archive: Any,
    prefix: str,
    index: int,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    cache = {} if cache is None else cache

    def cached(key: str) -> np.ndarray:
        if key not in cache:
            cache[key] = archive[key]
        return cache[key]

    group_id = int(cached("observation_group")[index])
    local_index = int(cached("observation_local")[index])
    suffix = f"{group_id:03d}"
    offsets = cached(f"{prefix}_offsets_{suffix}")
    shapes = cached(f"{prefix}_shapes_{suffix}")
    start, stop = int(offsets[local_index]), int(offsets[local_index + 1])
    shape = tuple(int(value) for value in shapes[local_index])
    return np.asarray(
        cached(f"{prefix}_data_{suffix}")[start:stop], dtype=np.uint8
    ).reshape(shape)


def _write_checksums(directory: Path) -> None:
    _assert_local(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write_text(directory / "SHA256SUMS", "\n".join(checksum_lines(directory)) + "\n")


def _verify_checksums(directory: Path) -> dict[str, bool]:
    path = directory / "SHA256SUMS"
    if not path.is_file():
        raise FileNotFoundError(path)
    results: dict[str, bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        target = directory / relative
        results[relative] = target.is_file() and sha256_file(target) == expected
    if not results or not all(results.values()):
        raise ValueError(f"checksum failure: {[key for key, value in results.items() if not value]}")
    return results


def _source_manifest() -> dict[str, Any]:
    files = {name: TASK_ROOT / name for name in SOURCE_FILES}
    files["requirements-lock.txt"] = REPOSITORY_ROOT / "requirements-lock.txt"
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing campaign sources: {missing}")
    return {
        "source_firewall": "No path below ../NewIdeas is read, imported, or hashed.",
        "files": {
            name: {
                "path": str(path.resolve().relative_to(REPOSITORY_ROOT.resolve())),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(files.items())
        },
    }


def _verify_sources(profile_name: str) -> None:
    manifest_path = roots(profile_name)["protocol"] / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("run prepare first")
    frozen = _read_json(manifest_path)
    current = _source_manifest()
    if frozen != current:
        changed = [
            name
            for name, value in frozen.get("files", {}).items()
            if current.get("files", {}).get(name) != value
        ]
        raise ValueError(f"campaign scientific sources changed after prepare: {changed}")


def _runtime() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def _protocol(profile_name: str) -> dict[str, Any]:
    profile = profile_named(profile_name)
    protocol = {
        "format": FORMAT,
        "profile": asdict(profile),
        "models": {
            model: [parameter_payload(item) for item in mechanics_cells(model)]
            for model in MODEL_NAMES
        },
        "lineage_contracts": {
            "protocell": {
                "states": ["empty", "X", "Y"],
                "updates": "random-order replication, degradation, diffusion",
                "initial": "10x10 X patch with central Y on periodic lattice",
                "individual": "radius-two molecular-proximity cluster with >=20 sites and Y",
                "boundary": "two Y-centred Voronoi lobes >=8 sites apart, each >=20 sites, persistent",
            },
            "evoloop": {
                "states": 9,
                "updates": "synchronous public rotate-four von-Neumann table",
                "ecology": "Poisson canonical-loop immigration per tick with empty 9x9 placement box",
                "individual": "component >=20 sites with >=1 enclosed region",
                "boundary": "provenance >=0.80, persistence, then construction-arm launch",
                "provenance_is_passive": True,
            },
        },
        "mechanics_gate": {
            "successful_seeds_required": profile.mechanics_required,
            "screen_seeds": profile.mechanics_seeds,
            "boundaries_per_seed": profile.mechanics_boundaries,
            "total_update_cap": profile.mechanics_cap,
            "per_boundary_update_cap": profile.boundary_cap,
            "occupancy_limit": 0.25,
            "maximum_ambiguity_fraction": 0.05,
            "selection_uses_endpoint_outcomes": False,
        },
        "source_boundary": {
            "newideas": "hypothesis only; directory must never be accessed",
            "existing_repository": "public event topology only; no scientific code imports",
            "public_models": [
                "Kamimura and Kaneko, Life 4 (2014) 586-597, doi:10.3390/life4040586",
                "Sayama, Artificial Life 5 (1999) 343-365, plus the public Golly Evoloop table",
            ],
        },
        "mechanics_amendments_before_outcomes": [
            "Evoloop qualification uses at least one enclosed region because the canonical seed contains several.",
            "Protocell individuality uses radius-two proximity and Y-centred persistent lobes because diffusion makes localized clouds porous.",
        ],
        "similarity": {
            "representation": "centroid-aligned non-background one-hot raster",
            "invariance": "maximum cosine over C4 rotations and shifts {-1,0,1}^2",
            "threshold": "model-specific method=higher 95th matched-stranger percentile",
            "strict_inheritance": "S > tau",
            "break": "S <= tau",
            "stranger_control": "child-size-matched child from a different independently seeded world block",
        },
        "endpoint": {
            "name": "CA_BREAK_RENEW_3_F12",
            "horizon": 12,
            "run_length": 3,
            "ordering": "break strictly before run",
            "failure": "adverse before certification",
            "simulated_boundaries": 16,
        },
        "inference": {
            "unit": "world block",
            "bootstrap_repetitions": 4096,
            "randomization_repetitions": 4096,
            "model_alpha_one_sided": 0.025,
            "models_pooled": False,
            "bootstrap_unit": "equal-weight world-block means",
            "headline": "either complete model gate passes",
        },
        "sensitivities": {
            "status": "descriptive and non-rescuing",
            "variants": ["raw S>0.9", "F8", "F16", "renewal run 2", "renewal run 4"],
        },
        "stop_rule": "pilot-report is mandatory stop; confirmation requires a later explicit command",
    }
    protocol["protocol_digest"] = canonical_digest(protocol)
    return protocol


def prepare(profile_name: str) -> dict[str, Any]:
    paths = roots(profile_name)
    registration = paths["protocol"] / "registration.json"
    protocol = _protocol(profile_name)
    manifest = _source_manifest()
    if registration.exists():
        frozen = _read_json(registration)
        if frozen.get("protocol_digest") != protocol["protocol_digest"]:
            raise ValueError("refusing to overwrite a different frozen protocol")
        _verify_sources(profile_name)
        _verify_checksums(paths["protocol"])
        return frozen
    for key in ("protocol", "work", "output", "verification"):
        paths[key].mkdir(parents=True, exist_ok=True)
    _write_json(paths["protocol"] / "protocol.json", protocol)
    _write_json(paths["protocol"] / "source_manifest.json", manifest)
    seed_registry = {
        "master_commitment": hashlib.sha256(b"20260820-cross-substrate-ca-v1").hexdigest(),
        "domains": [
            "mechanics", "calibration", "pilot/main", "pilot/future",
            "confirmation/main", "confirmation/future", "strangers",
            "bootstrap", "randomization", "replay",
        ],
    }
    _write_json(paths["protocol"] / "seed_registry.json", seed_registry)
    frozen = {
        "format": FORMAT,
        "profile": profile_name,
        "protocol_digest": protocol["protocol_digest"],
        "source_manifest_digest": canonical_digest(manifest),
        "runtime": _runtime(),
        "prepared": True,
        "scientific_outcomes_generated": False,
    }
    _write_json(registration, frozen)
    _write_checksums(paths["protocol"])
    return frozen


def validation_checks(profile_name: str) -> dict[str, Any]:
    _verify_sources(profile_name)
    _verify_checksums(roots(profile_name)["protocol"])
    checks: dict[str, bool] = {}
    threshold = 0.9
    checks["endpoint_positive"] = score_break_renewal([0.5, 0.95, 0.96, 0.97], threshold).event
    checks["endpoint_requires_later_run"] = not score_break_renewal([0.95, 0.96, 0.97, 0.5], threshold).event
    checks["threshold_is_strict"] = not score_break_renewal([0.5, 0.9, 0.95, 0.96], threshold).event
    checks["order_null_all_inherited_zero"] = exact_order_null_probability(12, 12) == 0.0
    checks["order_null_all_broken_zero"] = exact_order_null_probability(12, 0) == 0.0

    seed = EVOLOOP_SEED
    checks["similarity_identity"] = np.isclose(canonical_similarity(seed, seed), 1.0)
    checks["similarity_rotation"] = np.isclose(canonical_similarity(seed, np.rot90(seed)), 1.0)
    changed = seed.copy()
    changed[1, 1] = 1
    checks["similarity_detects_change"] = canonical_similarity(seed, changed) < 1.0

    rule = EvoloopRule()
    checks["evoloop_rule_count"] = len(rule.lut.shape) == 5 and rule.covered_neighborhoods == 55_139
    grid = np.zeros((96, 96), dtype=np.uint8)
    checks["evoloop_seed_placement"] = place_pattern(grid, seed, 40, 40)
    counts = {0: int(np.count_nonzero(grid))}
    for tick in range(1, 151):
        grid = rule.step(grid)
        if tick in (1, 10, 50, 100, 150):
            counts[tick] = int(np.count_nonzero(grid))
    checks["evoloop_published_fixture"] = counts == {0: 60, 1: 60, 10: 65, 50: 86, 100: 114, 150: 137}
    checks["evoloop_separates_descendant"] = len(connected_components_torus(grid > 0, min_size=20)) == 2

    protocell = protocell_initial(64).grid
    rng = np.random.default_rng(123)
    protocell_sweep(protocell, ProtocellParameters.from_pair(0.1, 1e-4), rng)
    checks["protocell_exclusion"] = bool(np.isin(protocell, [0, 1, 2]).all())
    checks["protocell_fixture_counts"] = int(np.count_nonzero(protocell == 1)) == 99 and int(np.count_nonzero(protocell == 2)) == 1
    checks["protocell_fixture_digest"] = hashlib.sha256(protocell.tobytes()).hexdigest() == "483e706c420189d43d26540889ea0cfbbf85250f20860675b6bc51e64253dbc1"
    checks["seed_domains_separate"] = len({derive_seed(name, 1) for name in ("mechanics", "calibration", "pilot", "confirmation", "replay")}) == 5
    checks["source_firewall"] = all("NewIdeas" not in item["path"] for item in _source_manifest()["files"].values())

    result = {
        "format": FORMAT,
        "profile": profile_name,
        "checks": checks,
        "evoloop_fixture_counts": counts,
        "all_passed": all(checks.values()),
        "scientific_outcomes_disclosed": False,
    }
    paths = roots(profile_name)
    _write_json(paths["protocol"] / "validation.json", result)
    _write_checksums(paths["protocol"])
    if not result["all_passed"]:
        raise ValueError(f"validation failed: {[name for name, passed in checks.items() if not passed]}")
    return result


def _mechanics_worker(job: tuple[str, dict[str, Any], Any, int]) -> dict[str, Any]:
    return asdict(mechanics_trial(*job))


def _parallel_map(
    function: Callable[[Any], Any], jobs: Sequence[Any], workers: int
) -> list[Any]:
    if workers <= 1:
        return [function(job) for job in jobs]
    output: list[Any] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, job): index for index, job in enumerate(jobs)}
        ordered: dict[int, Any] = {}
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
        output = [ordered[index] for index in range(len(jobs))]
    return output


def _parallel_results(
    function: Callable[[Any], Any], jobs: Sequence[Any], workers: int
) -> Iterator[Any]:
    """Yield large results as workers finish so raster blocks are not pooled in RAM."""

    if workers <= 1:
        for job in jobs:
            yield function(job)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(function, job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()


def _mechanics_checkpoint_path(
    paths: dict[str, Path], model: str, parameter_key: str, seed_index: int
) -> Path:
    cell = hashlib.sha256(parameter_key.encode("utf-8")).hexdigest()[:16]
    return (
        paths["work"]
        / "mechanics"
        / model
        / f"cell_{cell}_seed_{seed_index:02d}.json"
    )


def _calibration_checkpoint_paths(
    paths: dict[str, Path], model: str, block_id: int
) -> tuple[Path, Path, Path]:
    root = paths["work"] / "calibration" / "checkpoints" / model
    stem = f"block_{block_id:04d}"
    return (
        root / f"{stem}.json",
        root / f"{stem}_observations.csv",
        root / f"{stem}_crops.npz",
    )


def _write_calibration_checkpoint(
    paths: dict[str, Path], model: str, block_id: int, observations: Sequence[dict[str, Any]]
) -> None:
    manifest_path, table_path, crop_path = _calibration_checkpoint_paths(
        paths, model, block_id
    )
    rows: list[dict[str, Any]] = []
    crops: list[dict[str, Any]] = []
    for observation_index, item in enumerate(observations):
        rows.append(
            {
                key: (observation_index if key == "observation_index" else item[key])
                for key in CALIBRATION_CHECKPOINT_COLUMNS
            }
        )
        crops.append(
            {
                "observation_index": observation_index,
                "landmark": 0,
                "boundary": 0,
                "branch": observation_index,
                "parent_crop": item["parent_crop"],
                "child_crop": item["child_crop"],
            }
        )
    _write_csv(table_path, pd.DataFrame(rows, columns=CALIBRATION_CHECKPOINT_COLUMNS))
    _write_crop_archive(crop_path, crops)
    _write_json(
        manifest_path,
        {
            "model": model,
            "block_id": block_id,
            "observation_count": len(rows),
            "table_sha256": sha256_file(table_path),
            "crop_sha256": sha256_file(crop_path),
        },
    )


def _read_calibration_checkpoint(
    paths: dict[str, Path], model: str, block_id: int
) -> list[dict[str, Any]]:
    manifest_path, table_path, crop_path = _calibration_checkpoint_paths(
        paths, model, block_id
    )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("model") != model
        or int(manifest.get("block_id", -1)) != block_id
        or not table_path.is_file()
        or not crop_path.is_file()
        or sha256_file(table_path) != manifest.get("table_sha256")
        or sha256_file(crop_path) != manifest.get("crop_sha256")
    ):
        raise ValueError(f"invalid calibration checkpoint: {model}:{block_id}")
    frame = pd.read_csv(table_path, float_precision="round_trip")
    if int(frame.shape[0]) != int(manifest["observation_count"]):
        raise ValueError(f"calibration checkpoint length mismatch: {model}:{block_id}")
    observations: list[dict[str, Any]] = []
    with np.load(crop_path, allow_pickle=False) as archive:
        if int(archive["observation_group"].shape[0]) != int(frame.shape[0]):
            raise ValueError(f"calibration raster length mismatch: {model}:{block_id}")
        cache: dict[str, np.ndarray] = {}
        for _, row in frame.iterrows():
            observation_index = int(row["observation_index"])
            item = {
                key: row[key]
                for key in CALIBRATION_CHECKPOINT_COLUMNS
                if key != "observation_index"
            }
            item["block_id"] = int(item["block_id"])
            item["boundary_index"] = int(item["boundary_index"])
            item["lineage_attempt"] = int(item["lineage_attempt"])
            item["parent_size"] = int(item["parent_size"])
            item["child_size"] = int(item["child_size"])
            item["actual_similarity"] = float(item["actual_similarity"])
            item["parent_crop"] = _archive_raster(
                archive, "parent", observation_index, cache
            )
            item["child_crop"] = _archive_raster(
                archive, "child", observation_index, cache
            )
            observations.append(item)
    return observations


def ensure_mechanics(profile_name: str, models: Sequence[str], workers: int) -> dict[str, Any]:
    paths = roots(profile_name)
    summary_path = paths["protocol"] / "mechanics_summary.json"
    trials_path = paths["protocol"] / "mechanics_trials.csv"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if all(model in summary["models"] for model in models):
            return summary
    _verify_sources(profile_name)
    _verify_checksums(paths["protocol"])
    validation = paths["protocol"] / "validation.json"
    if not validation.is_file() or not _read_json(validation)["all_passed"]:
        raise ValueError("passing validation is required before mechanics screening")
    profile = profile_named(profile_name)
    requested_jobs = [job for model in models for job in mechanics_jobs(model, profile)]
    missing_jobs = [
        job
        for job in requested_jobs
        if not _mechanics_checkpoint_path(
            paths, job[0], str(job[1]["parameter_key"]), int(job[3])
        ).is_file()
    ]
    for result in _parallel_results(_mechanics_worker, missing_jobs, workers):
        checkpoint = _mechanics_checkpoint_path(
            paths,
            str(result["model"]),
            str(result["parameter_key"]),
            int(result["seed_index"]),
        )
        _write_json(checkpoint, result)

    available_models = [
        model
        for model in MODEL_NAMES
        if (paths["work"] / "mechanics" / model).is_dir()
    ]
    all_rows: list[dict[str, Any]] = []
    for model in available_models:
        for job in mechanics_jobs(model, profile):
            checkpoint = _mechanics_checkpoint_path(
                paths, model, str(job[1]["parameter_key"]), int(job[3])
            )
            if not checkpoint.is_file():
                raise ValueError(f"missing mechanics checkpoint: {checkpoint.name}")
            row = _read_json(checkpoint)
            if (
                row.get("model") != model
                or row.get("parameter_key") != job[1]["parameter_key"]
                or int(row.get("seed_index", -1)) != int(job[3])
            ):
                raise ValueError(f"invalid mechanics checkpoint: {checkpoint.name}")
            all_rows.append(row)
    frame = pd.DataFrame(all_rows)
    if not frame.empty:
        frame = frame.drop_duplicates(
            subset=["model", "parameter_key", "seed_index"], keep="last"
        ).sort_values(["model", "parameter_key", "seed_index"], ignore_index=True)
    viable: dict[str, list[dict[str, Any]]] = {}
    model_summaries: dict[str, Any] = {}
    summarized_models = sorted(set(str(value) for value in frame["model"].unique()))
    for model in summarized_models:
        model_frame = frame[frame["model"] == model]
        counts = model_frame.groupby("parameter_key")["passed"].sum().to_dict()
        eligible_keys = sorted(
            key for key, count in counts.items() if int(count) >= profile.mechanics_required
        )
        payloads = [parameter_payload(item) for item in mechanics_cells(model)]
        if profile_name == "smoke":
            payloads = [
                parameter_payload(ProtocellParameters.from_pair(1e-2, 1e-4))
                if model == "protocell" else parameter_payload(EvoloopParameters(1, 0.0))
            ]
        viable[model] = [item for item in payloads if item["parameter_key"] in eligible_keys]
        model_summaries[model] = {
            "tested_cells": int(model_frame["parameter_key"].nunique()),
            "viable_cells": len(viable[model]),
            "viable_parameter_keys": eligible_keys,
            "model_eligible_for_calibration": bool(viable[model]),
        }
    summary = {
        "format": FORMAT,
        "profile": profile_name,
        "models": model_summaries,
        "viable_parameters": viable,
        "mechanics_only": True,
    }
    checkpoint_files = sorted(
        (paths["work"] / "mechanics").glob("**/*.json")
    )
    summary["mechanics_checkpoint_digest"] = canonical_digest(
        {
            str(path.relative_to(paths["work"] / "mechanics")): sha256_file(path)
            for path in checkpoint_files
        }
    )
    _write_csv(trials_path, frame)
    summary["mechanics_trials_sha256"] = sha256_file(trials_path)
    _write_json(summary_path, summary)
    _write_checksums(paths["protocol"])
    return summary


def _calibration_worker(job: tuple[str, dict[str, Any], Any, int]) -> dict[str, Any]:
    return {
        "model": job[0],
        "block_id": int(job[3]),
        "observations": calibration_block(*job),
    }


def calibrate(profile_name: str, models: Sequence[str], workers: int) -> dict[str, Any]:
    _verify_sources(profile_name)
    paths = roots(profile_name)
    _verify_checksums(paths["protocol"])
    profile = profile_named(profile_name)
    mechanics = ensure_mechanics(profile_name, models, workers)
    thresholds_path = paths["protocol"] / "thresholds.json"
    existing = _read_json(thresholds_path) if thresholds_path.is_file() else {"models": {}}
    pilot_started = (
        (paths["work"] / "pilot" / "stage_summary.json").is_file()
        or (paths["output"] / "pilot_eligibility.json").is_file()
        or any((paths["work"] / "pilot").glob("**/block_*.json"))
    )
    missing_after_pilot = [
        model for model in models if model not in existing.get("models", {})
    ]
    if pilot_started and missing_after_pilot:
        raise ValueError("calibration is frozen once any pilot block has started")
    for model in models:
        if model in existing.get("models", {}):
            continue
        viable = mechanics["viable_parameters"].get(model, [])
        if not viable:
            existing.setdefault("models", {})[model] = {"eligible": False, "reason": "no viable mechanics cells"}
            continue
        jobs = [
            (model, viable[block_id % len(viable)], profile, block_id)
            for block_id in range(profile.calibration_blocks)
        ]
        missing_jobs = [
            job
            for job in jobs
            if not _calibration_checkpoint_paths(
                paths, model, int(job[3])
            )[0].is_file()
        ]
        for result in _parallel_results(_calibration_worker, missing_jobs, workers):
            _write_calibration_checkpoint(
                paths,
                str(result["model"]),
                int(result["block_id"]),
                result["observations"],
            )
        observations = [
            row
            for job in jobs
            for row in _read_calibration_checkpoint(paths, model, int(job[3]))
        ]
        assign_matched_strangers(
            observations,
            seed_parts=("calibration", "strangers", model),
            different_key="block_id",
        )
        rows = [
            {
                key: value
                for key, value in item.items()
                if key not in {"parent_crop", "child_crop"}
            }
            for item in observations
        ]
        frame = pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)
        calibration_root = paths["work"] / "calibration"
        calibration_path = calibration_root / f"{model}_pairs.csv"
        _write_csv(calibration_path, frame)
        usable = frame[np.isfinite(frame["stranger_similarity"])] if not frame.empty else frame
        required_pairs = 1_000 if profile_name == "full" else profile.calibration_blocks * profile.calibration_pairs
        required_blocks = 16 if profile_name == "full" else profile.calibration_blocks
        if usable.shape[0] < required_pairs or usable["block_id"].nunique() < required_blocks:
            existing.setdefault("models", {})[model] = {
                "eligible": False,
                "reason": "insufficient matched calibration pairs",
                "usable_pairs": int(usable.shape[0]),
                "usable_blocks": int(usable["block_id"].nunique()) if not usable.empty else 0,
                "calibration_file_sha256": sha256_file(calibration_path),
                "strangers_from_different_blocks": True,
            }
            continue
        threshold = calibrated_threshold(usable["stranger_similarity"].to_numpy())
        existing.setdefault("models", {})[model] = {
            "eligible": True,
            "threshold": threshold,
            "threshold_hex": threshold.hex(),
            "quantile": 0.95,
            "quantile_method": "higher",
            "usable_pairs": int(usable.shape[0]),
            "usable_blocks": int(usable["block_id"].nunique()),
            "actual_similarity_mean": float(usable["actual_similarity"].mean()),
            "stranger_similarity_mean": float(usable["stranger_similarity"].mean()),
            "calibration_file_sha256": sha256_file(calibration_path),
            "strangers_from_different_blocks": True,
        }
    existing.update(
        {
            "format": FORMAT,
            "profile": profile_name,
            "complete_model_dispositions": all(
                model in existing.get("models", {}) for model in MODEL_NAMES
            ),
        }
    )
    existing["frozen_before_pilot"] = existing["complete_model_dispositions"]
    _write_json(thresholds_path, existing)
    _write_checksums(paths["protocol"])
    return existing


def _block_digest(rows: Sequence[dict[str, Any]]) -> str:
    return canonical_digest(rows)


def _simulate_block_job(job: tuple[str, str, int, dict[str, Any], Any, float, int]) -> dict[str, Any]:
    stage, model, block_id, parameter_data, profile, threshold, branches = job
    from reviewer_cross_substrate_response.campaign import parameter_from_payload

    parameters = parameter_from_payload(parameter_data)
    states, history, attempt = generate_landmarks(
        model, parameters, profile, block_id, stage=stage
    )
    if attempt is None:
        half_width = branches // 2
        failed_futures = [
            {
                "model": model,
                "stage": stage,
                "block_id": block_id,
                "parameter_key": parameters.key,
                "landmark": landmark,
                "branch": branch,
                "future_id": f"{model}:{stage}:{block_id}:{landmark}:{branch}",
                "main_complete": 0,
                "half": "A" if branch < half_width else "B",
                "event": 0,
                "break_index": None,
                "renewal_start": None,
                "observed_boundaries": 0,
                "inherited_count": 0,
                "complete_horizon": 0,
                "order_null_probability": 0.0,
                "event_minus_order_null": 0.0,
                "failure": "main_unavailable_after_100_attempts",
            }
            for landmark in profile.landmarks
            for branch in range(branches)
        ]
        return {
            "model": model,
            "block_id": block_id,
            "parameter_key": parameters.key,
            "main_complete": False,
            "attempt": None,
            "future_rows": failed_futures,
            "boundary_rows": [],
            "crop_rows": [],
            "history_digest": None,
            "future_digest": canonical_digest(failed_futures),
            "boundary_digest": canonical_digest([]),
        }
    futures: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    crops: list[dict[str, Any]] = []
    for landmark, world in zip(profile.landmarks, states, strict=True):
        future_rows, boundary_rows, crop_rows = simulate_state_futures(
            model,
            parameters,
            profile,
            world,
            threshold,
            stage=stage,
            block_id=block_id,
            landmark=landmark,
            branches=branches,
        )
        crop_offset = len(crops)
        for row in boundary_rows:
            row["observation_index"] = int(row["observation_index"]) + crop_offset
        for row in crop_rows:
            row["observation_index"] = int(row["observation_index"]) + crop_offset
        futures.extend(future_rows)
        boundaries.extend(boundary_rows)
        crops.extend(crop_rows)
    return {
        "model": model,
        "block_id": block_id,
        "parameter_key": parameters.key,
        "main_complete": True,
        "attempt": attempt,
        "future_rows": futures,
        "boundary_rows": boundaries,
        "crop_rows": crops,
        "history_digest": canonical_digest(history),
        "future_digest": _block_digest(futures),
        "boundary_digest": _block_digest(boundaries),
    }


def _match_stage_strangers(
    paths: dict[str, Path], stage: str, profile_name: str, model: str
) -> dict[str, int]:
    """Match each parent to a size-compatible child from another world block."""

    model_root = paths["work"] / stage / model
    frames: dict[int, tuple[Path, pd.DataFrame]] = {}
    for boundary_path in sorted(model_root.glob("block_*_boundaries.csv")):
        try:
            frame = pd.read_csv(boundary_path, float_precision="round_trip")
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame(columns=BOUNDARY_COLUMNS)
        block_id = int(boundary_path.name.split("_")[1])
        frames[block_id] = (boundary_path, frame)

    matched = 0
    unmatched = 0
    group_keys = sorted(
        {
            (int(row["landmark"]), int(row["boundary"]))
            for _, frame in frames.values()
            for _, row in frame[["landmark", "boundary"]].drop_duplicates().iterrows()
        }
    )
    for landmark, boundary in group_keys:
        observations: list[dict[str, Any]] = []
        for block_id, (_, frame) in frames.items():
            selected = frame[
                (frame["landmark"] == landmark) & (frame["boundary"] == boundary)
            ]
            if selected.empty:
                continue
            archive_path = model_root / f"block_{block_id:04d}_crops.npz"
            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)
            with np.load(archive_path, allow_pickle=False) as archive:
                crop_count = int(archive["observation_group"].shape[0])
                if crop_count != int(frame.shape[0]):
                    raise ValueError(f"crop/table length mismatch in block {block_id}")
                cache: dict[str, np.ndarray] = {}
                for row_index, row in selected.iterrows():
                    observation_index = int(row["observation_index"])
                    observations.append(
                        {
                            "block_id": block_id,
                            "row_index": int(row_index),
                            "parent_size": int(row["parent_size"]),
                            "child_size": int(row["child_size"]),
                            "parent_crop": _archive_raster(
                                archive, "parent", observation_index, cache
                            ),
                            "child_crop": _archive_raster(
                                archive, "child", observation_index, cache
                            ),
                        }
                    )
        assign_matched_strangers(
            observations,
            seed_parts=(
                stage, "strangers", profile_name, model, landmark, boundary,
            ),
            different_key="block_id",
        )
        for item in observations:
            value = float(item["stranger_similarity"])
            frame = frames[int(item["block_id"])][1]
            frame.at[int(item["row_index"]), "stranger_similarity"] = value
            if np.isfinite(value):
                matched += 1
            else:
                unmatched += 1

    for block_id, (boundary_path, frame) in frames.items():
        _write_csv(boundary_path, frame.loc[:, list(BOUNDARY_COLUMNS)])
        manifest_path = model_root / f"block_{block_id:04d}.json"
        manifest = _read_json(manifest_path)
        manifest["boundary_file_sha256"] = sha256_file(boundary_path)
        manifest["boundary_digest"] = canonical_digest(frame.to_dict(orient="records"))
        crop_path = model_root / f"block_{block_id:04d}_crops.npz"
        manifest["crop_file_sha256"] = sha256_file(crop_path)
        manifest["strangers_from_different_blocks"] = True
        _write_json(manifest_path, manifest)
    return {"matched_boundaries": matched, "unmatched_boundaries": unmatched}


def _stage_models(
    stage: str,
    profile_name: str,
    models: Sequence[str],
    workers: int,
) -> dict[str, Any]:
    if stage not in {"pilot", "confirmation"}:
        raise ValueError(f"unknown campaign stage: {stage}")
    _verify_sources(profile_name)
    paths = roots(profile_name)
    _verify_checksums(paths["protocol"])
    if stage == "pilot" and (paths["output"] / "pilot_eligibility.json").is_file():
        raise ValueError("the pilot is sealed and cannot be changed after pilot-report")
    if stage == "confirmation" and (paths["output"] / "primary_metrics.json").is_file():
        raise ValueError("the confirmation is sealed and cannot be changed after report")
    if stage == "confirmation" and not (
        paths["protocol"] / "confirmation_registration.json"
    ).is_file():
        raise ValueError("confirmation registration is required")
    profile = profile_named(profile_name)
    thresholds = _read_json(paths["protocol"] / "thresholds.json")
    if stage == "pilot" and not all(
        model in thresholds.get("models", {}) for model in MODEL_NAMES
    ):
        raise ValueError("both model calibration dispositions must be frozen before pilot")
    mechanics = _read_json(paths["protocol"] / "mechanics_summary.json")
    _verify_work_files(paths)
    branches = profile.pilot_branches if stage == "pilot" else profile.confirmation_branches
    block_count = profile.pilot_blocks if stage == "pilot" else profile.confirmation_blocks
    stage_root = paths["work"] / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    stage_summary_path = stage_root / "stage_summary.json"
    summary: dict[str, Any] = (
        _read_json(stage_summary_path)
        if stage_summary_path.is_file()
        else {"format": FORMAT, "profile": profile_name, "stage": stage, "models": {}}
    )
    for model in models:
        threshold_info = thresholds.get("models", {}).get(model, {})
        viable = mechanics["viable_parameters"].get(model, [])
        if not threshold_info.get("eligible") or not viable:
            summary["models"][model] = {"generated": False, "reason": "model not calibration eligible"}
            continue
        jobs: list[tuple[str, str, int, dict[str, Any], Any, float, int]] = []
        completed_results: list[dict[str, Any]] = []
        model_root = stage_root / model
        model_root.mkdir(parents=True, exist_ok=True)
        for block_id in range(block_count):
            manifest_path = model_root / f"block_{block_id:04d}.json"
            futures_path = model_root / f"block_{block_id:04d}_futures.csv"
            boundaries_path = model_root / f"block_{block_id:04d}_boundaries.csv"
            crops_path = model_root / f"block_{block_id:04d}_crops.npz"
            if all(path.is_file() for path in (manifest_path, futures_path, boundaries_path, crops_path)):
                completed_results.append(_read_json(manifest_path))
                continue
            payload = viable[block_id % len(viable)]
            jobs.append((stage, model, block_id, payload, profile, float(threshold_info["threshold"]), branches))
        for result in _parallel_results(_simulate_block_job, jobs, workers):
            block_id = int(result["block_id"])
            future_rows = result.pop("future_rows")
            boundary_rows = result.pop("boundary_rows")
            crop_rows = result.pop("crop_rows")
            futures_path = model_root / f"block_{block_id:04d}_futures.csv"
            boundaries_path = model_root / f"block_{block_id:04d}_boundaries.csv"
            crops_path = model_root / f"block_{block_id:04d}_crops.npz"
            _write_csv(futures_path, pd.DataFrame(future_rows, columns=FUTURE_COLUMNS))
            _write_csv(boundaries_path, pd.DataFrame(boundary_rows, columns=BOUNDARY_COLUMNS))
            _write_crop_archive(crops_path, crop_rows)
            result["unmatched_boundary_digest"] = result.pop("boundary_digest")
            result["future_file_sha256"] = sha256_file(futures_path)
            result["boundary_file_sha256"] = sha256_file(boundaries_path)
            result["crop_file_sha256"] = sha256_file(crops_path)
            _write_json(model_root / f"block_{block_id:04d}.json", result)
            completed_results.append(result)
        stranger_summary = _match_stage_strangers(paths, stage, profile_name, model)
        manifest_paths = sorted(model_root.glob("block_[0-9][0-9][0-9][0-9].json"))
        summary["models"][model] = {
            "generated": True,
            "blocks_expected": block_count,
            "blocks_complete": sum(bool(item.get("main_complete")) for item in completed_results),
            "blocks_checkpointed": len(completed_results),
            "stranger_control": stranger_summary,
            "block_manifest_digest": canonical_digest(
                {path.name: sha256_file(path) for path in manifest_paths}
            ),
        }
    _write_json(stage_summary_path, summary)
    return summary


def _read_stage_tables(paths: dict[str, Path], stage: str, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_root = paths["work"] / stage / model
    future_files = sorted(model_root.glob("block_*_futures.csv"))
    boundary_files = sorted(model_root.glob("block_*_boundaries.csv"))
    def read_many(files: Sequence[Path], columns: Sequence[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for path in files:
            try:
                frames.append(pd.read_csv(path, float_precision="round_trip"))
            except pd.errors.EmptyDataError:
                frames.append(pd.DataFrame(columns=columns))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

    futures = read_many(future_files, FUTURE_COLUMNS)
    boundaries = read_many(boundary_files, BOUNDARY_COLUMNS)
    return futures, boundaries


def pilot_report(profile_name: str) -> dict[str, Any]:
    _verify_sources(profile_name)
    paths = roots(profile_name)
    _verify_checksums(paths["protocol"])
    profile = profile_named(profile_name)
    stage_summary_path = paths["work"] / "pilot" / "stage_summary.json"
    if not stage_summary_path.is_file():
        raise FileNotFoundError("completed pilot stage is required")
    stage_summary = _read_json(stage_summary_path)
    if not all(model in stage_summary.get("models", {}) for model in MODEL_NAMES):
        raise ValueError("both preregistered model dispositions are required before pilot report")
    for model, disposition in stage_summary["models"].items():
        if disposition.get("generated") and (
            disposition.get("blocks_checkpointed") != disposition.get("blocks_expected")
        ):
            raise ValueError(f"pilot stage is incomplete for {model}")
    _verify_work_files(paths)
    summary: dict[str, Any] = {
        "format": FORMAT,
        "profile": profile_name,
        "developmental_only": True,
        "mandatory_stop_after_this_report": True,
        "pilot_stage_summary_sha256": sha256_file(stage_summary_path),
        "models": {},
    }
    lines = ["# Developmental CA substrate-transfer pilot", "", "These outcomes are developmental and have no confirmation evidential status.", ""]
    for model in MODEL_NAMES:
        futures, _ = _read_stage_tables(paths, "pilot", model)
        result = pilot_eligibility(futures, profile.pilot_blocks, profile)
        summary["models"][model] = result
        lines.extend(
            [
                f"## {model}",
                "",
                f"- Confirmation eligible: **{result['eligible']}**",
                f"- Complete blocks: {result['complete_blocks']}",
                f"- Complete-horizon fraction: {result['complete_horizon_fraction']:.6f}",
                f"- Break futures: {result['breaks']}",
                f"- F12 events: {result['events']} across {result['event_blocks']} blocks",
                "",
            ]
        )
    lines.extend(["## Stop gate", "", "Confirmation is not authorized by this pilot report. A later explicit research instruction is required.", ""])
    _write_json(paths["output"] / "pilot_eligibility.json", summary)
    _write_text(paths["output"] / "PILOT_REPORT.md", "\n".join(lines))
    _write_checksums(paths["output"])
    return summary


def register_confirmation(profile_name: str) -> dict[str, Any]:
    if profile_name != "full":
        raise ValueError("smoke profiles can never authorize confirmation")
    _verify_sources(profile_name)
    paths = roots(profile_name)
    _verify_checksums(paths["protocol"])
    pilot_path = paths["output"] / "pilot_eligibility.json"
    if not pilot_path.is_file():
        raise FileNotFoundError("sealed pilot report is required")
    _verify_checksums(paths["output"])
    _verify_work_files(paths)
    registration_path = paths["protocol"] / "confirmation_registration.json"
    if registration_path.exists():
        registration = _read_json(registration_path)
        if registration.get("pilot_sha256") != sha256_file(pilot_path):
            raise ValueError("sealed pilot report differs from confirmation registration")
        return registration
    pilot = _read_json(pilot_path)
    stage_summary_path = paths["work"] / "pilot" / "stage_summary.json"
    if pilot.get("pilot_stage_summary_sha256") != sha256_file(stage_summary_path):
        raise ValueError("pilot work changed after its sealed report")
    eligible = [model for model, result in pilot["models"].items() if result["eligible"]]
    registration = {
        "format": FORMAT,
        "pilot_sha256": sha256_file(pilot_path),
        "eligible_models": eligible,
        "confirmation_blocks": 128,
        "landmarks": [20, 35, 50, 65, 80],
        "branches": 64,
        "explicit_registration_command_is_authorization": True,
        "closed_without_eligible_model": not bool(eligible),
    }
    registration["registration_digest"] = canonical_digest(registration)
    _write_json(registration_path, registration)
    _write_checksums(paths["protocol"])
    return registration


def confirmation_report(profile_name: str) -> dict[str, Any]:
    if profile_name != "full":
        raise ValueError("smoke profiles cannot produce scientific reports")
    _verify_sources(profile_name)
    paths = roots(profile_name)
    _verify_checksums(paths["protocol"])
    registration = _read_json(paths["protocol"] / "confirmation_registration.json")
    stage_summary_path = paths["work"] / "confirmation" / "stage_summary.json"
    if not stage_summary_path.is_file():
        raise FileNotFoundError("completed confirmation stage is required")
    stage_summary = _read_json(stage_summary_path)
    if not all(model in stage_summary.get("models", {}) for model in registration["eligible_models"]):
        raise ValueError("every eligible model requires a confirmation disposition")
    for model in registration["eligible_models"]:
        disposition = stage_summary["models"][model]
        if disposition.get("blocks_checkpointed") != disposition.get("blocks_expected"):
            raise ValueError(f"confirmation stage is incomplete for {model}")
    _verify_work_files(paths)
    thresholds = _read_json(paths["protocol"] / "thresholds.json")
    pilot = _read_json(paths["output"] / "pilot_eligibility.json")
    results: dict[str, Any] = {}
    eligible_models = set(registration["eligible_models"])
    for model in MODEL_NAMES:
        if model not in eligible_models:
            results[model] = {
                "model": model,
                "confirmation_entered": False,
                "reason": "not pilot eligible",
                "pilot_eligibility": pilot["models"].get(model, {}),
                "model_passed": False,
            }
            continue
        futures, boundaries = _read_stage_tables(paths, "confirmation", model)
        result = confirmation_metrics(
            model,
            futures,
            boundaries,
            profile_named(profile_name),
            float(thresholds["models"][model]["threshold"]),
        )
        result["confirmation_entered"] = True
        results[model] = result
    passing = [model for model, result in results.items() if result["model_passed"]]
    verdict = {
        "at_least_one_ca_model_passed": bool(passing),
        "both_ca_models_passed": set(passing) == set(MODEL_NAMES),
        "passing_models": passing,
        "models_pooled": False,
        "confirmation_registration_digest": registration["registration_digest"],
        "confirmation_stage_summary_sha256": sha256_file(stage_summary_path),
        "models": results,
    }
    lines = [
        "# Prospective CA substrate-transfer confirmation",
        "",
        f"At least one complete model gate passed: **{verdict['at_least_one_ca_model_passed']}**.",
        "",
        "The models were adjudicated separately at one-sided alpha 0.025 and were not pooled.",
        "",
    ]
    for model, result in results.items():
        lines.extend([f"## {model}", ""])
        if not result["confirmation_entered"]:
            lines.extend(["Not entered into confirmation because it failed the developmental eligibility gate.", ""])
            continue
        fidelity = result["fidelity"]
        dependence = result["state_dependence"]
        lines.extend(
            [
                f"- Complete model gate: **{result['model_passed']}**",
                f"- Parent-minus-stranger mean: {fidelity['mean_similarity_difference']}",
                f"- Fidelity lower 97.5% bound: {fidelity['ci97_5_one_sided_lower']}",
                f"- Nondegenerate event gate: {result['nondegenerate_event_gate']['passed']}",
                f"- A-half prevalence: {result['halves']['A']['prevalence']:.6f}",
                f"- B-half prevalence: {result['halves']['B']['prevalence']:.6f}",
                f"- State-dependence rho: {dependence['estimate']:.6f}",
                f"- State-dependence lower 97.5% bound: {dependence['ci97_5_one_sided_lower']:.6f}",
                "",
            ]
        )
    lines.extend([
        "## Claim boundary", "",
        "A positive result is limited to the tested CA contract(s). It does not establish universality, life, biological memory, Phi/PhiID, real chemistry, or a shared GARD mechanism.", "",
    ])
    _write_json(paths["output"] / "primary_metrics.json", verdict)
    _write_text(paths["output"] / "SCIENTIFIC_REPORT.md", "\n".join(lines))
    _write_checksums(paths["output"])
    return verdict


def _verify_work_files(paths: dict[str, Path]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    mechanics_path = paths["protocol"] / "mechanics_summary.json"
    if mechanics_path.is_file():
        mechanics = _read_json(mechanics_path)
        target = paths["protocol"] / "mechanics_trials.csv"
        checks["mechanics_trials.csv"] = (
            target.is_file()
            and sha256_file(target) == mechanics.get("mechanics_trials_sha256")
        )
        checkpoint_root = paths["work"] / "mechanics"
        checkpoint_files = sorted(checkpoint_root.glob("**/*.json"))
        current_digest = canonical_digest(
            {
                str(path.relative_to(checkpoint_root)): sha256_file(path)
                for path in checkpoint_files
            }
        )
        checks["mechanics/checkpoints"] = bool(
            checkpoint_files
            and current_digest == mechanics.get("mechanics_checkpoint_digest")
        )
    thresholds_path = paths["protocol"] / "thresholds.json"
    if thresholds_path.is_file():
        thresholds = _read_json(thresholds_path)
        for model, info in thresholds.get("models", {}).items():
            expected = info.get("calibration_file_sha256")
            if expected is None:
                continue
            target = paths["work"] / "calibration" / f"{model}_pairs.csv"
            checks[f"calibration/{model}_pairs.csv"] = (
                target.is_file() and sha256_file(target) == expected
            )
    checkpoint_root = paths["work"] / "calibration" / "checkpoints"
    for manifest_path in sorted(
        checkpoint_root.glob("*/block_[0-9][0-9][0-9][0-9].json")
    ):
        manifest = _read_json(manifest_path)
        stem = manifest_path.stem
        table_path = manifest_path.with_name(f"{stem}_observations.csv")
        crop_path = manifest_path.with_name(f"{stem}_crops.npz")
        relative = manifest_path.relative_to(checkpoint_root)
        checks[f"calibration/checkpoint/{relative}:table"] = bool(
            table_path.is_file()
            and sha256_file(table_path) == manifest.get("table_sha256")
        )
        checks[f"calibration/checkpoint/{relative}:crop"] = bool(
            crop_path.is_file()
            and sha256_file(crop_path) == manifest.get("crop_sha256")
        )
    for stage in ("pilot", "confirmation"):
        for model in MODEL_NAMES:
            model_root = paths["work"] / stage / model
            for manifest_path in sorted(model_root.glob("block_[0-9][0-9][0-9][0-9].json")):
                manifest = _read_json(manifest_path)
                stem = manifest_path.stem
                targets = {
                    "future": (model_root / f"{stem}_futures.csv", manifest.get("future_file_sha256")),
                    "boundary": (model_root / f"{stem}_boundaries.csv", manifest.get("boundary_file_sha256")),
                    "crop": (model_root / f"{stem}_crops.npz", manifest.get("crop_file_sha256")),
                }
                for kind, (target, expected) in targets.items():
                    checks[f"{stage}/{model}/{stem}:{kind}"] = bool(
                        expected and target.is_file() and sha256_file(target) == expected
                    )
    if checks and not all(checks.values()):
        raise ValueError(
            f"work-file checksum failure: {[key for key, value in checks.items() if not value]}"
        )
    return checks


def verify(profile_name: str, full_replay: bool) -> dict[str, Any]:
    _verify_sources(profile_name)
    paths = roots(profile_name)
    protocol_checks = _verify_checksums(paths["protocol"])
    output_checks = _verify_checksums(paths["output"])
    work_checks = _verify_work_files(paths)
    replay: dict[str, Any] = {"requested": full_replay, "blocks_checked": 0, "all_match": True}
    if full_replay:
        registration_path = paths["protocol"] / "confirmation_registration.json"
        if not registration_path.is_file():
            raise FileNotFoundError("full scientific replay requires confirmation registration")
        registration = _read_json(registration_path)
        profile = profile_named(profile_name)
        thresholds = _read_json(paths["protocol"] / "thresholds.json")
        mechanics = _read_json(paths["protocol"] / "mechanics_summary.json")
        for model in registration["eligible_models"]:
            viable = mechanics["viable_parameters"][model]
            for block_id in range(profile.confirmation_blocks):
                manifest_path = paths["work"] / "confirmation" / model / f"block_{block_id:04d}.json"
                frozen = _read_json(manifest_path)
                job = (
                    "confirmation", model, block_id, viable[block_id % len(viable)], profile,
                    float(thresholds["models"][model]["threshold"]), profile.confirmation_branches,
                )
                regenerated = _simulate_block_job(job)
                replay["blocks_checked"] += 1
                if (
                    regenerated.get("future_digest") != frozen.get("future_digest")
                    or regenerated.get("boundary_digest")
                    != frozen.get("unmatched_boundary_digest")
                ):
                    replay["all_match"] = False
                    replay.setdefault("failed", []).append(f"{model}:{block_id}")
    result = {
        "format": FORMAT,
        "profile": profile_name,
        "protocol_checks": protocol_checks,
        "output_checks": output_checks,
        "work_checks": work_checks,
        "replay": replay,
        "complete": bool(
            protocol_checks and output_checks and work_checks and replay["all_match"]
        ),
    }
    _write_json(paths["verification"] / "verification_audit.json", result)
    _write_checksums(paths["verification"])
    return result


def status(profile_name: str) -> dict[str, Any]:
    paths = roots(profile_name)
    profile = profile_named(profile_name)
    output: dict[str, Any] = {"profile": profile_name, "prepared": (paths["protocol"] / "registration.json").is_file()}
    thresholds_path = paths["protocol"] / "thresholds.json"
    threshold_models = (
        _read_json(thresholds_path).get("models", {})
        if thresholds_path.is_file() else {}
    )
    output["mechanics"] = {}
    output["calibration"] = {}
    for model in MODEL_NAMES:
        output["mechanics"][model] = {
            "checkpoints": len(
                list((paths["work"] / "mechanics" / model).glob("*.json"))
            ),
            "expected": len(mechanics_jobs(model, profile)),
        }
        output["calibration"][model] = {
            "block_checkpoints": len(
                list(
                    (paths["work"] / "calibration" / "checkpoints" / model).glob(
                        "block_[0-9][0-9][0-9][0-9].json"
                    )
                )
            ),
            "blocks_expected": profile.calibration_blocks,
            "threshold_disposition_frozen": model in threshold_models,
        }
    for stage in ("pilot", "confirmation"):
        output[stage] = {}
        for model in MODEL_NAMES:
            root = paths["work"] / stage / model
            output[stage][model] = {
                "block_manifests": len(list(root.glob("block_*.json"))) if root.exists() else 0,
                "future_tables": len(list(root.glob("block_*_futures.csv"))) if root.exists() else 0,
            }
    output["pilot_reported"] = (paths["output"] / "pilot_eligibility.json").is_file()
    output["confirmation_registered"] = (paths["protocol"] / "confirmation_registration.json").is_file()
    output["confirmation_reported"] = (paths["output"] / "primary_metrics.json").is_file()
    return output


def _models(value: str) -> list[str]:
    if value == "all":
        return list(MODEL_NAMES)
    if value in MODEL_NAMES:
        return [value]
    if value == "eligible":
        return []
    raise ValueError(f"unknown model selector: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare", "validate", "calibrate", "pilot", "pilot-report", "status",
        "register-confirmation", "confirm", "report", "verify",
    ))
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--model", choices=("all", "protocell", "evoloop", "eligible"), default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--full-replay", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")

    if args.command == "prepare":
        result = prepare(args.profile)
    elif args.command == "validate":
        result = validation_checks(args.profile)
    elif args.command == "calibrate":
        result = calibrate(args.profile, _models(args.model), args.workers)
    elif args.command == "pilot":
        result = _stage_models("pilot", args.profile, _models(args.model), args.workers)
    elif args.command == "pilot-report":
        result = pilot_report(args.profile)
    elif args.command == "status":
        result = status(args.profile)
    elif args.command == "register-confirmation":
        result = register_confirmation(args.profile)
    elif args.command == "confirm":
        if args.profile != "full":
            raise ValueError("smoke profiles cannot run confirmation")
        registration = _read_json(roots(args.profile)["protocol"] / "confirmation_registration.json")
        requested = registration["eligible_models"] if args.model == "eligible" else _models(args.model)
        unauthorized = sorted(set(requested) - set(registration["eligible_models"]))
        if unauthorized:
            raise ValueError(f"models not confirmation eligible: {unauthorized}")
        result = _stage_models("confirmation", args.profile, requested, args.workers)
    elif args.command == "report":
        result = confirmation_report(args.profile)
    elif args.command == "verify":
        result = verify(args.profile, args.full_replay)
    else:
        raise AssertionError(args.command)
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
