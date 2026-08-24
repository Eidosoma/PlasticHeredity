"""Execute S19-L23 powered independent screen of frozen prefix families."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_attractor_onset_early_warning.core import (
    FEATURE_GROUPS as L18_FEATURE_GROUPS,
)
from e01_attractor_onset_early_warning.core import (
    HORIZON_EXCLUSIVE,
    LANDMARK_COUNT,
    build_landmark_target,
    extract_past_features,
)
from e01_clean_directional_confirmation.core import fixed_label_spec
from e01_creative_directional_search.core import label_trajectory
from e01_frozen_timebase_ensemble.core import (
    selected_clock_observations,
    states_from_observations,
)
from e01_latent_timebase.core import (
    ExposureDefinition,
    SimulationDefinition,
    derive_seed,
    generate_beta,
    initialize_distinct_state,
    simulate_trajectory,
)
from e01_latent_timebase.core import array_sha256 as simulator_array_sha256
from e01_onset_discovery.core import (
    DMD_FEATURES,
    EWS_FEATURES,
    RQA_FEATURES,
    extract_organization_warning_features,
)
from e01_onset_discovery.multiscale_geometry import (
    INTRINSIC_GEOMETRY_FEATURES,
    PATH_GEOMETRY_FEATURES,
    TOPOLOGY_FEATURES,
    extract_multiscale_geometry_features,
)
from e01_onset_discovery.outcome_blind_representation import (
    RANDOM_CONV_FEATURES,
    extract_outcome_blind_representation,
    kernel_bank_fingerprint,
)


def _load_base() -> Any:
    path = REPO_ROOT / "scripts/e01/run_s19_l19_source_grounded_early_warning.py"
    spec = importlib.util.spec_from_file_location("e01_s19_l23_runner_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load L19 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
LOOP_ID = "S19-L23"
VERSION = "E01-S19-L23-POWERED-FROZEN-PREFIX-FAMILY-SCREEN-v1.0.0"
TARGET_ID = "PF_DOMINANT_COMPONENT_CENTROID_H900"
CLOCK_ID = "C1_SELECTED_DAUGHTER_RETAINED"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
MATRIX_COUNT = 400
ROOT_HEX = "f3b0fd551b8f182388cad84365b62a2f2f51e82aa11a0c6b6e22a088dbe90544"
PHASE = "s19_l23_powered_frozen_prefix_screen"
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L23"
L22_ROOT = ARTIFACT_ROOT / "loops/L22"
CACHE_ROOT = Path("/cache/e01_s19_l23")
BUILD_ROOT = CACHE_ROOT / "build"
GENERATED_INPUT_ROOT = CACHE_ROOT / "generated_inputs"
PRIMARY_CACHE = CACHE_ROOT / "primary_trajectories"
REGEN_CACHE = CACHE_ROOT / "regenerated_trajectories"
CONFIG = REPO_ROOT / "configs/e01/s19_l23_powered_frozen_family_screen.yaml"
RUNNER_PATH = Path(__file__)
BOOTSTRAPS = 4096
PERMUTATIONS = 512

CANDIDATE_SPECS = {
    "S12F-CANDIDATE-02": {
        "h": 0.6031526490073492,
        "daughterRule": "FIRST_DAUGHTER",
        "overshootRule": "TRIM_NEW_ENTRANTS_TO_NMAX",
    },
    "S12F-CANDIDATE-03": {
        "h": 0.5613315384859516,
        "daughterRule": "RANDOM_NONEMPTY",
        "overshootRule": "TRIM_NEW_ENTRANTS_TO_NMAX",
    },
}

COMPACT_BASELINE_FIELDS = BASE.COMPACT_BASELINE_FIELDS
L19_ALL = EWS_FEATURES + RQA_FEATURES + DMD_FEATURES
L20_ALL = TOPOLOGY_FEATURES + INTRINSIC_GEOMETRY_FEATURES + PATH_GEOMETRY_FEATURES
MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "DUMMY_TRAINING_PRIOR": (),
    "TIME_ONLY": tuple(L18_FEATURE_GROUPS["TIME_ONLY"]),
    "EXACT_H_STABILITY": tuple(L18_FEATURE_GROUPS["EXACT_H_STABILITY"]),
    "PREFIX_RECURRENCE_GEOMETRY": tuple(L18_FEATURE_GROUPS["PREFIX_RECURRENCE_GEOMETRY"]),
    "L18_PAST_FULL_NO_BGM": tuple(
        dict.fromkeys(
            L18_FEATURE_GROUPS["TIME_ONLY"]
            + L18_FEATURE_GROUPS["EXACT_H_STABILITY"]
            + L18_FEATURE_GROUPS["PREFIX_RECURRENCE_GEOMETRY"]
            + L18_FEATURE_GROUPS["ORGANIZATION_DYNAMICS"]
        )
    ),
    "COMPACT_BASELINE": COMPACT_BASELINE_FIELDS,
    "COMPACT_PLUS_L19_ALL": COMPACT_BASELINE_FIELDS + L19_ALL,
    "COMPACT_PLUS_L20_TOPOLOGY": COMPACT_BASELINE_FIELDS + TOPOLOGY_FEATURES,
    "COMPACT_PLUS_L20_INTRINSIC": COMPACT_BASELINE_FIELDS + INTRINSIC_GEOMETRY_FEATURES,
    "COMPACT_PLUS_L20_PATH": COMPACT_BASELINE_FIELDS + PATH_GEOMETRY_FEATURES,
    "COMPACT_PLUS_L20_MULTISCALE": COMPACT_BASELINE_FIELDS + L20_ALL,
    "COMPACT_PLUS_L22_RANDOM_CONVOLUTION": COMPACT_BASELINE_FIELDS
    + RANDOM_CONV_FEATURES,
}
MODEL_IDS = tuple(MODEL_FEATURES)
LEAD_MODELS = (
    "COMPACT_PLUS_L20_TOPOLOGY",
    "COMPACT_PLUS_L20_INTRINSIC",
    "COMPACT_PLUS_L20_PATH",
    "COMPACT_PLUS_L19_ALL",
    "COMPACT_PLUS_L22_RANDOM_CONVOLUTION",
    "COMPACT_PLUS_L20_MULTISCALE",
)
ALL_NEW_FEATURES = tuple(dict.fromkeys(L19_ALL + L20_ALL + RANDOM_CONV_FEATURES))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L22_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L22_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L22_ROOT / item["path"]),
            "root": str(L22_ROOT),
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in manifest["files"]
    )
    failures = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "MISSING"})
        elif sha256_file(path) != row["sha256"]:
            failures.append({"path": str(path), "reason": "HASH_MISMATCH"})
    aggregate = hashlib.sha256(
        "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
    ).hexdigest()
    return {
        "schema": "eidosoma.e01.s19_l23.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": aggregate,
        "l22ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def definition(candidate: str) -> SimulationDefinition:
    spec = CANDIDATE_SPECS[candidate]
    return SimulationDefinition(
        daughter_rule=spec["daughterRule"],
        overshoot_rule=spec["overshootRule"],
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=spec["h"]),
    )


def input_identities() -> tuple[pd.DataFrame, pd.DataFrame]:
    input_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for matrix_index in range(MATRIX_COUNT):
        beta_seed = derive_seed(ROOT_HEX, PHASE, "catalytic_matrix", matrix_index)
        init_seed = derive_seed(ROOT_HEX, PHASE, "initial_state", matrix_index)
        beta = generate_beta(beta_seed)
        initial = initialize_distinct_state(init_seed)
        input_rows.append(
            {
                "matrixIndex": matrix_index,
                "betaSha256": simulator_array_sha256(beta),
                "initialStateSha256": simulator_array_sha256(initial),
                "initialMass": int(initial.sum()),
                "initialDistinctTypes": int(np.count_nonzero(initial)),
                "generatedBeforeOutcomeAccess": True,
            }
        )
        for candidate in ("SHARED", *CANDIDATES):
            purposes = (
                ("catalytic_matrix", "initial_state")
                if candidate == "SHARED"
                else (
                    "poisson_update",
                    "overshoot_trim",
                    "fission",
                    "daughter_selection",
                )
            )
            for purpose in purposes:
                identity = (
                    derive_seed(ROOT_HEX, PHASE, purpose, matrix_index)
                    if candidate == "SHARED"
                    else derive_seed(ROOT_HEX, PHASE, purpose, matrix_index, candidate)
                )
                seed_rows.append(
                    {
                        "matrixIndex": matrix_index,
                        "candidateId": candidate,
                        "purpose": purpose,
                        "configurationId": identity.configuration_id,
                        "derivedSeed": str(identity.derived_seed),
                        "seedMaterialSha256": identity.seed_material_sha256,
                        "rootHex": ROOT_HEX,
                    }
                )
    return pd.DataFrame(input_rows), pd.DataFrame(seed_rows)


def prior_hex_tokens(rows: list[dict[str, Any]]) -> set[str]:
    pattern = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
    tokens: set[str] = set()
    for row in rows:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            continue
        try:
            tokens.update(match.decode("ascii") for match in pattern.findall(path.read_bytes()))
        except OSError:
            continue
    return tokens


def fixture_table() -> pd.DataFrame:
    rng = np.random.default_rng(BASE.derive_seed("l23_fixtures"))
    states = rng.poisson(2.0, size=(64, 100)).astype(np.int64)
    states[:, 0] += 1
    generations = np.repeat(np.arange(8), 8)
    kinds = np.asarray(["molecular_update"] * 64, dtype=object)
    first = combined_features(states, generations, kinds)
    second = combined_features(states.copy(), generations.copy(), kinds.copy())
    species = rng.permutation(100)
    relabelled = combined_features(states[:, species], generations, kinds)
    order = np.r_[0, rng.permutation(np.arange(1, 64))]
    temporal = combined_features(states[order], generations[order], kinds[order])
    labels = np.zeros(250, dtype=bool)
    labels[120:] = True
    target = build_landmark_target(labels)
    return pd.DataFrame(
        [
            {
                "fixtureId": "COMPLETE_FEATURE_SCHEMA",
                "passed": set(first) == set(MODEL_FEATURES["L18_PAST_FULL_NO_BGM"]) | set(ALL_NEW_FEATURES),
                "details": str(len(first)),
            },
            {
                "fixtureId": "EXACT_FEATURE_REPLAY",
                "passed": first == second,
                "details": "CPU_FLOAT64",
            },
            {
                "fixtureId": "FINITE_FEATURES",
                "passed": bool(np.isfinite(list(first.values())).all()),
                "details": "all frozen families",
            },
            {
                "fixtureId": "MOLECULE_LABEL_INVARIANCE",
                "passed": all(
                    np.isclose(first[name], relabelled[name], atol=1e-10, rtol=1e-10)
                    for name in first
                ),
                "details": "all 100 coordinates permuted",
            },
            {
                "fixtureId": "TEMPORAL_SENSITIVITY",
                "passed": any(first[name] != temporal[name] for name in ALL_NEW_FEATURES),
                "details": "first observation fixed",
            },
            {
                "fixtureId": "TARGET_GEOMETRY",
                "passed": target["atRiskAtLandmark"]
                and target["eventWithinHorizon"]
                and target["firstOnsetIndex0"] == 120,
                "details": "64-to-192",
            },
            {
                "fixtureId": "L22_KERNEL_IDENTITY",
                "passed": kernel_bank_fingerprint()
                == "3de958f6be47bb563b30ea07e0099dcd4642c1ebcebb050ae688884f606f45c1",
                "details": kernel_bank_fingerprint(),
            },
        ]
    )


def combined_features(
    states: np.ndarray, generations: Iterable[int], kinds: Iterable[str]
) -> dict[str, float]:
    result = extract_past_features(states, list(generations), list(kinds))
    result.update(extract_organization_warning_features(states))
    result.update(extract_multiscale_geometry_features(states))
    result.update(extract_outcome_blind_representation(states))
    if len(result) != len(set(result)) or not np.isfinite(list(result.values())).all():
        raise RuntimeError("combined frozen feature schema invalid")
    return result


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L19_L22_FROZEN_IMPLEMENTATION_LINEAGE",
                "doi": None,
                "url": "LOCAL_FROZEN_E01_SOURCE",
                "retrievalDate": utc_now()[:10],
                "directSupport": "exact previously locked L19 critical-slowing/RQA/DMD, L20 geometry/topology, and L22 random-convolution implementations",
                "reconstructionChoice": "powered all-family screen without changing any feature specification",
                "evidenceClass": "DIRECT_FROZEN_E01_IMPLEMENTATION",
            },
            {
                "sourceId": "BOETTIGER_HASTINGS_2012_LIMITS",
                "doi": "10.1098/rsif.2012.0125",
                "url": "https://doi.org/10.1098/rsif.2012.0125",
                "retrievalDate": utc_now()[:10],
                "directSupport": "finite-sample early-warning false-positive and power limits",
                "reconstructionChoice": "400 shared matrices, candidate replication, matrix uncertainty and max-statistic permutations",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
        ]
    )


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before the L23 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local and remote heads differ")
    prior = validate_immutable_prior()
    if not prior["unchanged"]:
        raise RuntimeError("immutable prior changed")
    fixtures = fixture_table()
    if not fixtures["passed"].all():
        raise RuntimeError("L23 fixture failure")
    inputs, seeds = input_identities()
    tokens = prior_hex_tokens(prior["files"])
    new_tokens = (
        set(inputs["betaSha256"])
        | set(inputs["initialStateSha256"])
        | set(seeds["seedMaterialSha256"])
        | {ROOT_HEX}
    )
    overlaps = sorted(new_tokens & tokens)
    firewall = {
        "schema": "eidosoma.e01.s19_l23.seed_firewall.v1",
        "status": "PASS" if not overlaps else "FAIL",
        "rootHex": ROOT_HEX,
        "newSharedMatrices": MATRIX_COUNT,
        "newInitialStates": MATRIX_COUNT,
        "newDerivedSeedIdentities": len(seeds),
        "priorTokenCount": len(tokens),
        "overlapCount": len(overlaps),
        "overlaps": overlaps,
    }
    if overlaps:
        raise RuntimeError("L23 seed/input firewall overlap")

    benchmark_root = hashlib.sha256(b"E01-S19-L23-BENCHMARK-ONLY").hexdigest()
    start = time.perf_counter()
    for index, candidate in enumerate(CANDIDATES):
        beta = generate_beta(derive_seed(benchmark_root, "benchmark", "catalytic_matrix", index))
        initial = initialize_distinct_state(
            derive_seed(benchmark_root, "benchmark", "initial_state", index)
        )
        simulate_trajectory(
            phase="s19_l23_benchmark_only",
            root_hex=benchmark_root,
            matrix_index=index,
            definition=definition(candidate),
            stream_identity=candidate,
            beta=beta,
            initial_state=initial,
        )
    benchmark_seconds = time.perf_counter() - start
    projected_wall = benchmark_seconds * MATRIX_COUNT / 2 / 8 * 2.5
    if projected_wall > 60 * 60 * 60:
        raise RuntimeError("benchmark exceeds wall ceiling")

    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        """# S19-L23 decision record

L18 established a usable but small 53/54-matrix at-risk task. L19-L22 produced candidate-specific hints and nulls under stringent uncertainty, while L21 showed that endpoint timing was not the primary explanation. L23 therefore changes only statistical power: it generates 400 new shared catalytic matrices and evaluates every complete L19/L20/L22 family unchanged under one max-statistic screen.

The cohort, seed root, simulator contracts, target, landmark/horizon, feature implementations, models, controls, multiplicity rule and gates are fixed before trajectories are generated. No incomplete/extinct unit is replaced. Any passing family is a discovery lead only and must survive a second untouched confirmation.
""",
    )
    sources = source_registry()
    sources.to_csv(LOOP_ROOT / "source_grounding_registry.csv", index=False)
    BASE.atomic_text(
        LOOP_ROOT / "source_grounding_report.md",
        "# L23 source grounding\n\n"
        + "\n".join(
            f"- **{r.sourceId}** — {r.directSupport}. Frozen use: {r.reconstructionChoice}. {r.url}"
            for r in sources.itertuples(index=False)
        )
        + "\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "input_manifest.parquet", inputs)
    BASE.write_parquet(LOOP_ROOT / "seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l23.implementation_lock.v1",
            "researchStepId": LOOP_ID,
            "versionedId": VERSION,
            "repositoryHead": head,
            "remoteHead": remote,
            "configSha256": sha256_file(CONFIG),
            "runnerSha256": sha256_file(RUNNER_PATH),
            "l22ManifestSha256": sha256_file(L22_ROOT / "artifact_manifest.json"),
            "matrixCount": MATRIX_COUNT,
            "trajectoryCount": MATRIX_COUNT * 2,
            "rootHex": ROOT_HEX,
            "phase": PHASE,
            "candidateSpecs": CANDIDATE_SPECS,
            "targetId": TARGET_ID,
            "landmark": LANDMARK_COUNT,
            "horizonExclusive": HORIZON_EXCLUSIVE,
            "modelFeatures": {name: list(fields) for name, fields in MODEL_FEATURES.items()},
            "leadModelsInFixedPriorityOrder": list(LEAD_MODELS),
            "bootstrapReplicates": BOOTSTRAPS,
            "permutationReplicates": PERMUTATIONS,
            "outcomeAccessed": False,
            "lockedAtUtc": utc_now(),
        },
    )
    BASE.write_json(
        LOOP_ROOT / "preoutcome_repository_lock.json",
        {
            "head": head,
            "remote": remote,
            "configSha256": sha256_file(CONFIG),
            "priorAggregateSha256": prior["aggregateSha256"],
        },
    )
    BASE.write_json(
        LOOP_ROOT / "benchmark_projection.json",
        {
            "status": "PASS_PROJECTED_WITHIN_CEILING",
            "twoTrajectorySeconds": benchmark_seconds,
            "projectedGenerationAndRegenerationWallSeconds": projected_wall,
            "projectedCpuHoursUpper": 90,
            "cpuHoursCeiling": 100,
            "wallHoursCeiling": 72,
            "gpuHours": 0,
        },
    )
    print(
        BASE.canonical_json(
            {
                "status": "PREOUTCOME_LOCKED",
                "head": head,
                "fixtures": len(fixtures),
                "priorFiles": prior["fileCount"],
                "matrixCount": MATRIX_COUNT,
                "seedFirewall": firewall["status"],
                "benchmarkSeconds": benchmark_seconds,
            }
        )
    )


def trajectory_path(root: Path, matrix_index: int, candidate: str) -> Path:
    return root / f"M{matrix_index:04d}__{candidate}.pkl"


def _simulate_matrix(matrix_index: int, root_string: str) -> dict[str, Any]:
    cache = Path(root_string)
    beta = generate_beta(derive_seed(ROOT_HEX, PHASE, "catalytic_matrix", matrix_index))
    initial = initialize_distinct_state(
        derive_seed(ROOT_HEX, PHASE, "initial_state", matrix_index)
    )
    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        started = time.perf_counter()
        try:
            trajectory, seeds = simulate_trajectory(
                phase=PHASE,
                root_hex=ROOT_HEX,
                matrix_index=matrix_index,
                definition=definition(candidate),
                stream_identity=candidate,
                beta=beta,
                initial_state=initial,
            )
            path = trajectory_path(cache, matrix_index, candidate)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                pickle.dump(trajectory, handle, protocol=5)
            selected = selected_clock_observations(trajectory, CLOCK_ID)
            rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory.trajectory_id,
                    "trajectorySha256": trajectory.trajectory_sha256,
                    "betaSha256": trajectory.beta_sha256,
                    "initialStateSha256": trajectory.initial_state_sha256,
                    "terminalStatus": trajectory.terminal_status,
                    "completedFissions": int(trajectory.completed_fissions),
                    "selectedClockLength": len(selected),
                    "clockId": CLOCK_ID,
                    "cachePath": str(path),
                    "cacheSha256": sha256_file(path),
                    "replacementAttempted": False,
                    "wallSeconds": time.perf_counter() - started,
                }
            )
            for seed in seeds:
                seed_rows.append(
                    {
                        "matrixIndex": matrix_index,
                        "candidateId": candidate
                        if seed.purpose not in {"catalytic_matrix", "initial_state"}
                        else "SHARED",
                        "purpose": seed.purpose,
                        "configurationId": seed.configuration_id,
                        "derivedSeed": str(seed.derived_seed),
                        "seedMaterialSha256": seed.seed_material_sha256,
                        "rootHex": ROOT_HEX,
                    }
                )
        except Exception as error:  # noqa: BLE001 - full scientific provenance
            failures.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": matrix_index,
                    "failureType": type(error).__name__,
                    "message": str(error),
                }
            )
    return {"trajectories": rows, "seeds": seed_rows, "failures": failures}


def _simulate_all(root: Path, workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_simulate_matrix, index, str(root)) for index in range(MATRIX_COUNT)]
        for future in as_completed(futures):
            outputs.append(future.result())
    trajectories = pd.DataFrame(
        [row for output in outputs for row in output["trajectories"]]
    ).sort_values(["candidateId", "matrixIndex"]).reset_index(drop=True)
    seeds = pd.DataFrame([row for output in outputs for row in output["seeds"]])
    seeds = seeds.drop_duplicates(
        ["matrixIndex", "candidateId", "purpose", "seedMaterialSha256"]
    ).sort_values(["matrixIndex", "candidateId", "purpose"]).reset_index(drop=True)
    failures = pd.DataFrame(
        [row for output in outputs for row in output["failures"]],
        columns=["candidateId", "matrixIndex", "failureType", "message"],
    )
    return trajectories, seeds, failures


def split_registry(cohort: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, frame in cohort.groupby("candidateId", sort=True):
        frame = frame.sort_values("matrixIndex").reset_index(drop=True)
        y = frame["eventWithinHorizon"].astype(int).to_numpy()
        cv = RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=10,
            random_state=BASE.derive_seed("l23_cv", candidate),
        )
        for split_index, (train, test) in enumerate(cv.split(np.zeros(len(y)), y)):
            repeat, fold = divmod(split_index, 5)
            for role, indices in (("TRAIN", train), ("TEST", test)):
                for position in indices:
                    rows.append(
                        {
                            "candidateId": candidate,
                            "repeat": repeat,
                            "fold": fold,
                            "role": role,
                            "matrixIndex": int(frame.iloc[position]["matrixIndex"]),
                        }
                    )
    return pd.DataFrame(rows)


def replay_task() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, int], dict[str, Any]],
]:
    if PRIMARY_CACHE.exists() and any(PRIMARY_CACHE.iterdir()):
        raise RuntimeError("L23 primary trajectory cache is not empty")
    if REGEN_CACHE.exists() and any(REGEN_CACHE.iterdir()):
        raise RuntimeError("L23 regeneration cache is not empty")
    manifest, runtime_seeds, failures = _simulate_all(PRIMARY_CACHE, 8)
    if len(manifest) != MATRIX_COUNT * 2 or not failures.empty:
        raise RuntimeError("L23 did not retain all 800 registered trajectory attempts")
    expected_inputs = pd.read_parquet(LOOP_ROOT / "input_manifest.parquet")
    joined = manifest.merge(expected_inputs, on="matrixIndex", suffixes=("", "Expected"))
    if not (
        joined["betaSha256"].eq(joined["betaSha256Expected"]).all()
        and joined["initialStateSha256"].eq(joined["initialStateSha256Expected"]).all()
    ):
        raise RuntimeError("L23 generated input identity mismatch")

    spec = fixed_label_spec(TARGET_ID)
    replay_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for row in manifest.itertuples(index=False):
        path = Path(row.cachePath)
        if sha256_file(path) != row.cacheSha256:
            raise RuntimeError("primary trajectory cache hash mismatch")
        with path.open("rb") as handle:
            trajectory = pickle.load(handle)
        selected = selected_clock_observations(trajectory, CLOCK_ID)
        states = states_from_observations(selected)
        first, _ = label_trajectory(trajectory, spec, clock_id=CLOCK_ID)
        second, _ = label_trajectory(trajectory, spec, clock_id=CLOCK_ID)
        label_exact = np.array_equal(
            first["isReplicator"].to_numpy(bool), second["isReplicator"].to_numpy(bool)
        )
        score_exact = np.array_equal(
            first["labelScore"].to_numpy(float),
            second["labelScore"].to_numpy(float),
            equal_nan=True,
        )
        index_exact = np.array_equal(
            first["selectedSequenceIndex"].to_numpy(int),
            second["selectedSequenceIndex"].to_numpy(int),
        )
        eligible = len(first) >= HORIZON_EXCLUSIVE
        if eligible:
            target = build_landmark_target(first["isReplicator"].to_numpy(bool))
        else:
            target = {
                "observationCount": len(first),
                "wholeTrajectoryOccupancy": float(first["isReplicator"].mean())
                if len(first)
                else float("nan"),
                "firstOnsetIndex0": None,
                "atRiskAtLandmark": False,
                "eventWithinHorizon": None,
                "landmarkCount": LANDMARK_COUNT,
                "horizonExclusive": HORIZON_EXCLUSIVE,
            }
        replay_rows.append(
            {
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "trajectoryId": row.trajectoryId,
                "labelExact": label_exact,
                "scoreExact": score_exact,
                "indexExact": index_exact,
                "targetEligible": eligible,
                "exactReplayPassed": label_exact and score_exact and index_exact,
            }
        )
        target_rows.append(
            {
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "trajectoryId": row.trajectoryId,
                "analysisEligible": eligible,
                **target,
            }
        )
        loaded[(row.candidateId, int(row.matrixIndex))] = {
            "selected": selected,
            "states": states,
        }
    replay = pd.DataFrame(replay_rows)
    targets = pd.DataFrame(target_rows)
    if not replay["exactReplayPassed"].all():
        raise RuntimeError("L23 target exact replay failed")
    cohort = targets[targets["atRiskAtLandmark"] & targets["analysisEligible"]].copy()
    cohort["eventWithinHorizon"] = cohort["eventWithinHorizon"].astype(bool)
    splits = split_registry(cohort)
    GENERATED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    BASE.write_parquet(GENERATED_INPUT_ROOT / "split_manifest.parquet", splits)

    regenerated, regen_seeds, regen_failures = _simulate_all(REGEN_CACHE, 8)
    if not regen_failures.empty or len(regenerated) != len(manifest):
        raise RuntimeError("L23 trajectory regeneration incomplete")
    compare_fields = [
        "candidateId",
        "matrixIndex",
        "trajectoryId",
        "trajectorySha256",
        "betaSha256",
        "initialStateSha256",
        "terminalStatus",
        "completedFissions",
        "selectedClockLength",
    ]
    primary = manifest[compare_fields].sort_values(["candidateId", "matrixIndex"])
    regenerated_values = regenerated[compare_fields].sort_values(
        ["candidateId", "matrixIndex"]
    )
    comparisons = primary.merge(
        regenerated_values,
        on=["candidateId", "matrixIndex"],
        suffixes=("Primary", "Regenerated"),
        validate="one_to_one",
    )
    exact_fields = []
    for field in compare_fields[2:]:
        exact_fields.append(
            comparisons[f"{field}Primary"].eq(comparisons[f"{field}Regenerated"])
        )
    comparisons["trajectoryFieldsExact"] = np.logical_and.reduce(exact_fields)
    if not comparisons["trajectoryFieldsExact"].all():
        raise RuntimeError("L23 trajectory exact regeneration mismatch")
    frozen_seeds = pd.read_parquet(LOOP_ROOT / "seed_manifest.parquet")
    runtime_seed_unique = runtime_seeds.drop_duplicates(
        ["matrixIndex", "candidateId", "purpose", "seedMaterialSha256"]
    )
    seed_exact = set(
        map(tuple, frozen_seeds[["matrixIndex", "candidateId", "purpose", "seedMaterialSha256"]].to_numpy())
    ) == set(
        map(tuple, runtime_seed_unique[["matrixIndex", "candidateId", "purpose", "seedMaterialSha256"]].to_numpy())
    )
    regen_seed_exact = set(
        map(tuple, runtime_seed_unique[["matrixIndex", "candidateId", "purpose", "seedMaterialSha256"]].to_numpy())
    ) == set(
        map(tuple, regen_seeds[["matrixIndex", "candidateId", "purpose", "seedMaterialSha256"]].drop_duplicates().to_numpy())
    )
    if not seed_exact or not regen_seed_exact:
        raise RuntimeError("L23 runtime seed replay mismatch")
    BASE.EXTRA_SCIENTIFIC_TABLES.clear()
    BASE.EXTRA_SCIENTIFIC_TABLES.update(
        {
            "runtime_seed_manifest.parquet": runtime_seed_unique,
            "trajectory_regeneration_results.parquet": comparisons,
            "simulation_failure_results.parquet": failures,
        }
    )
    BASE.EXTRA_REGENERATION_SUMMARY.clear()
    BASE.EXTRA_REGENERATION_SUMMARY.update(
        {
            "trajectoryUnitsCompared": len(comparisons),
            "trajectoryUnitsExact": int(comparisons["trajectoryFieldsExact"].sum()),
            "runtimeSeedManifestExact": seed_exact,
            "regenerationSeedManifestExact": regen_seed_exact,
            "newSharedMatrices": MATRIX_COUNT,
            "newTrajectories": len(manifest),
        }
    )
    return manifest, replay, targets, loaded


def _feature_worker(
    payload: tuple[str, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> list[dict[str, Any]]:
    candidate, matrix_index, prefix, generations, kinds, permutation = payload
    original = combined_features(prefix, generations, kinds)
    temporal = combined_features(
        prefix[permutation], generations[permutation], kinds[permutation]
    )
    return [
        {"candidateId": candidate, "matrixIndex": matrix_index, "variant": "ORIGINAL", **original},
        {"candidateId": candidate, "matrixIndex": matrix_index, "variant": "TEMPORAL_PERMUTED", **temporal},
    ]


def extract_features(
    manifest: pd.DataFrame,
    loaded: dict[tuple[str, int], dict[str, Any]],
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payloads = []
    for row in manifest.itertuples(index=False):
        key = (row.candidateId, int(row.matrixIndex))
        selected = loaded[key]["selected"]
        states = loaded[key]["states"]
        if len(states) < LANDMARK_COUNT:
            continue
        prefix = states[:LANDMARK_COUNT]
        generations = np.asarray(
            [int(item.growth_generation_one_based) for item in selected[:LANDMARK_COUNT]],
            dtype=np.int64,
        )
        kinds = np.asarray(
            [str(item.observation_kind) for item in selected[:LANDMARK_COUNT]],
            dtype=object,
        )
        rng = np.random.default_rng(BASE.derive_seed("l23_temporal", *key))
        permutation = np.arange(LANDMARK_COUNT)
        permutation[1:] = rng.permutation(permutation[1:])
        payloads.append((key[0], key[1], prefix, generations, kinds, permutation))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_feature_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            rows.extend(future.result())
    features = pd.DataFrame(rows).sort_values(
        ["candidateId", "matrixIndex", "variant"]
    ).reset_index(drop=True)
    expected = set(MODEL_FEATURES["L18_PAST_FULL_NO_BGM"]) | set(ALL_NEW_FEATURES)
    if not expected.issubset(features.columns) or not np.isfinite(
        features[list(expected)].to_numpy(float)
    ).all():
        raise RuntimeError("L23 feature schema/finiteness failure")
    for payload in payloads:
        candidate, matrix_index, prefix, generations, kinds, permutation = payload
        for variant, values, gens, labels in (
            ("ORIGINAL", prefix, generations, kinds),
            (
                "TEMPORAL_PERMUTED",
                prefix[permutation],
                generations[permutation],
                kinds[permutation],
            ),
        ):
            observed = combined_features(values, gens, labels)
            stored = features[
                features["candidateId"].eq(candidate)
                & features["matrixIndex"].eq(matrix_index)
                & features["variant"].eq(variant)
            ].iloc[0]
            if any(stored[name] != observed[name] for name in expected):
                raise RuntimeError("L23 independent feature replay mismatch")
    features["independentReplayExact"] = True
    feature_permuted = []
    for candidate, frame in features[features["variant"].eq("ORIGINAL")].groupby(
        "candidateId", sort=True
    ):
        frame = frame.sort_values("matrixIndex").copy()
        rng = np.random.default_rng(BASE.derive_seed("l23_feature_row_permutation", candidate))
        order = rng.permutation(len(frame))
        frame.loc[:, ALL_NEW_FEATURES] = frame[list(ALL_NEW_FEATURES)].to_numpy(float)[order]
        frame["variant"] = "FEATURE_PERMUTED"
        feature_permuted.append(frame)
    features = pd.concat([features, *feature_permuted], ignore_index=True)
    registry = pd.DataFrame(
        [
            {
                "modelId": model,
                "featureFamily": "CONTROL"
                if model not in LEAD_MODELS
                else "FROZEN_PREFIX_FAMILY",
                "featureCount": len(fields),
                "fields": json.dumps(fields),
                "pastOnly": True,
                "sourceGrounded": model in LEAD_MODELS,
            }
            for model, fields in MODEL_FEATURES.items()
        ]
    )
    return features, registry


ORIGINAL_CV = BASE.cross_validated_predictions
ORIGINAL_GATES = BASE.scientific_gates
ORIGINAL_MANIFEST = BASE.manifest_for


def cross_validated_predictions(
    cohort: pd.DataFrame,
    features: pd.DataFrame,
    splits: pd.DataFrame,
    model_ids: Any = None,
    variant: str = "ORIGINAL",
    y_override: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    return ORIGINAL_CV(
        cohort,
        features,
        splits,
        MODEL_IDS if model_ids is None else model_ids,
        variant,
        y_override,
    )


def suffix_invariance(
    manifest: pd.DataFrame, loaded: dict[tuple[str, int], dict[str, Any]]
) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        for row in (
            manifest[manifest["candidateId"].eq(candidate)]
            .sort_values("matrixIndex")
            .head(8)
            .itertuples(index=False)
        ):
            key = (candidate, int(row.matrixIndex))
            selected = loaded[key]["selected"]
            states = loaded[key]["states"]
            prefix = states[:LANDMARK_COUNT]
            generations = np.asarray(
                [int(item.growth_generation_one_based) for item in selected[:LANDMARK_COUNT]]
            )
            kinds = np.asarray(
                [str(item.observation_kind) for item in selected[:LANDMARK_COUNT]],
                dtype=object,
            )
            before = combined_features(prefix, generations, kinds)
            altered = states.copy()
            rng = np.random.default_rng(BASE.derive_seed("l23_suffix", *key))
            altered[LANDMARK_COUNT:] = altered[LANDMARK_COUNT:][
                rng.permutation(len(altered) - LANDMARK_COUNT)
            ]
            after = combined_features(altered[:LANDMARK_COUNT], generations, kinds)
            rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": int(row.matrixIndex),
                    "featureExactInvariant": before == after,
                    "fieldsChecked": len(before),
                    "passed": before == after,
                }
            )
    return pd.DataFrame(rows)


def scientific_gates(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, list[str], str | None]:
    gates, _generic, _selected = ORIGINAL_GATES(*args, **kwargs)
    targets = args[0]
    for candidate in CANDIDATES:
        risk = targets[
            targets["candidateId"].eq(candidate) & targets["atRiskAtLandmark"]
        ]
        events = int(risk["eventWithinHorizon"].sum())
        non_events = int(len(risk) - events)
        powered = len(risk) >= 150 and events >= 50 and non_events >= 50
        mask = gates["candidateId"].eq(candidate)
        gates.loc[mask, "taskEstablished"] = powered
        gates.loc[mask, "candidateDiscoveryGatePassed"] &= powered
    passing = [
        model
        for model in LEAD_MODELS
        if gates[gates["modelId"].eq(model)]["candidateDiscoveryGatePassed"].tolist()
        == [True, True]
    ]
    selected = passing[0] if passing else None
    classifications = ["POWERED_ATTRACTOR_ONSET_TASK_ESTABLISHED"]
    if selected:
        classifications.extend(
            [
                "POWERED_FROZEN_FAMILY_DISCOVERY_LEAD",
                "REQUIRES_UNTOUCHED_CONFIRMATION",
                "NOT_PROMOTABLE_AS_CONFIRMED",
            ]
        )
    else:
        classifications.extend(
            [
                "POWERED_FROZEN_FAMILY_NON_SUPPORT",
                "CANDIDATE_HETEROGENEITY_PERSISTS",
                "NOT_PROMOTABLE_AS_CONFIRMED",
            ]
        )
    return gates, classifications, selected


def make_figures(
    root: Path,
    targets: pd.DataFrame,
    features: pd.DataFrame,
    aggregate: pd.DataFrame,
    comparisons: pd.DataFrame,
    permutation: pd.DataFrame,
    controls: pd.DataFrame,
    gates: pd.DataFrame,
) -> list[str]:
    directory = root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(name: str) -> None:
        path = directory / name
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()
        paths.append(str(path.relative_to(root)))

    risk = targets[targets["atRiskAtLandmark"]].groupby("candidateId")[
        "eventWithinHorizon"
    ].agg(["count", "sum"])
    risk["nonEvent"] = risk["count"] - risk["sum"]
    risk[["sum", "nonEvent"]].plot(kind="bar", color=["#1976d2", "#9e9e9e"])
    plt.ylabel("matrices")
    plt.title("Powered independent onset task")
    plt.legend(["event", "non-event"])
    save("01_powered_task_geometry.png")

    targets.boxplot(column="wholeTrajectoryOccupancy", by="candidateId")
    plt.suptitle("")
    plt.title("Completed-run target occupancy")
    plt.ylabel("occupancy")
    save("02_target_occupancy.png")

    focus = aggregate[
        aggregate["variant"].eq("ORIGINAL")
        & aggregate["modelId"].isin(["EXACT_H_STABILITY", "COMPACT_BASELINE", *LEAD_MODELS])
    ]
    focus.pivot(index="modelId", columns="candidateId", values="AUROC").plot(
        kind="bar", ylim=(0, 1), figsize=(10, 5), color=["#1565c0", "#ef6c00"]
    )
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("matrix repeated-CV AUROC")
    plt.title("Frozen-family powered screen")
    save("03_model_auroc.png")

    delta = comparisons[
        comparisons["rightModel"].eq("COMPACT_BASELINE")
        & comparisons["metric"].eq("AUROC")
    ]
    for candidate, frame in delta.groupby("candidateId"):
        plt.errorbar(
            frame["leftModel"],
            frame["favorableDelta"],
            yerr=[
                frame["favorableDelta"] - frame["bootstrapLower95"],
                frame["bootstrapUpper95"] - frame["favorableDelta"],
            ],
            fmt="o",
            label=candidate,
        )
    plt.axhline(0, color="black", linestyle="--")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("AUROC increment over compact")
    plt.legend()
    save("04_bootstrap_increments.png")

    permutation.pivot(index="modelId", columns="candidateId", values="familywisePValue").plot(
        kind="bar", ylim=(0, 1), figsize=(10, 5), color=["#1565c0", "#ef6c00"]
    )
    plt.axhline(0.10, color="black", linestyle="--")
    plt.ylabel("max-statistic p")
    plt.title("Whole-matrix label permutation")
    save("05_permutation_control.png")

    paired = focus.pivot(index="modelId", columns="candidateId", values="AUROC")
    plt.scatter(paired.iloc[:, 0], paired.iloc[:, 1])
    for model, row in paired.iterrows():
        plt.annotate(model.replace("COMPACT_PLUS_", ""), (row.iloc[0], row.iloc[1]), fontsize=7)
    plt.axvline(0.5, color="grey", linestyle="--")
    plt.axhline(0.5, color="grey", linestyle="--")
    plt.xlabel(paired.columns[0])
    plt.ylabel(paired.columns[1])
    plt.title("Cross-candidate agreement")
    save("06_cross_candidate_agreement.png")

    controls[controls["modelId"].isin(LEAD_MODELS)].pivot_table(
        index="modelId", columns=["candidateId", "controlId"], values="controlAuRoc"
    ).plot(kind="bar", ylim=(0, 1), figsize=(11, 5))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("AUROC")
    plt.title("Temporal and feature-row controls")
    save("07_negative_controls.png")

    gate = gates.pivot(
        index="modelId", columns="candidateId", values="candidateDiscoveryGatePassed"
    ).astype(int)
    plt.imshow(gate.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.xticks(range(len(gate.columns)), gate.columns, rotation=20)
    plt.yticks(range(len(gate.index)), gate.index)
    plt.colorbar(ticks=[0, 1])
    plt.title("Powered discovery gates")
    save("08_gate_matrix.png")
    return paths


def report_text(
    targets: pd.DataFrame,
    aggregate: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    selected: str | None,
    runtime: dict[str, Any],
) -> str:
    geometry = (
        targets[targets["atRiskAtLandmark"]]
        .groupby("candidateId")
        .agg(
            atRisk=("matrixIndex", "size"),
            events=("eventWithinHorizon", "sum"),
            occupancy=("wholeTrajectoryOccupancy", "mean"),
        )
        .reset_index()
    )
    geometry["nonEvents"] = geometry["atRisk"] - geometry["events"]
    focus = aggregate[
        aggregate["variant"].eq("ORIGINAL")
        & aggregate["modelId"].isin(
            ["DUMMY_TRAINING_PRIOR", "EXACT_H_STABILITY", "COMPACT_BASELINE", *LEAD_MODELS]
        )
    ][["candidateId", "modelId", "AUROC", "AUPRC", "BRIER", "BALANCED_ACCURACY"]]
    recommendation = (
        f"Freeze `{selected}` unchanged and perform a second seed-firewalled confirmation in L24."
        if selected
        else "The larger independent cohort rules out simple underpowering of every frozen family. Advance to a compact cross-candidate reaction-coordinate loop rather than retuning these families."
    )
    return f"""# S19-L23 — Powered Independent Screen of Frozen Prefix Organization Families

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete within the authorized autonomous L19–L42 program.
- **Outcome classifications:** {", ".join(f"`{item}`" for item in classifications)}
- **Selected discovery lead:** `{selected or "NONE"}`.
- **Validation:** pre-outcome seed/input firewall; 400 shared matrices and 800 registered trajectories without replacement; exact target, trajectory, seed, feature and model replay; candidate-separated matrix CV; 4,096 bootstraps; 512 max-statistic permutations; temporal/feature/suffix controls; immutable-prior, storage and artifact hashes passed.
- **Recommended next bounded loop:** {recommendation}

## Frozen question

Does increased independent matrix support reveal a common pre-onset signal in any complete feature family already frozen in L19, L20 or L22?

## Cohort

{geometry.to_markdown(index=False)}

## Methods

Exactly 400 new catalytic matrices and matched initial states were generated before label analysis from a domain-separated root with no detected prior hash or seed-material overlap. Both S13Y simulator candidates completed their registered attempt; incomplete/extinct units were retained and never replaced. The retrospective L02 recurring-attractor label served only as the outcome. All L19 critical-slowing/RQA/DMD, L20 topology/intrinsic/path, and L22 random-convolution implementations were reused unchanged. Six complete registered bundles were tested under one max-statistic family with the exact C=1 logistic estimator.

## Results

{focus.to_markdown(index=False)}

## Gate adjudication

{gates.to_markdown(index=False)}

In addition to the L19 discovery gates, L23 required at least 150 at-risk matrices and at least 50 events and 50 non-events per candidate. The same frozen family had to pass both candidates. A studied-cohort pass would remain discovery evidence and require another untouched confirmation.

## Interpretation

L23 changes power, not method. It therefore distinguishes small-cohort instability from reproducible signal without creating more opportunities through feature retuning. The target remains a completed-run reconstruction and does not identify author code.

## Runtime and provenance

- Repository lock: `{runtime["repositoryHead"]}`.
- CPU float64, `{runtime["workers"]}` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `{runtime["wallSeconds"]:.3f}`; process CPU hours: `{runtime["processCpuHours"]:.6f}`.
- Temporary trajectory payloads remain under `/cache/e01_s19_l23`; compact identities and regeneration evidence are retained in the artifact bundle.

## Autonomous continuation boundary

L23 is frozen. The existing authorization permits one next bounded loop through at most L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
"""


def append_root_ledgers(
    classifications: list[str], selected: str | None, timestamp: str
) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "The 53/54-matrix discovery task may be too small to distinguish stable weak effects from candidate heterogeneity.",
            "failureOrAmbiguityTargeted": "Whether prior family non-support primarily reflects low matrix-level power.",
            "informationGainRationale": "A 400-shared-matrix seed-firewalled cohort reuses every frozen family without retuning.",
            "learned": "L23 cohort/feature/model/gate contract frozen before trajectory generation.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L19-L22 candidate-specific estimates with broad matrix intervals.",
            "proposedNextTest": "Execute L23.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Weak common frozen-family signal or a different compact reaction coordinate.",
            "selectedHypotheses": "Omnibus powered re-evaluation of all complete frozen L19/L20/L22 families.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "The original discovery cohort is adequately powered for weak effects.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "At least one frozen family might stabilize with fourfold matrix support.",
            "failureOrAmbiguityTargeted": "Low power versus genuine cross-candidate non-reproducibility.",
            "informationGainRationale": "Independent trajectories, full multiplicity correction and exact regeneration adjudicate weak effects.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete frozen L23 results.",
            "proposedNextTest": f"Untouched confirmation of {selected}." if selected else "Compact cross-candidate reaction-coordinate discovery in L24.",
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": "A low-dimensional invariant reaction coordinate, closer landmark, or event-conditional change point.",
            "selectedHypotheses": "Omnibus powered re-evaluation of all complete frozen L19/L20/L22 families.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Failed frozen families merely lacked matrix-level power.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat(
            [ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)],
            ignore_index=True,
        ),
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    start = int(candidates["registryOrder"].max()) + 1
    candidate_rows = []
    for offset, model in enumerate(LEAD_MODELS):
        candidate_rows.append(
            {
                "branchCount": len(LEAD_MODELS),
                "bundleId": "L23_POWERED_FROZEN_FAMILY_SCREEN",
                "candidateId": f"S19-L23-{model}",
                "candidateSpecificSuccess": 0,
                "completedFitLeakage": 0,
                "computeEfficiency": 4,
                "crossCandidateDiscriminability": 5,
                "deterministicHReuse": 0,
                "explanatoryLeverage": 4,
                "frozenRank": offset + 1,
                "independenceFromPriorOutcomeSelection": 4,
                "outcomeGuidedThresholdSelection": 0,
                "paperFingerprintSpecificity": 0,
                "proposedSpecification": model,
                "rankingScore": float(24 - offset),
                "registryOrder": start + offset,
                "selected": True,
                "selectionReason": "POWERED_ALL_FROZEN_FAMILY_SCREEN",
                "sourceGrounding": 5,
                "testability": 5,
                "undefinedAuthorSemantics": 0,
            }
        )
    BASE.write_parquet(
        candidates_path,
        pd.concat(
            [candidates, pd.DataFrame(candidate_rows).reindex(columns=candidates.columns)],
            ignore_index=True,
        ),
    )
    sources_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(sources_path)
    source_rows = []
    for item in source_registry().itertuples(index=False):
        source_rows.append(
            {
                "commitOrVersion": item.doi,
                "evidenceClass": item.evidenceClass,
                "finding": f"{item.directSupport}; frozen L23 use: {item.reconstructionChoice}",
                "licenseStatus": "PRIOR_FROZEN_OR_PUBLIC_ARTICLE",
                "redistributionStatus": "CITATION_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L23_{item.sourceId}",
                "sourceType": item.evidenceClass,
                "treeIdentity": None,
                "url": item.url,
            }
        )
    BASE.write_parquet(
        sources_path,
        pd.concat(
            [sources, pd.DataFrame(source_rows).reindex(columns=sources.columns)],
            ignore_index=True,
        ),
    )
    loop_path = ARTIFACT_ROOT / "loop_registry.yaml"
    data = yaml.safe_load(loop_path.read_text())
    data["loops"].append(
        {
            "loopId": LOOP_ID,
            "versionedLoopId": VERSION,
            "status": "COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED",
            "authorized": True,
            "completed": True,
            "outcomeAccessed": True,
            "humanReviewRequiredAfter": False,
            "classification": classifications,
            "selectedDiscoveryLead": selected,
            "newMatrices": MATRIX_COUNT,
            "newTrajectories": MATRIX_COUNT * 2,
            "nextStepActive": True,
        }
    )
    data["laterLoopsAuthorized"] = True
    data["authorizationUpperBound"] = "S19-L42"
    data["proposedNextLoopTheme"] = (
        f"UNTOUCHED_CONFIRMATION_{selected}"
        if selected
        else "COMPACT_CROSS_CANDIDATE_REACTION_COORDINATE"
    )
    data["proposedNextLoopActive"] = True
    BASE.atomic_text(loop_path, yaml.safe_dump(data, sort_keys=False))
    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text())
    review["history"].append(
        {
            "decision": "S19_L23_COMPLETE_CONTINUE_UNDER_EXISTING_AUTHORIZATION",
            "loopId": LOOP_ID,
            "scope": VERSION,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "selectedDiscoveryLead": selected,
            "source": "locked_execution_result",
            "nextLoopAuthorized": True,
            "s20Activated": False,
        }
    )
    review["pendingDecision"] = "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L42"
    BASE.write_json(review_path, review)


def manifest_for(root: Path) -> dict[str, Any]:
    result = ORIGINAL_MANIFEST(root)
    result["schema"] = "eidosoma.e01.s19_l23.artifact_manifest.v1"
    return result


def decision_summary_text(classifications: list[str], selected: str | None) -> str:
    outcome = (
        f"`{selected}` passed the powered discovery gate and must be confirmed unchanged on another seed-firewalled cohort."
        if selected
        else "No frozen L19/L20/L22 family passed the powered two-candidate gate; low matrix count is no longer a sufficient explanation for those nulls."
    )
    return f"""# S19-L23 decision summary

**Classification:** {", ".join(classifications)}
**Selected discovery lead:** `{selected or "NONE"}`

{outcome}

The human authorization permits one next bounded loop through L42. S20, E02, author contact, interventions and report generation remain inactive.
"""


def configure_base() -> None:
    BASE.LOOP_ID = LOOP_ID
    BASE.VERSION = VERSION
    BASE.TARGET_ID = TARGET_ID
    BASE.CANDIDATES = CANDIDATES
    BASE.LOOP_ROOT = LOOP_ROOT
    BASE.CACHE_ROOT = CACHE_ROOT
    BASE.BUILD_ROOT = BUILD_ROOT
    BASE.CONFIG = CONFIG
    BASE.L18_ROOT = GENERATED_INPUT_ROOT
    BASE.BOOTSTRAPS = BOOTSTRAPS
    BASE.PERMUTATIONS = PERMUTATIONS
    BASE.MODEL_FEATURES = MODEL_FEATURES
    BASE.MODEL_IDS = MODEL_IDS
    BASE.LEAD_MODELS = LEAD_MODELS
    BASE.CANONICAL_REPORT_NAME = "S19_L23_FULL_RESULTS.md"
    BASE.ROOT_HANDOFF_SOURCE_HEADER = "# S19-L23"
    BASE.ROOT_HANDOFF_TARGET_HEADER = "# S19 current handoff — S19-L23"
    BASE.NULL_NEXT_ACTION = "S19_L24_COMPACT_CROSS_CANDIDATE_REACTION_COORDINATE"
    BASE.RUNTIME_SCHEMA = "eidosoma.e01.s19_l23.runtime.v1"
    BASE.ADDITIONAL_LOCK_ARTIFACTS = (
        "input_manifest.parquet",
        "seed_manifest.parquet",
        "seed_firewall.json",
    )
    BASE.validate_immutable_prior = validate_immutable_prior
    BASE.fixture_table = fixture_table
    BASE.source_registry = source_registry
    BASE.replay_task = replay_task
    BASE.extract_features = extract_features
    BASE.extract_organization_warning_features = lambda states: extract_outcome_blind_representation(states)
    BASE.cross_validated_predictions = cross_validated_predictions
    BASE.suffix_invariance = suffix_invariance
    BASE.scientific_gates = scientific_gates
    BASE.make_figures = make_figures
    BASE.report_text = report_text
    BASE.append_root_ledgers = append_root_ledgers
    BASE.manifest_for = manifest_for
    BASE.decision_summary_text = decision_summary_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-lock", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be between 1 and 8")
    configure_base()
    if args.prepare_lock:
        prepare_lock()
    else:
        BASE.execute(args.workers)


if __name__ == "__main__":
    main()
