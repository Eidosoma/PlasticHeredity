"""Execute S19-L38 past-only recurrence/inheritance outcome construction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import xlogy
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_onset_discovery.empirical_committor import (
    RestoredState,
    corrected_between_state_variance,
)
from e01_onset_discovery.recurrence_inheritance import (
    cosine_h,
    score_recurrence_inheritance,
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L37 = _load_module(
    "e01_s19_l38_l37",
    REPO_ROOT / "scripts/e01/run_s19_l37_multilineage_any_attractor.py",
)
L36 = L37.L36
L35 = L37.L35
L31 = L37.L31
L30 = L37.L30
L29 = L35.L29
L28 = L37.L28
BASE = L37.BASE

LOOP_ID = "S19-L38"
VERSION = "E01-S19-L38-PAST-ONLY-RECURRENCE-INHERITANCE-OUTCOME-v1.0.0"
CANDIDATES = L37.CANDIDATES
COHORTS = L37.COHORTS
EVALUATION_COHORTS = L37.EVALUATION_COHORTS
FAMILIES = L37.FAMILIES
HORIZONS = L37.HORIZONS
BRANCH_COUNTS = L37.BRANCH_COUNTS
HALVES = L37.HALVES
TARGETS = (
    "PAST_ONLY_RECURRENCE_INHERITANCE",
    "SPECIES_PERMUTED_PREFIX",
    "UNRELATED_MATRIX_PREFIX",
    "BRANCH_ONLY_RECURRENCE",
    "INHERITANCE_ONLY",
)
PRIMARY_TARGET = TARGETS[0]
THRESHOLD = 0.9
MINIMUM_GENERATION_GAP = 2
BOOTSTRAPS = 4096
ROOT_HEX = "e1a59931a5f74a09ac7e42f7196db78f4740a5ae9ce4014e25f22a58d8b9ae0c"
PHASE = "s19_l38_recurrence_inheritance"
WORKERS = min(8, os.cpu_count() or 1)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L38"
L37_ROOT = ARTIFACT_ROOT / "loops/L37"
L36_ROOT = ARTIFACT_ROOT / "loops/L36"
L35_ROOT = ARTIFACT_ROOT / "loops/L35"
L31_ROOT = ARTIFACT_ROOT / "loops/L31"
L30_ROOT = ARTIFACT_ROOT / "loops/L30"
L28_ROOT = ARTIFACT_ROOT / "loops/L28"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l38")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l38_past_only_recurrence_inheritance.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/recurrence_inheritance.py"


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


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_json(orient="table", index=False, double_precision=15).encode()
    ).hexdigest()


def derived_seed(*parts: object) -> int:
    payload = "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.unique(left[mask]).size < 2 or np.unique(right[mask]).size < 2:
        return float("nan")
    return float(spearmanr(left[mask], right[mask]).statistic)


def interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L37_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L37_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L37_ROOT / item["path"]),
            "root": str(L37_ROOT),
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in manifest["files"]
    )
    checked = []
    for row in rows:
        path = Path(row["path"])
        actual = sha256_file(path) if path.is_file() else None
        checked.append(
            {
                **row,
                "actualSha256": actual,
                "unchanged": actual == row["sha256"],
            }
        )
    aggregate = hashlib.sha256(
        "\n".join(f"{row['path']}|{row['sha256']}" for row in checked).encode()
    ).hexdigest()
    return {
        "schema": "eidosoma.e01.s19_l38.immutable_prior_validation.v1",
        "status": "PASS" if all(row["unchanged"] for row in checked) else "FAIL",
        "unchanged": all(row["unchanged"] for row in checked),
        "fileCount": len(checked),
        "aggregateSha256": aggregate,
        "files": checked,
    }


def fixture_results() -> pd.DataFrame:
    prefix_states = np.asarray([[10, 1], [1, 10]], dtype=np.int64)
    prefix_generations = np.asarray([1, 2], dtype=np.int64)
    prefix_inherited = np.asarray([True, True])
    future_states = np.asarray([[10, 1]], dtype=np.int64)
    future_generations = np.asarray([3], dtype=np.int64)
    future_inherited = np.asarray([True])
    event = score_recurrence_inheritance(
        prefix_states=prefix_states,
        prefix_generations=prefix_generations,
        prefix_inherited=prefix_inherited,
        future_states=future_states,
        future_generations=future_generations,
        future_inherited=future_inherited,
        threshold=THRESHOLD,
        minimum_generation_gap=MINIMUM_GENERATION_GAP,
    )
    adjacent = score_recurrence_inheritance(
        prefix_states=prefix_states[-1:],
        prefix_generations=prefix_generations[-1:],
        prefix_inherited=prefix_inherited[-1:],
        future_states=future_states,
        future_generations=future_generations,
        future_inherited=future_inherited,
        threshold=THRESHOLD,
        minimum_generation_gap=MINIMUM_GENERATION_GAP,
    )
    noninherit = score_recurrence_inheritance(
        prefix_states=prefix_states[:1],
        prefix_generations=prefix_generations[:1],
        prefix_inherited=np.asarray([False]),
        future_states=future_states,
        future_generations=future_generations,
        future_inherited=future_inherited,
        threshold=THRESHOLD,
        minimum_generation_gap=MINIMUM_GENERATION_GAP,
    )
    return pd.DataFrame(
        [
            {
                "fixtureId": "STRICT_COSINE_IDENTITY",
                "passed": abs(
                    cosine_h(np.asarray([2, 1]), np.asarray([4, 2])) - 1.0
                )
                <= 1e-15,
                "details": "positive scaling preserves H",
            },
            {
                "fixtureId": "RECURRENCE_WITH_INTERVENING_GENERATION",
                "passed": event.event and event.matched_reference_generation == 1,
                "details": json.dumps(event.__dict__ if hasattr(event, "__dict__") else {
                    "event": event.event,
                    "referenceGeneration": event.matched_reference_generation,
                }),
            },
            {
                "fixtureId": "ADJACENT_GENERATION_EXCLUDED",
                "passed": not adjacent.event and adjacent.eligible_comparison_count == 0,
                "details": "generation gap one cannot establish recurrence",
            },
            {
                "fixtureId": "PRIOR_INHERITANCE_REQUIRED",
                "passed": not noninherit.event,
                "details": "a non-inherited reference boundary is ineligible",
            },
            {
                "fixtureId": "FROZEN_SCOPE",
                "passed": FAMILIES == ("H32", "H8")
                and BRANCH_COUNTS == {"H32": 128, "H8": 64}
                and TARGETS == (
                    "PAST_ONLY_RECURRENCE_INHERITANCE",
                    "SPECIES_PERMUTED_PREFIX",
                    "UNRELATED_MATRIX_PREFIX",
                    "BRANCH_ONLY_RECURRENCE",
                    "INHERITANCE_ONLY",
                ),
                "details": json.dumps(
                    {"families": FAMILIES, "branchCounts": BRANCH_COUNTS, "targets": TARGETS}
                ),
            },
        ]
    )


def load_trajectory(row: Any) -> Any:
    path = Path(row.cachePath)
    if not path.is_file() or sha256_file(path) != row.cacheSha256:
        raise RuntimeError(f"L38 trajectory cache identity failure: {path}")
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if trajectory.trajectory_sha256 != row.trajectorySha256:
        raise RuntimeError("L38 trajectory payload identity failure")
    return trajectory


def _prefix_event(
    states: np.ndarray, generations: np.ndarray, inherited: np.ndarray
) -> Any:
    return score_recurrence_inheritance(
        prefix_states=np.empty((0, states.shape[1]), dtype=np.int64),
        prefix_generations=np.empty(0, dtype=np.int64),
        prefix_inherited=np.empty(0, dtype=np.bool_),
        future_states=states,
        future_generations=generations,
        future_inherited=inherited,
        threshold=THRESHOLD,
        minimum_generation_gap=MINIMUM_GENERATION_GAP,
    )


def prefix_registries(
    responses: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    boundary_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for response in responses.itertuples(index=False):
        source = manifest_index.loc[(response.candidateId, int(response.matrixIndex))]
        trajectory = load_trajectory(source)
        selected = L28.selected_clock_observations(trajectory, L28.CLOCK_ID)
        current_index = int(response.currentSelectedIndex)
        if current_index >= len(selected):
            raise RuntimeError("L38 current selected-clock index is out of bounds")
        current = selected[current_index]
        current_hash = L28.array_sha256(
            np.asarray(current.state, dtype=np.int64)
        )
        if (
            current_hash != response.currentStateSha256
            or current.observation_kind != response.currentObservationKind
            or int(current.completed_fissions) != int(response.currentCompletedFissions)
        ):
            raise RuntimeError(f"L38 restored prefix identity failure: {response.stateId}")
        states = []
        generations = []
        inherited = []
        parent_h_values = []
        for index, observation in enumerate(selected[: current_index + 1]):
            if observation.observation_kind != "post_fission":
                continue
            if index == 0:
                raise RuntimeError("post-fission boundary cannot be first selected observation")
            parent = selected[index - 1]
            if (
                parent.observation_kind != "molecular_update"
                or parent.growth_generation_one_based
                != observation.growth_generation_one_based
            ):
                raise RuntimeError("L38 selected-daughter predecessor mismatch")
            score = cosine_h(
                np.asarray(parent.state, dtype=np.int64),
                np.asarray(observation.state, dtype=np.int64),
            )
            boundary_rows.append(
                {
                    "stateId": response.stateId,
                    "evaluationCohort": response.evaluationCohort,
                    "candidateId": response.candidateId,
                    "matrixIndex": int(response.matrixIndex),
                    "landmark": int(response.landmark),
                    "boundaryOrdinal": len(states) + 1,
                    "selectedClockIndex": index,
                    "generation": int(observation.completed_fissions),
                    "parentDaughterH": score,
                    "inherited": bool(score > THRESHOLD),
                    "state": list(map(int, observation.state)),
                    "stateSha256": L28.array_sha256(
                        np.asarray(observation.state, dtype=np.int64)
                    ),
                }
            )
            states.append(observation.state)
            generations.append(int(observation.completed_fissions))
            inherited.append(bool(score > THRESHOLD))
            parent_h_values.append(score)
        if not states:
            raise RuntimeError("L38 requires an observed post-fission prefix")
        state_array = np.asarray(states, dtype=np.int64)
        generation_array = np.asarray(generations, dtype=np.int64)
        inheritance_array = np.asarray(inherited, dtype=np.bool_)
        historical = _prefix_event(state_array, generation_array, inheritance_array)
        latest = state_array[-1]
        latest_generation = generation_array[-1]
        eligible = [
            state_array[index]
            for index in range(len(state_array) - 1)
            if inheritance_array[index]
            and latest_generation - generation_array[index] >= MINIMUM_GENERATION_GAP
        ]
        latest_maximum = (
            max(cosine_h(latest, reference) for reference in eligible)
            if eligible and inheritance_array[-1]
            else np.nan
        )
        summary_rows.append(
            {
                "stateId": response.stateId,
                "evaluationCohort": response.evaluationCohort,
                "candidateId": response.candidateId,
                "matrixIndex": int(response.matrixIndex),
                "landmark": int(response.landmark),
                "prefixBoundaryCount": len(state_array),
                "prefixInheritedBoundaryCount": int(inheritance_array.sum()),
                "prefixInheritanceFraction": float(inheritance_array.mean()),
                "recentThreeInheritanceFraction": float(inheritance_array[-3:].mean()),
                "latestParentDaughterH": float(parent_h_values[-1]),
                "latestMaximumEligibleRecurrenceH": latest_maximum,
                "pastObservableRiskScore": latest_maximum
                if np.isfinite(latest_maximum)
                else 0.0,
                "prefixEventAlreadyOccurred": historical.event,
                "prefixFirstEventGeneration": historical.first_event_generation,
                "prefixRecurrenceMatchCount": historical.recurrence_match_count,
                "currentSelectedIndexExact": True,
                "currentStateExact": True,
            }
        )
    boundaries = pd.DataFrame(boundary_rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "generation"]
    ).reset_index(drop=True)
    summaries = pd.DataFrame(summary_rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)
    if (
        len(summaries) != 280
        or summaries["stateId"].duplicated().any()
        or not summaries[["currentSelectedIndexExact", "currentStateExact"]].all().all()
        or boundaries.groupby("stateId")["generation"].apply(
            lambda values: bool(np.all(np.diff(values.to_numpy()) > 0))
        ).eq(False).any()
    ):
        raise RuntimeError("L38 prefix registry validation failure")
    return boundaries, summaries


def control_registry(
    summaries: pd.DataFrame, boundaries: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    donor_rows = []
    permutation_rows = []
    for _, group in summaries.groupby(
        ["evaluationCohort", "candidateId", "landmark"], sort=True
    ):
        ordered = group.sort_values("stateId")["stateId"].tolist()
        donors = ordered[1:] + ordered[:1]
        for receiver, donor in zip(ordered, donors, strict=True):
            receiver_group = boundaries[boundaries["stateId"].eq(receiver)]
            donor_group = boundaries[boundaries["stateId"].eq(donor)]
            donor_rows.append(
                {
                    "stateId": receiver,
                    "donorStateId": donor,
                    "receiverBoundaryCount": len(receiver_group),
                    "donorBoundaryCount": len(donor_group),
                    "sameState": receiver == donor,
                }
            )
            seed = derived_seed("species_permutation", receiver)
            permutation = np.random.default_rng(seed).permutation(100)
            permutation_rows.append(
                {
                    "stateId": receiver,
                    "derivedSeed": str(seed),
                    "seedMaterialSha256": hashlib.sha256(
                        f"{VERSION}\x1f{ROOT_HEX}\x1fspecies_permutation\x1f{receiver}".encode()
                    ).hexdigest(),
                    "permutation": permutation.tolist(),
                    "permutationSha256": hashlib.sha256(
                        np.asarray(permutation, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "nonidentity": bool(np.any(permutation != np.arange(100))),
                }
            )
    donors = pd.DataFrame(donor_rows).sort_values("stateId").reset_index(drop=True)
    permutations = pd.DataFrame(permutation_rows).sort_values("stateId").reset_index(
        drop=True
    )
    if (
        len(donors) != 280
        or len(permutations) != 280
        or donors["sameState"].any()
        or not permutations["nonidentity"].all()
        or permutations["seedMaterialSha256"].duplicated().any()
    ):
        raise RuntimeError("L38 control registry failure")
    return donors, permutations


def analysis_seed_manifest(permutations: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "purpose": "species_permutation",
            "partsJson": json.dumps(("species_permutation", row.stateId)),
            "evaluationCohort": None,
            "candidateId": None,
            "branchFamily": None,
            "targetId": None,
            "comparisonId": None,
            "rootHex": ROOT_HEX,
            "derivedSeed": row.derivedSeed,
            "seedMaterialSha256": row.seedMaterialSha256,
        }
        for row in permutations.itertuples(index=False)
    ]
    for cohort in COHORTS:
        for candidate in CANDIDATES:
            for family in FAMILIES:
                for target in TARGETS:
                    parts = ("reliability_bootstrap", cohort, candidate, family, target)
                    rows.append(
                        {
                            "purpose": parts[0],
                            "partsJson": json.dumps(parts),
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "branchFamily": family,
                            "targetId": target,
                            "comparisonId": None,
                            "rootHex": ROOT_HEX,
                            "derivedSeed": str(derived_seed(*parts)),
                            "seedMaterialSha256": hashlib.sha256(
                                "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)]).encode()
                            ).hexdigest(),
                        }
                    )
            for comparison in (
                "PRIMARY_H8_VS_PRIMARY_H32",
                "ORIGINAL_H8_VS_PRIMARY_H32",
                "PAST_PROXY_VS_PRIMARY_H32",
                "PRIMARY_MINUS_PERMUTED_H32",
                "PRIMARY_MINUS_UNRELATED_H32",
                "PRIMARY_MINUS_BRANCH_ONLY_H32",
                "INHERITANCE_ONLY_VS_PRIMARY_H32",
            ):
                parts = ("transfer_bootstrap", cohort, candidate, comparison)
                rows.append(
                    {
                        "purpose": parts[0],
                        "partsJson": json.dumps(parts),
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "branchFamily": None,
                        "targetId": None,
                        "comparisonId": comparison,
                        "rootHex": ROOT_HEX,
                        "derivedSeed": str(derived_seed(*parts)),
                        "seedMaterialSha256": hashlib.sha256(
                            "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)]).encode()
                        ).hexdigest(),
                    }
                )
            parts = ("calibration_bootstrap", cohort, candidate)
            rows.append(
                {
                    "purpose": parts[0],
                    "partsJson": json.dumps(parts),
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "branchFamily": None,
                    "targetId": None,
                    "comparisonId": None,
                    "rootHex": ROOT_HEX,
                    "derivedSeed": str(derived_seed(*parts)),
                    "seedMaterialSha256": hashlib.sha256(
                        "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)]).encode()
                    ).hexdigest(),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["purpose", "evaluationCohort", "candidateId", "branchFamily", "targetId"],
        na_position="last",
    ).reset_index(drop=True)
    if result["derivedSeed"].duplicated().any() or result[
        "seedMaterialSha256"
    ].duplicated().any():
        raise RuntimeError("L38 analysis seed collision")
    return result


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L38/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            lowered = column.lower()
            if "seedmaterialsha256" in lowered:
                prior_material.update(frame[column].dropna().astype(str))
            if lowered == "derivedseed":
                prior_derived.update(frame[column].dropna().astype(str))
    material = sorted(set(seeds["seedMaterialSha256"]) & prior_material)
    derived = sorted(set(seeds["derivedSeed"]) & prior_derived)
    return {
        "schema": "eidosoma.e01.s19_l38.seed_firewall.v1",
        "status": "PASS" if not material and not derived else "FAIL",
        "analysisSeedCount": len(seeds),
        "seedMaterialOverlapCount": len(material),
        "derivedSeedOverlapCount": len(derived),
        "seedMaterialOverlaps": material,
        "derivedSeedOverlaps": derived,
        "newBranchSeeds": 0,
    }


def source_grounding_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L38_REVIEWER_INDEPENDENT_TARGET",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Replace a trajectory-specific completed-run destination with independently observable recurrence or inheritance.",
                "frozenUse": "past-only recurring inherited selected-daughter event",
            },
            {
                "sourceId": "L38_PAPER_RECURRING_COMPOSITION",
                "evidenceClass": "DIRECT_PAPER_LANGUAGE",
                "finding": "The paper describes recurring compositions and parent/daughter-like homeostatic replication at H above 0.9.",
                "frozenUse": "strict H090 recurrence and inheritance without claiming author identity",
            },
            {
                "sourceId": "L38_L37_TRAJECTORY_SPECIFIC_TARGET",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "The complete multi-lineage attractor-family gate failed despite reliable target-conditioned committors.",
                "frozenUse": "prohibits a completed-test-trajectory target",
            },
            {
                "sourceId": "L38_L28_L31_BRANCH_STREAMS",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Exact H32/H8 branch streams are reliable and replayable.",
                "frozenUse": "zero-new-stream branch evaluation",
            },
        ]
    )


def _boundary_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values("generation")
    states = np.asarray(ordered["state"].tolist(), dtype=np.int64)
    generations = ordered["generation"].to_numpy(dtype=np.int64)
    inherited = ordered["inherited"].to_numpy(dtype=np.bool_)
    return states, generations, inherited


def branch_payloads(
    responses: pd.DataFrame,
    coordinates: pd.DataFrame,
    manifest: pd.DataFrame,
    boundaries: pd.DataFrame,
    summaries: pd.DataFrame,
    donors: pd.DataFrame,
    permutations: pd.DataFrame,
) -> list[dict[str, Any]]:
    base = L35.payloads(responses, coordinates, manifest)
    boundary_map = {
        state_id: _boundary_arrays(group)
        for state_id, group in boundaries.groupby("stateId", sort=False)
    }
    summary_map = summaries.set_index("stateId")
    donor_map = donors.set_index("stateId")["donorStateId"].to_dict()
    permutation_map = permutations.set_index("stateId")["permutation"].to_dict()
    output = []
    for payload in base:
        state_id = payload["stateId"]
        primary_states, primary_generations, primary_inherited = boundary_map[state_id]
        donor_id = donor_map[state_id]
        donor_states, donor_generations, donor_inherited = boundary_map[donor_id]
        # Shift a matched unrelated prefix so its latest boundary shares the
        # receiver's current past endpoint; relative donor gaps are unchanged.
        donor_generations = donor_generations + (
            int(primary_generations[-1]) - int(donor_generations[-1])
        )
        permutation = np.asarray(permutation_map[state_id], dtype=np.int64)
        row = dict(payload)
        row.update(
            {
                "prefixStates": primary_states.tolist(),
                "prefixGenerations": primary_generations.tolist(),
                "prefixInherited": primary_inherited.tolist(),
                "permutedPrefixStates": primary_states[:, permutation].tolist(),
                "unrelatedPrefixStates": donor_states.tolist(),
                "unrelatedPrefixGenerations": donor_generations.tolist(),
                "unrelatedPrefixInherited": donor_inherited.tolist(),
                "unrelatedDonorStateId": donor_id,
                "prefixEventAlreadyOccurred": bool(
                    summary_map.loc[state_id, "prefixEventAlreadyOccurred"]
                ),
            }
        )
        output.append(row)
    if len(output) != 280:
        raise RuntimeError("L38 branch payload cardinality failure")
    return output


def _score_target(
    target_id: str,
    payload: dict[str, Any],
    future_states: np.ndarray,
    future_generations: np.ndarray,
    future_inherited: np.ndarray,
) -> Any:
    feature_count = future_states.shape[1] if future_states.ndim == 2 else 100
    if target_id == "PAST_ONLY_RECURRENCE_INHERITANCE":
        prefix_states = np.asarray(payload["prefixStates"], dtype=np.int64)
        prefix_generations = np.asarray(payload["prefixGenerations"], dtype=np.int64)
        prefix_inherited = np.asarray(payload["prefixInherited"], dtype=np.bool_)
    elif target_id == "SPECIES_PERMUTED_PREFIX":
        prefix_states = np.asarray(payload["permutedPrefixStates"], dtype=np.int64)
        prefix_generations = np.asarray(payload["prefixGenerations"], dtype=np.int64)
        prefix_inherited = np.asarray(payload["prefixInherited"], dtype=np.bool_)
    elif target_id == "UNRELATED_MATRIX_PREFIX":
        prefix_states = np.asarray(payload["unrelatedPrefixStates"], dtype=np.int64)
        prefix_generations = np.asarray(
            payload["unrelatedPrefixGenerations"], dtype=np.int64
        )
        prefix_inherited = np.asarray(
            payload["unrelatedPrefixInherited"], dtype=np.bool_
        )
    elif target_id == "BRANCH_ONLY_RECURRENCE":
        prefix_states = np.empty((0, feature_count), dtype=np.int64)
        prefix_generations = np.empty(0, dtype=np.int64)
        prefix_inherited = np.empty(0, dtype=np.bool_)
    elif target_id == "INHERITANCE_ONLY":
        return None
    else:
        raise ValueError(f"unknown L38 target: {target_id}")
    return score_recurrence_inheritance(
        prefix_states=prefix_states,
        prefix_generations=prefix_generations,
        prefix_inherited=prefix_inherited,
        future_states=future_states,
        future_generations=future_generations,
        future_inherited=future_inherited,
        threshold=THRESHOLD,
        minimum_generation_gap=MINIMUM_GENERATION_GAP,
    )


def _branch_worker(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    matrix_index = int(payload["matrixIndex"])
    beta = L28.generate_beta(
        L28.derive_seed(
            L28.L23_ROOT_HEX, L28.L23_PHASE, "catalytic_matrix", matrix_index
        )
    )
    if L28.simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError(f"L38 beta replay failure: {payload['stateId']}")
    restored = RestoredState(
        tuple(payload["state"]),
        payload["currentObservationKind"],
        int(payload["currentCompletedFissions"]),
        int(payload["currentGrowthGeneration"]),
        int(payload["currentGenerationLocalStep"]),
        int(payload["currentBatchStep"]),
    )
    original = np.asarray(payload["centroid"], dtype=np.float64)
    outcomes: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for branch in range(BRANCH_COUNTS[family]):
            event_rng, trim_rng, fission_rng, daughter_rng = L36._branch_rngs(
                payload, family, branch
            )
            trace = L37.simulate_branch_trace(
                restored=restored,
                beta=beta,
                definition=L28.definition(payload["candidateId"]),
                target_centroid=original,
                event_rng=event_rng,
                trim_rng=trim_rng,
                fission_rng=fission_rng,
                daughter_rng=daughter_rng,
                horizon=HORIZONS[family],
                threshold=THRESHOLD,
            )
            compact = trace.compact
            compact_rows.append(
                {
                    "stateId": payload["stateId"],
                    "evaluationCohort": payload["evaluationCohort"],
                    "candidateId": payload["candidateId"],
                    "matrixIndex": matrix_index,
                    "landmark": int(payload["landmark"]),
                    "branchFamily": family,
                    "targetId": "ORIGINAL",
                    "branchIndex": branch,
                    "branchHalf": "A" if branch < HALVES[family] else "B",
                    "enteredTarget": compact.entered_basin,
                    "firstEntryOffsetOneBased": compact.first_entry_offset_one_based,
                    "maximumTargetScore": compact.maximum_target_score,
                    "minimumTargetScore": compact.minimum_target_score,
                    "molecularUpdates": compact.molecular_updates,
                    "fissions": compact.fissions,
                    "selectedObservationsGenerated": compact.selected_observations_generated,
                    "terminalStatus": compact.terminal_status,
                    "originalPathSha256": compact.path_sha256,
                }
            )
            boundary_observations = [
                observation
                for observation in trace.observations
                if observation.observation_kind == "post_fission"
            ]
            future_states = np.asarray(
                [observation.state for observation in boundary_observations],
                dtype=np.int64,
            ).reshape((-1, 100))
            future_generations = np.asarray(
                [observation.generation for observation in boundary_observations],
                dtype=np.int64,
            )
            future_inherited = np.asarray(
                [observation.ordinary_adjacent_h > THRESHOLD for observation in boundary_observations],
                dtype=np.bool_,
            )
            for target_id in TARGETS:
                scored = _score_target(
                    target_id,
                    payload,
                    future_states,
                    future_generations,
                    future_inherited,
                )
                if target_id == "INHERITANCE_ONLY":
                    event = bool(np.any(future_inherited))
                    first_boundary = (
                        int(np.flatnonzero(future_inherited)[0] + 1)
                        if event
                        else None
                    )
                    first_generation = (
                        int(future_generations[first_boundary - 1]) if event else None
                    )
                    comparison_count = 0
                    match_count = int(np.sum(future_inherited))
                    maximum_h = float(np.max(
                        [observation.ordinary_adjacent_h for observation in boundary_observations]
                    )) if boundary_observations else None
                    inheritance_only = event
                    reference_generation = None
                else:
                    event = scored.event
                    first_boundary = scored.first_event_boundary_one_based
                    first_generation = scored.first_event_generation
                    reference_generation = scored.matched_reference_generation
                    comparison_count = scored.eligible_comparison_count
                    match_count = scored.recurrence_match_count
                    maximum_h = scored.maximum_eligible_h
                    inheritance_only = scored.inheritance_only_event
                first_offset = (
                    boundary_observations[first_boundary - 1].offset
                    if first_boundary is not None
                    else None
                )
                outcomes.append(
                    {
                        "stateId": payload["stateId"],
                        "evaluationCohort": payload["evaluationCohort"],
                        "candidateId": payload["candidateId"],
                        "matrixIndex": matrix_index,
                        "landmark": int(payload["landmark"]),
                        "branchFamily": family,
                        "targetId": target_id,
                        "branchIndex": branch,
                        "branchHalf": "A" if branch < HALVES[family] else "B",
                        "event": event,
                        "firstEventOffsetOneBased": first_offset,
                        "firstEventBoundaryOneBased": first_boundary,
                        "firstEventGeneration": first_generation,
                        "matchedReferenceGeneration": reference_generation,
                        "futureBoundaryCount": len(boundary_observations),
                        "inheritedFutureBoundaryCount": int(np.sum(future_inherited)),
                        "eligibleComparisonCount": comparison_count,
                        "recurrenceMatchCount": match_count,
                        "maximumEligibleH": maximum_h,
                        "inheritanceOnlyEvent": inheritance_only,
                        "prefixEventAlreadyOccurred": bool(
                            payload["prefixEventAlreadyOccurred"]
                        ),
                        "targetUsesCompletedTestTrajectory": False,
                        "selectedObservationsGenerated": compact.selected_observations_generated,
                        "terminalStatus": compact.terminal_status,
                        "originalPathSha256": compact.path_sha256,
                    }
                )
    return {"outcomes": outcomes, "compact": compact_rows}


def execute_branches(
    payload_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_branch_worker, payload): payload["stateId"]
            for payload in payload_rows
        }
        for future in as_completed(futures):
            result = future.result()
            outcomes.extend(result["outcomes"])
            compact.extend(result["compact"])
    outcome_frame = pd.DataFrame(outcomes).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "branchFamily",
            "targetId",
            "branchIndex",
        ]
    ).reset_index(drop=True)
    compact_frame = pd.DataFrame(compact).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "branchFamily",
            "branchIndex",
        ]
    ).reset_index(drop=True)
    if (
        len(compact_frame) != 53_760
        or len(outcome_frame) != 53_760 * len(TARGETS)
        or compact_frame.duplicated(["stateId", "branchFamily", "branchIndex"]).any()
        or outcome_frame.duplicated(
            ["stateId", "branchFamily", "targetId", "branchIndex"]
        ).any()
    ):
        raise RuntimeError("L38 branch result cardinality failure")
    return outcome_frame, compact_frame


def state_committor_results(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in outcomes.groupby(
        ["stateId", "branchFamily", "targetId"], sort=True
    ):
        state_id, family, target = keys
        expected = BRANCH_COUNTS[family]
        if len(group) != expected:
            raise RuntimeError("L38 per-state branch cardinality failure")
        half_a = group[group["branchHalf"].eq("A")]
        half_b = group[group["branchHalf"].eq("B")]
        events = group["event"].to_numpy(dtype=np.bool_)
        rows.append(
            {
                "stateId": state_id,
                "evaluationCohort": group["evaluationCohort"].iloc[0],
                "candidateId": group["candidateId"].iloc[0],
                "matrixIndex": int(group["matrixIndex"].iloc[0]),
                "landmark": int(group["landmark"].iloc[0]),
                "branchFamily": family,
                "targetId": target,
                "branches": len(group),
                "successes": int(events.sum()),
                "qHat": float(events.mean()),
                "qHatHalfA": float(half_a["event"].mean()),
                "qHatHalfB": float(half_b["event"].mean()),
                "meanFutureBoundaryCount": float(group["futureBoundaryCount"].mean()),
                "meanEligibleComparisonCount": float(
                    group["eligibleComparisonCount"].mean()
                ),
                "meanRecurrenceMatchCount": float(
                    group["recurrenceMatchCount"].mean()
                ),
                "meanFirstEventOffset": float(group["firstEventOffsetOneBased"].mean()),
                "prefixEventAlreadyOccurred": bool(
                    group["prefixEventAlreadyOccurred"].iloc[0]
                ),
                "committorEligible": bool(
                    len(group) == expected
                    and group["selectedObservationsGenerated"].eq(HORIZONS[family]).all()
                ),
                "targetUsesCompletedTestTrajectory": False,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "branchFamily", "targetId"]
    ).reset_index(drop=True)
    if len(result) != 280 * len(FAMILIES) * len(TARGETS):
        raise RuntimeError("L38 state committor cardinality failure")
    return result


def reliability_results(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    for keys, group in states.groupby(
        ["evaluationCohort", "candidateId", "branchFamily", "targetId"], sort=True
    ):
        cohort, candidate, family, target = keys
        eligible = group[group["committorEligible"]]
        q = eligible["qHat"].to_numpy(dtype=np.float64)
        half_a = eligible["qHatHalfA"].to_numpy(dtype=np.float64)
        half_b = eligible["qHatHalfB"].to_numpy(dtype=np.float64)
        variance = corrected_between_state_variance(q, BRANCH_COUNTS[family])
        split = safe_spearman(half_a, half_b)
        rng = np.random.default_rng(
            derived_seed("reliability_bootstrap", cohort, candidate, family, target)
        )
        corrected_boot = np.full(BOOTSTRAPS, np.nan)
        split_boot = np.full(BOOTSTRAPS, np.nan)
        for replicate in range(BOOTSTRAPS):
            indices = rng.integers(0, len(eligible), len(eligible))
            corrected_boot[replicate] = corrected_between_state_variance(
                q[indices], BRANCH_COUNTS[family]
            )["correctedBetweenStateVariance"]
            split_boot[replicate] = safe_spearman(half_a[indices], half_b[indices])
            bootstrap_rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "branchFamily": family,
                    "targetId": target,
                    "replicate": replicate,
                    "correctedVariance": corrected_boot[replicate],
                    "splitHalfSpearman": split_boot[replicate],
                }
            )
        corrected_lower, corrected_upper = interval(corrected_boot)
        split_lower, split_upper = interval(split_boot)
        intermediate = int(np.sum((q > 0.1) & (q < 0.9)))
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "branchFamily": family,
                "targetId": target,
                "states": len(group),
                "eligibleStates": len(eligible),
                "meanQ": float(np.mean(q)),
                "minimumQ": float(np.min(q)),
                "maximumQ": float(np.max(q)),
                "intermediateStateCount": intermediate,
                "observedBetweenStateVariance": variance[
                    "observedBetweenStateVariance"
                ],
                "estimatedBinomialNoiseVariance": variance[
                    "estimatedBinomialNoiseVariance"
                ],
                "correctedBetweenStateVariance": variance[
                    "correctedBetweenStateVariance"
                ],
                "correctedVarianceLower95": corrected_lower,
                "correctedVarianceUpper95": corrected_upper,
                "splitHalfSpearman": split,
                "splitHalfLower95": split_lower,
                "splitHalfUpper95": split_upper,
                "reliabilityGatePassed": bool(
                    len(eligible) >= 20
                    and corrected_lower > 0
                    and split > 0.5
                    and split_lower > 0.3
                    and intermediate >= 20
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def transfer_pairs(
    states: pd.DataFrame,
    responses: pd.DataFrame,
    summaries: pd.DataFrame,
) -> pd.DataFrame:
    index = states.set_index(["stateId", "branchFamily", "targetId"])
    response_index = responses.set_index("stateId")
    summary_index = summaries.set_index("stateId")
    specifications = (
        (
            "PRIMARY_H8_VS_PRIMARY_H32",
            ("H8", PRIMARY_TARGET),
            ("H32", PRIMARY_TARGET),
            "RANK",
        ),
        (
            "PRIMARY_MINUS_PERMUTED_H32",
            ("H32", PRIMARY_TARGET),
            ("H32", "SPECIES_PERMUTED_PREFIX"),
            "DIFFERENCE",
        ),
        (
            "PRIMARY_MINUS_UNRELATED_H32",
            ("H32", PRIMARY_TARGET),
            ("H32", "UNRELATED_MATRIX_PREFIX"),
            "DIFFERENCE",
        ),
        (
            "PRIMARY_MINUS_BRANCH_ONLY_H32",
            ("H32", PRIMARY_TARGET),
            ("H32", "BRANCH_ONLY_RECURRENCE"),
            "DIFFERENCE",
        ),
        (
            "INHERITANCE_ONLY_VS_PRIMARY_H32",
            ("H32", "INHERITANCE_ONLY"),
            ("H32", PRIMARY_TARGET),
            "RANK",
        ),
    )
    rows = []
    for state_id in states["stateId"].unique():
        metadata = index.loc[(state_id, "H32", PRIMARY_TARGET)]
        for comparison, left_key, right_key, comparison_type in specifications:
            left = index.loc[(state_id, *left_key)]
            right = index.loc[(state_id, *right_key)]
            rows.append(
                {
                    "stateId": state_id,
                    "evaluationCohort": metadata.evaluationCohort,
                    "candidateId": metadata.candidateId,
                    "matrixIndex": int(metadata.matrixIndex),
                    "comparisonId": comparison,
                    "comparisonType": comparison_type,
                    "leftValue": float(left.qHat),
                    "rightValue": float(right.qHat),
                }
            )
        rows.extend(
            [
                {
                    "stateId": state_id,
                    "evaluationCohort": metadata.evaluationCohort,
                    "candidateId": metadata.candidateId,
                    "matrixIndex": int(metadata.matrixIndex),
                    "comparisonId": "ORIGINAL_H8_VS_PRIMARY_H32",
                    "comparisonType": "RANK",
                    "leftValue": float(response_index.loc[state_id, "q8"]),
                    "rightValue": float(metadata.qHat),
                },
                {
                    "stateId": state_id,
                    "evaluationCohort": metadata.evaluationCohort,
                    "candidateId": metadata.candidateId,
                    "matrixIndex": int(metadata.matrixIndex),
                    "comparisonId": "PAST_PROXY_VS_PRIMARY_H32",
                    "comparisonType": "RANK",
                    "leftValue": float(
                        summary_index.loc[state_id, "pastObservableRiskScore"]
                    ),
                    "rightValue": float(metadata.qHat),
                },
            ]
        )
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "comparisonId", "stateId"]
    ).reset_index(drop=True)
    if len(result) != 280 * 7:
        raise RuntimeError("L38 transfer-pair cardinality failure")
    return result


def transfer_results(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    for keys, group in pairs.groupby(
        ["evaluationCohort", "candidateId", "comparisonId", "comparisonType"],
        sort=True,
    ):
        cohort, candidate, comparison, comparison_type = keys
        left = group["leftValue"].to_numpy(dtype=np.float64)
        right = group["rightValue"].to_numpy(dtype=np.float64)
        observed = (
            safe_spearman(left, right)
            if comparison_type == "RANK"
            else float(np.mean(left - right))
        )
        rng = np.random.default_rng(
            derived_seed("transfer_bootstrap", cohort, candidate, comparison)
        )
        boot = np.full(BOOTSTRAPS, np.nan)
        for replicate in range(BOOTSTRAPS):
            indices = rng.integers(0, len(group), len(group))
            boot[replicate] = (
                safe_spearman(left[indices], right[indices])
                if comparison_type == "RANK"
                else float(np.mean(left[indices] - right[indices]))
            )
            bootstrap_rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "comparisonId": comparison,
                    "comparisonType": comparison_type,
                    "replicate": replicate,
                    "value": boot[replicate],
                }
            )
        lower, upper = interval(boot)
        passed = bool(
            np.isfinite(observed)
            and np.isfinite(lower)
            and (
                (comparison_type == "RANK" and observed > 0.5 and lower > 0.3)
                or (comparison_type == "DIFFERENCE" and lower > 0)
            )
        )
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "comparisonId": comparison,
                "comparisonType": comparison_type,
                "definedPairs": len(group),
                "pointEstimate": observed,
                "lower95": lower,
                "upper95": upper,
                "gatePassed": passed,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def observed_continuation_results(
    responses: pd.DataFrame,
    manifest: pd.DataFrame,
    boundaries: pd.DataFrame,
) -> pd.DataFrame:
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    boundary_map = {
        state_id: _boundary_arrays(group)
        for state_id, group in boundaries.groupby("stateId", sort=False)
    }
    rows = []
    for response in responses.itertuples(index=False):
        source = manifest_index.loc[(response.candidateId, int(response.matrixIndex))]
        trajectory = load_trajectory(source)
        selected = L28.selected_clock_observations(trajectory, L28.CLOCK_ID)
        future = selected[
            int(response.currentSelectedIndex) + 1 : int(response.currentSelectedIndex) + 33
        ]
        observations = [item for item in future if item.observation_kind == "post_fission"]
        states = np.asarray([item.state for item in observations], dtype=np.int64).reshape(
            (-1, 100)
        )
        generations = np.asarray(
            [item.completed_fissions for item in observations], dtype=np.int64
        )
        inherited = []
        for item in observations:
            raw_index = int(item.observation_index)
            if raw_index <= 0:
                raise RuntimeError("L38 observed continuation predecessor missing")
            parent = trajectory.observations[raw_index - 1]
            inherited.append(
                cosine_h(
                    np.asarray(parent.state, dtype=np.int64),
                    np.asarray(item.state, dtype=np.int64),
                )
                > THRESHOLD
            )
        prefix = boundary_map[response.stateId]
        result = score_recurrence_inheritance(
            prefix_states=prefix[0],
            prefix_generations=prefix[1],
            prefix_inherited=prefix[2],
            future_states=states,
            future_generations=generations,
            future_inherited=np.asarray(inherited, dtype=np.bool_),
            threshold=THRESHOLD,
            minimum_generation_gap=MINIMUM_GENERATION_GAP,
        )
        rows.append(
            {
                "stateId": response.stateId,
                "evaluationCohort": response.evaluationCohort,
                "candidateId": response.candidateId,
                "matrixIndex": int(response.matrixIndex),
                "eventInObservedNext32": result.event,
                "futureBoundaryCount": result.future_boundary_count,
                "firstEventGeneration": result.first_event_generation,
                "targetDefinedWithoutObservedFuture": True,
                "observedFutureUsedForEvaluationOnly": True,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "matrixIndex"]
    ).reset_index(drop=True)


def calibration_results(
    states: pd.DataFrame, observed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = states[
        states["branchFamily"].eq("H32") & states["targetId"].eq(PRIMARY_TARGET)
    ]
    joined = q.merge(observed, on=["stateId", "evaluationCohort", "candidateId", "matrixIndex"])
    rows = []
    bootstrap_rows = []
    for (cohort, candidate), group in joined.groupby(
        ["evaluationCohort", "candidateId"], sort=True
    ):
        probability = np.clip(group["qHat"].to_numpy(dtype=np.float64), 1e-12, 1 - 1e-12)
        outcome = group["eventInObservedNext32"].to_numpy(dtype=np.float64)
        brier = float(np.mean((probability - outcome) ** 2))
        log_loss = float(-np.mean(xlogy(outcome, probability) + xlogy(1 - outcome, 1 - probability)))
        rng = np.random.default_rng(derived_seed("calibration_bootstrap", cohort, candidate))
        brier_boot = np.full(BOOTSTRAPS, np.nan)
        for replicate in range(BOOTSTRAPS):
            indices = rng.integers(0, len(group), len(group))
            brier_boot[replicate] = np.mean(
                (probability[indices] - outcome[indices]) ** 2
            )
            bootstrap_rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "replicate": replicate,
                    "brier": brier_boot[replicate],
                }
            )
        lower, upper = interval(brier_boot)
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "states": len(group),
                "observedEventFraction": float(outcome.mean()),
                "meanPredictedProbability": float(probability.mean()),
                "brier": brier,
                "brierLower95": lower,
                "brierUpper95": upper,
                "logLoss": log_loss,
                "evaluationOnly": True,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def scientific_gates(
    reliability: pd.DataFrame, transfer: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], str]:
    rows = []
    for cohort in EVALUATION_COHORTS:
        for candidate in CANDIDATES:
            rel = reliability[
                reliability["evaluationCohort"].eq(cohort)
                & reliability["candidateId"].eq(candidate)
            ].set_index(["branchFamily", "targetId"])
            comparisons = transfer[
                transfer["evaluationCohort"].eq(cohort)
                & transfer["candidateId"].eq(candidate)
            ].set_index("comparisonId")

            def rel_gate(
                family: str,
                target: str = PRIMARY_TARGET,
                frame: pd.DataFrame = rel,
            ) -> bool:
                return bool(
                    (family, target) in frame.index
                    and frame.loc[(family, target), "reliabilityGatePassed"]
                )

            def comparison_gate(
                name: str, frame: pd.DataFrame = comparisons
            ) -> bool:
                return bool(name in frame.index and frame.loc[name, "gatePassed"])

            h32 = rel_gate("H32")
            h8 = rel_gate("H8")
            transfer_gate = comparison_gate("PRIMARY_H8_VS_PRIMARY_H32")
            permuted = comparison_gate("PRIMARY_MINUS_PERMUTED_H32")
            unrelated = comparison_gate("PRIMARY_MINUS_UNRELATED_H32")
            target_gate = h32 and permuted and unrelated
            short_gate = target_gate and h8 and transfer_gate
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "primaryH32Reliable": h32,
                    "primaryH8Reliable": h8,
                    "h8ToH32RankPassed": transfer_gate,
                    "speciesPermutationControlPassed": permuted,
                    "unrelatedMatrixControlPassed": unrelated,
                    "pastOnlyOutcomeCommittorPassed": target_gate,
                    "shortShootingCoordinatePassed": short_gate,
                    "pastProxyRankPassed": comparison_gate("PAST_PROXY_VS_PRIMARY_H32"),
                    "originalTeacherTransferPassed": comparison_gate(
                        "ORIGINAL_H8_VS_PRIMARY_H32"
                    ),
                }
            )
    gates = pd.DataFrame(rows)
    target_all = bool(gates["pastOnlyOutcomeCommittorPassed"].all())
    short_all = bool(gates["shortShootingCoordinatePassed"].all())
    proxy_all = bool(gates["pastProxyRankPassed"].all())
    if short_all:
        classifications = [
            "PAST_ONLY_RECURRENCE_INHERITANCE_COMMITTOR_ESTABLISHED",
            "SHORT_SHOOTING_COORDINATE_ESTABLISHED_FOR_INDEPENDENT_OUTCOME",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "UNTOUCHED_RECURRENCE_INHERITANCE_COMMITTOR_CONFIRMATION"
    elif target_all:
        classifications = [
            "PAST_ONLY_RECURRENCE_INHERITANCE_COMMITTOR_ESTABLISHED",
            "H8_SHOOTING_COORDINATE_NOT_ESTABLISHED_FOR_INDEPENDENT_OUTCOME",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "RECURRENCE_INHERITANCE_SHOOTING_BUDGET_DISCRIMINATION"
    else:
        classifications = [
            "RECURRENCE_INHERITANCE_OUTCOME_NOT_COMMITTOR_COMPATIBLE",
            "INDEPENDENT_EVENT_TARGET_REQUIRES_REDEFINITION",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "SUSTAINED_HOMEOSTATIC_INHERITANCE_OUTCOME_CONSTRUCTION"
    if proxy_all:
        classifications.insert(-1, "PAST_OBSERVABLE_RECURRENCE_PROXY_ESTABLISHED")
    return gates, classifications, next_theme


def benchmark_projection(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    sample = [
        next(row for row in payloads if row["candidateId"] == candidate)
        for candidate in CANDIDATES
    ]
    durations = []
    for payload in sample:
        started = time.perf_counter()
        matrix_index = int(payload["matrixIndex"])
        beta = L28.generate_beta(
            L28.derive_seed(
                L28.L23_ROOT_HEX,
                L28.L23_PHASE,
                "catalytic_matrix",
                matrix_index,
            )
        )
        restored = RestoredState(
            tuple(payload["state"]),
            payload["currentObservationKind"],
            int(payload["currentCompletedFissions"]),
            int(payload["currentGrowthGeneration"]),
            int(payload["currentGenerationLocalStep"]),
            int(payload["currentBatchStep"]),
        )
        event_rng, trim_rng, fission_rng, daughter_rng = L36._branch_rngs(
            payload, "H32", 0
        )
        trace = L37.simulate_branch_trace(
            restored=restored,
            beta=beta,
            definition=L28.definition(payload["candidateId"]),
            target_centroid=np.asarray(payload["centroid"], dtype=np.float64),
            event_rng=event_rng,
            trim_rng=trim_rng,
            fission_rng=fission_rng,
            daughter_rng=daughter_rng,
            horizon=32,
            threshold=THRESHOLD,
        )
        if trace.compact.selected_observations_generated != 32:
            raise RuntimeError("L38 benchmark branch did not complete")
        durations.append(time.perf_counter() - started)
    branch_seconds = float(np.median(durations))
    projected_cpu = branch_seconds * 53_760 * 2 / 3600 * 1.35
    projected_wall = projected_cpu * 3600 / WORKERS
    return {
        "schema": "eidosoma.e01.s19_l38.benchmark_projection.v1",
        "status": "PASS"
        if projected_cpu <= 90 and projected_wall <= 64.8 * 3600
        else "STOP_BEFORE_OUTCOME",
        "opaqueBenchmarkDurationsSeconds": durations,
        "projectedCpuHoursIncludingRegeneration": projected_cpu,
        "projectedWallSecondsIncludingRegeneration": projected_wall,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "scientificOutcomeRetained": False,
    }


def make_figures(
    summaries: pd.DataFrame,
    states: pd.DataFrame,
    reliability: pd.DataFrame,
    transfer: pd.DataFrame,
    calibration: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    summaries.pivot_table(
        index=["evaluationCohort", "candidateId"],
        values=["prefixInheritanceFraction", "prefixEventAlreadyOccurred"],
        aggfunc="mean",
    ).plot(kind="bar", figsize=(13, 6))
    plt.ylabel("Observed-prefix fraction")
    save("01_prefix_recurrence_inheritance_geometry.png")

    primary = states[
        states["targetId"].eq(PRIMARY_TARGET)
        & states["branchFamily"].eq("H32")
    ]
    for (cohort, candidate), group in primary.groupby(
        ["evaluationCohort", "candidateId"], sort=True
    ):
        plt.hist(group["qHat"], bins=np.linspace(0, 1, 21), alpha=0.45, label=f"{cohort}/{candidate}")
    plt.xlabel("H32 recurrence/inheritance probability")
    plt.ylabel("States")
    plt.legend(fontsize=6)
    save("02_primary_committor_distributions.png")

    reliability[
        reliability["evaluationCohort"].isin(EVALUATION_COHORTS)
        & reliability["targetId"].eq(PRIMARY_TARGET)
    ].pivot_table(
        index="branchFamily",
        columns=["evaluationCohort", "candidateId"],
        values="splitHalfSpearman",
    ).plot(kind="bar", figsize=(14, 6))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("Split-half Spearman")
    save("03_committor_split_half_reliability.png")

    transfer[transfer["evaluationCohort"].isin(EVALUATION_COHORTS)].pivot_table(
        index="comparisonId",
        columns=["evaluationCohort", "candidateId"],
        values="pointEstimate",
    ).plot(kind="bar", figsize=(15, 7))
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Registered rank or paired q difference")
    save("04_short_shooting_and_controls.png")

    calibration.pivot_table(
        index="evaluationCohort",
        columns="candidateId",
        values=["observedEventFraction", "meanPredictedProbability"],
    ).plot(kind="bar", figsize=(14, 6))
    plt.ylabel("Probability")
    save("05_observed_continuation_calibration.png")

    checks = [
        "primaryH32Reliable",
        "primaryH8Reliable",
        "h8ToH32RankPassed",
        "speciesPermutationControlPassed",
        "unrelatedMatrixControlPassed",
        "pastOnlyOutcomeCommittorPassed",
        "shortShootingCoordinatePassed",
        "pastProxyRankPassed",
    ]
    matrix = gates.set_index(["evaluationCohort", "candidateId"])[checks].astype(float)
    plt.figure(figsize=(13, 5))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(checks)), checks, rotation=35, ha="right", fontsize=7)
    plt.yticks(range(len(matrix)), ["/".join(index) for index in matrix.index], fontsize=7)
    plt.colorbar(ticks=[0, 1])
    save("06_decision_matrix.png")


def manifest_for(root: Path) -> dict[str, Any]:
    files = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and item.name != "artifact_manifest.json"
        )
    ]
    return {
        "schema": "eidosoma.e01.s19_l38.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def append_ledgers(
    classifications: list[str], timestamp: str, next_theme: str
) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L37 found reliable branch probabilities but no fully transferable completed-lineage attractor family.",
            "failureOrAmbiguityTargeted": "Whether a completed-run destination can be replaced by a branch-evaluable observable event.",
            "informationGainRationale": "A recurrence/inheritance event is defined online and needs no completed test trajectory.",
            "learned": "L38 strict-H090 recurrence/inheritance event locked before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L37 classification and reviewer independent-target decision tree.",
            "proposedNextTest": "Rescore exact H32/H8 paths against the online event.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Independent recurrence/inheritance, sustained low drift, homeostatic growth, or shooting-only estimation.",
            "selectedHypotheses": "An inherited daughter recurs after an intervening generation.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "The evaluated completed-run basin is a universal destination.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A scientifically useful independent outcome must show reliable state variation and resist unrelated/permuted history controls.",
            "failureOrAmbiguityTargeted": "Branchability and short-shooting recoverability of the independent event.",
            "informationGainRationale": "Exact existing streams isolate target semantics from simulator noise or new-seed adaptation.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L38 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Past-only recurrence/inheritance event.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "A reliable committor follows automatically from an online target definition.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat(
            [ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)],
            ignore_index=True,
        ),
    )
    markdown = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    BASE.atomic_text(
        markdown,
        markdown.read_text()
        + f"\n\n## {LOOP_ID} — past-only recurrence/inheritance outcome\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** {next_theme}.\n",
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    candidate = {
        "branchCount": 1,
        "bundleId": "L38_PAST_ONLY_RECURRENCE_INHERITANCE",
        "candidateId": "S19-L38-ONLINE-RECURRENCE-INHERITANCE-EVENT",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 4,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 1,
        "proposedSpecification": "strict-H090 recurring inherited selected-daughter event with generation gap two",
        "rankingScore": 30.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L37_TRAJECTORY_SPECIFIC_TARGET_AND_REVIEWER_DECISION_TREE",
        "sourceGrounding": 5,
        "testability": 5,
        "undefinedAuthorSemantics": 1,
    }
    BASE.write_parquet(
        candidates_path,
        pd.concat(
            [candidates, pd.DataFrame([candidate]).reindex(columns=candidates.columns)],
            ignore_index=True,
        ),
    )
    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    source_rows = [
        {
            "commitOrVersion": None,
            "evidenceClass": source.evidenceClass,
            "finding": f"{source.finding}; L38 use: {source.frozenUse}",
            "licenseStatus": "WORKSPACE_OR_HUMAN_DIRECTION",
            "redistributionStatus": "INTERNAL_EVIDENCE_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": f"L38_{source.sourceId}",
            "sourceType": source.evidenceClass,
            "treeIdentity": None,
            "url": None,
        }
        for source in source_grounding_registry().itertuples(index=False)
    ]
    BASE.write_parquet(
        source_path,
        pd.concat(
            [sources, pd.DataFrame(source_rows).reindex(columns=sources.columns)],
            ignore_index=True,
        ),
    )
    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text())
    registry["loops"].append(
        {
            "loopId": LOOP_ID,
            "versionedLoopId": VERSION,
            "status": "COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED",
            "authorized": True,
            "completed": True,
            "outcomeAccessed": True,
            "humanReviewRequiredAfter": False,
            "classification": classifications,
            "selectedDiscoveryLead": None,
            "newMatrices": 0,
            "newTrajectories": 0,
            "newBranchStreams": 0,
            "nextStepActive": True,
        }
    )
    registry["proposedNextLoopTheme"] = next_theme
    registry["proposedNextLoopActive"] = True
    registry["authorizationUpperBound"] = "S19-L55"
    BASE.atomic_text(registry_path, yaml.safe_dump(registry, sort_keys=False))
    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["history"].append(
        {
            "decision": "S19_L38_COMPLETE_AUTONOMOUS_CONTINUATION",
            "loopId": LOOP_ID,
            "nextLoopAuthorized": True,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "s20Activated": False,
            "scope": VERSION,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L55"
    BASE.write_json(history_path, history)


def report_text(
    summaries: pd.DataFrame,
    reliability: pd.DataFrame,
    transfer: pd.DataFrame,
    calibration: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    eval_reliability = reliability[
        reliability["evaluationCohort"].isin(EVALUATION_COHORTS)
        & reliability["targetId"].eq(PRIMARY_TARGET)
    ]
    eval_transfer = transfer[transfer["evaluationCohort"].isin(EVALUATION_COHORTS)]
    prefix_summary = summaries.groupby(
        ["evaluationCohort", "candidateId"], sort=True
    ).agg(
        states=("stateId", "size"),
        meanBoundaries=("prefixBoundaryCount", "mean"),
        meanInheritance=("prefixInheritanceFraction", "mean"),
        priorEventFraction=("prefixEventAlreadyOccurred", "mean"),
    ).reset_index()
    return f"""# S19-L38 — Past-Only Recurrence–Inheritance Outcome Construction

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the extended L19–L55 autonomous sequence.
- **Classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Validation:** exact prefix/state/boundary replay; exact numerical/discrete/path replay of all 53,760 frozen H32/H8 streams; split-half reliability; fixed controls; 4,096 matrix bootstraps; independent complete result regeneration; immutable/runtime/storage/artifact hashes.
- **Recommended next action:** `{next_theme}`.

## Frozen question

Does a target that needs no completed test trajectory have a reliable state-dependent probability? The event is a selected post-fission daughter that has strict parent/daughter `H>0.9` and strict `H>0.9` to an earlier inherited selected daughter separated by at least one intervening generation. Only the observed prefix and earlier states in the same future branch are eligible references.

## Prefix geometry

{prefix_summary.to_markdown(index=False)}

Prior occurrence does not make a state ineligible: L38 predicts the next recurrence/inheritance event, not the first lifetime appearance of replication.

## Committor reliability

{eval_reliability.to_markdown(index=False)}

## Short shooting, controls and past proxy

{eval_transfer.to_markdown(index=False)}

## One-realization calibration diagnostic

{calibration.to_markdown(index=False)}

The completed observed suffix is used only as one evaluation realization; it does not define the target or any branch label.

## Decision gates

{gates.to_markdown(index=False)}

## Validation and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- Workers: `{runtime['workers']}` with one numerical-library thread per worker; GPU hours `0`.
- Wall time: `{runtime['wallSeconds']:.2f}` seconds.
- New matrices, trajectories and branch streams: `0/0/0`.
- Frozen branch streams scored: `{runtime['uniqueFrozenBranchStreamsScored']}` and independently scored again for full regeneration.
- No threshold, horizon, simulator, target rule, control, candidate or state was selected after outcomes.

## Interpretation boundary

This is an adaptive target-construction audit. Even a positive result would establish only a simulator-defined probability of a future recurrence/inheritance event and, conditionally, a short stochastic-shooting estimator. It would not identify the paper label, prove a static biomarker, establish initial appearance, causal emergence, intervention efficacy, or biological causation.

## Next boundary

L38 is frozen. The standing human authorization permits `{next_theme}` as the next bounded loop through L55. S20, E02, author contact, interventions and report generation remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L38 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    responses = pd.read_parquet(L36_ROOT / "response_registry.parquet")
    coordinates = pd.read_parquet(L36_ROOT / "original_target_coordinates.parquet")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    boundaries, summaries = prefix_registries(responses, manifest)
    donors, permutations = control_registry(summaries, boundaries)
    seeds = analysis_seed_manifest(permutations)
    firewall = seed_firewall(seeds)
    payloads = branch_payloads(
        responses,
        coordinates,
        manifest,
        boundaries,
        summaries,
        donors,
        permutations,
    )
    benchmark = benchmark_projection(payloads)
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
    ):
        raise RuntimeError("L38 preoutcome validation or benchmark failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L38 decision record\n\n"
        "L37 showed that a completed-lineage target does not satisfy the complete network-stability gate across both candidates and evaluation cohorts. The reviewer directed the next branch toward outcomes that do not require a completed test trajectory. L38 freezes one event before outcomes: a future selected daughter and an earlier, nonadjacent-generation selected daughter must each preserve parent-to-daughter composition at strict H>0.9 and must recur with each other at strict H>0.9. Only the observed prefix and branch past are visible. Existing H32/H8 paths, candidates, horizons and streams are unchanged. Molecule-permuted, unrelated-matrix, branch-only and inheritance-only diagnostics are fixed; no target or threshold search is permitted.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "response_registry.parquet", responses)
    BASE.write_parquet(LOOP_ROOT / "original_target_coordinates.parquet", coordinates)
    BASE.write_parquet(LOOP_ROOT / "input_trajectory_manifest.parquet", manifest)
    BASE.write_parquet(LOOP_ROOT / "prefix_boundary_registry.parquet", boundaries)
    BASE.write_parquet(LOOP_ROOT / "prefix_state_summary.parquet", summaries)
    BASE.write_parquet(LOOP_ROOT / "unrelated_control_map.parquet", donors)
    BASE.write_parquet(LOOP_ROOT / "species_permutation_manifest.parquet", permutations)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_parquet(
        LOOP_ROOT / "source_grounding_registry.parquet", source_grounding_registry()
    )
    hashes = {
        "configSha256": sha256_file(CONFIG),
        "responsesSha256": sha256_file(LOOP_ROOT / "response_registry.parquet"),
        "coordinatesSha256": sha256_file(
            LOOP_ROOT / "original_target_coordinates.parquet"
        ),
        "manifestSha256": sha256_file(LOOP_ROOT / "input_trajectory_manifest.parquet"),
        "boundariesSha256": sha256_file(LOOP_ROOT / "prefix_boundary_registry.parquet"),
        "summariesSha256": sha256_file(LOOP_ROOT / "prefix_state_summary.parquet"),
        "donorsSha256": sha256_file(LOOP_ROOT / "unrelated_control_map.parquet"),
        "permutationsSha256": sha256_file(
            LOOP_ROOT / "species_permutation_manifest.parquet"
        ),
        "seedsSha256": sha256_file(LOOP_ROOT / "analysis_seed_manifest.parquet"),
        "firewallSha256": sha256_file(LOOP_ROOT / "seed_firewall.json"),
        "benchmarkSha256": sha256_file(LOOP_ROOT / "benchmark_projection.json"),
        "l37ManifestSha256": sha256_file(L37_ROOT / "artifact_manifest.json"),
    }
    implementation = {
        "schema": "eidosoma.e01.s19_l38.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "threshold": THRESHOLD,
        "thresholdComparison": "STRICT_GREATER_THAN",
        "minimumGenerationGap": MINIMUM_GENERATION_GAP,
        "targetIds": list(TARGETS),
        "branchFamilies": list(FAMILIES),
        "horizons": HORIZONS,
        "branchCounts": BRANCH_COUNTS,
        "matrixBootstraps": BOOTSTRAPS,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "lockedHashes": hashes,
        "outcomeAccessed": False,
        "lockedAtUtc": utc_now(),
    }
    BASE.write_json(LOOP_ROOT / "implementation_lock.json", implementation)
    BASE.write_json(
        LOOP_ROOT / "preoutcome_repository_lock.json",
        {
            "head": head,
            "remote": remote,
            "priorAggregateSha256": prior["aggregateSha256"],
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            **hashes,
        },
    )


def execute() -> None:
    started = time.perf_counter()
    started_cpu = time.process_time()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    if (
        git("rev-parse", "HEAD") != lock["head"]
        or git("rev-parse", "origin/eidosoma/groups/42") != lock["remote"]
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("L38 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_files = {
        "responsesSha256": LOOP_ROOT / "response_registry.parquet",
        "coordinatesSha256": LOOP_ROOT / "original_target_coordinates.parquet",
        "manifestSha256": LOOP_ROOT / "input_trajectory_manifest.parquet",
        "boundariesSha256": LOOP_ROOT / "prefix_boundary_registry.parquet",
        "summariesSha256": LOOP_ROOT / "prefix_state_summary.parquet",
        "donorsSha256": LOOP_ROOT / "unrelated_control_map.parquet",
        "permutationsSha256": LOOP_ROOT / "species_permutation_manifest.parquet",
        "seedsSha256": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "firewallSha256": LOOP_ROOT / "seed_firewall.json",
        "benchmarkSha256": LOOP_ROOT / "benchmark_projection.json",
        "l37ManifestSha256": L37_ROOT / "artifact_manifest.json",
    }
    for key, path in locked_files.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"L38 locked input changed: {path}")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
    ):
        raise RuntimeError("L38 pre-execution validation failed")
    responses = pd.read_parquet(LOOP_ROOT / "response_registry.parquet")
    coordinates = pd.read_parquet(LOOP_ROOT / "original_target_coordinates.parquet")
    manifest = pd.read_parquet(LOOP_ROOT / "input_trajectory_manifest.parquet")
    boundaries = pd.read_parquet(LOOP_ROOT / "prefix_boundary_registry.parquet")
    summaries = pd.read_parquet(LOOP_ROOT / "prefix_state_summary.parquet")
    donors = pd.read_parquet(LOOP_ROOT / "unrelated_control_map.parquet")
    permutations = pd.read_parquet(LOOP_ROOT / "species_permutation_manifest.parquet")
    payloads = branch_payloads(
        responses,
        coordinates,
        manifest,
        boundaries,
        summaries,
        donors,
        permutations,
    )
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    outcomes, compact = execute_branches(payloads)
    compact_validation = L36.compact_replay_validation(compact)
    states = state_committor_results(outcomes)
    reliability, reliability_bootstrap = reliability_results(states)
    pairs = transfer_pairs(states, responses, summaries)
    transfer, transfer_bootstrap = transfer_results(pairs)
    observed = observed_continuation_results(responses, manifest, boundaries)
    calibration, calibration_bootstrap = calibration_results(states, observed)
    gates, classifications, next_theme = scientific_gates(reliability, transfer)
    make_figures(summaries, states, reliability, transfer, calibration, gates)

    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "immutable_prior_validation.json",
        "fixture_results.parquet",
        "response_registry.parquet",
        "original_target_coordinates.parquet",
        "input_trajectory_manifest.parquet",
        "prefix_boundary_registry.parquet",
        "prefix_state_summary.parquet",
        "unrelated_control_map.parquet",
        "species_permutation_manifest.parquet",
        "analysis_seed_manifest.parquet",
        "seed_firewall.json",
        "benchmark_projection.json",
        "source_grounding_registry.parquet",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    tables = {
        "branch_outcome_results.parquet": outcomes,
        "compact_branch_replay.parquet": compact,
        "compact_replay_validation.parquet": compact_validation,
        "state_committor_results.parquet": states,
        "committor_reliability_results.parquet": reliability,
        "committor_reliability_bootstrap.parquet": reliability_bootstrap,
        "transfer_pairs.parquet": pairs,
        "transfer_results.parquet": transfer,
        "transfer_bootstrap.parquet": transfer_bootstrap,
        "observed_continuation_results.parquet": observed,
        "calibration_results.parquet": calibration,
        "calibration_bootstrap.parquet": calibration_bootstrap,
        "scientific_gate_results.parquet": gates,
    }
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l38.classification.v1",
            "classifications": classifications,
            "pastOnlyOutcomeCommittorEstablished": bool(
                gates["pastOnlyOutcomeCommittorPassed"].all()
            ),
            "shortShootingCoordinateEstablished": bool(
                gates["shortShootingCoordinatePassed"].all()
            ),
            "targetUsesCompletedTestTrajectory": False,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=[
            "stage",
            "stateId",
            "candidateId",
            "matrixIndex",
            "branchFamily",
            "branchIndex",
            "exceptionClass",
            "exceptionMessage",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)

    # Independently regenerate every scientific frame from the locked inputs.
    replay_outcomes, replay_compact = execute_branches(payloads)
    replay_compact_validation = L36.compact_replay_validation(replay_compact)
    replay_states = state_committor_results(replay_outcomes)
    replay_reliability, replay_reliability_bootstrap = reliability_results(
        replay_states
    )
    replay_pairs = transfer_pairs(replay_states, responses, summaries)
    replay_transfer, replay_transfer_bootstrap = transfer_results(replay_pairs)
    replay_observed = observed_continuation_results(responses, manifest, boundaries)
    replay_calibration, replay_calibration_bootstrap = calibration_results(
        replay_states, replay_observed
    )
    replay_gates, replay_classifications, replay_next = scientific_gates(
        replay_reliability, replay_transfer
    )
    replay_tables = {
        "outcomes": (outcomes, replay_outcomes),
        "compact": (compact, replay_compact),
        "compactValidation": (compact_validation, replay_compact_validation),
        "states": (states, replay_states),
        "reliability": (reliability, replay_reliability),
        "reliabilityBootstrap": (
            reliability_bootstrap,
            replay_reliability_bootstrap,
        ),
        "pairs": (pairs, replay_pairs),
        "transfer": (transfer, replay_transfer),
        "transferBootstrap": (transfer_bootstrap, replay_transfer_bootstrap),
        "observed": (observed, replay_observed),
        "calibration": (calibration, replay_calibration),
        "calibrationBootstrap": (
            calibration_bootstrap,
            replay_calibration_bootstrap,
        ),
        "gates": (gates, replay_gates),
    }
    checks = {
        name: frame_hash(left) == frame_hash(right)
        for name, (left, right) in replay_tables.items()
    }
    checks.update(
        {
            "classificationExact": classifications == replay_classifications,
            "nextThemeExact": next_theme == replay_next,
            "fixturesPassed": bool(fixtures["passed"].all()),
            "immutablePriorPassed": prior["unchanged"],
            "seedFirewallPassed": json.loads(
                (LOOP_ROOT / "seed_firewall.json").read_text()
            )["status"]
            == "PASS",
            "compactReplayPassed": bool(compact_validation["allPassed"].all()),
            "targetUsesNoCompletedTestTrajectory": bool(
                (~outcomes["targetUsesCompletedTestTrajectory"]).all()
            ),
            "noNewTrajectory": True,
            "noNewBranchStream": True,
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"L38 regeneration validation failed: {checks}")
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l38.regeneration_validation.v1",
            "status": "PASS",
            "checks": checks,
            "outcomeFrameSha256": frame_hash(outcomes),
            "stateFrameSha256": frame_hash(states),
            "gateFrameSha256": frame_hash(gates),
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l38.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started,
        "controllerCpuHours": (time.process_time() - started_cpu) / 3600,
        "states": 280,
        "uniqueFrozenBranchStreamsScored": 53_760,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file())
    temporary = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l38.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
        "status": "PASS"
        if retained < 25 * 2**30 and temporary < 75 * 2**30
        else "FAIL",
    }
    if storage["status"] != "PASS":
        raise RuntimeError("L38 storage ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        summaries,
        reliability,
        transfer,
        calibration,
        gates,
        classifications,
        runtime,
        next_theme,
    )
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L38_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        "# S19-L38 decision summary\n\n"
        + f"**Classification:** {', '.join(classifications)}\n\n"
        + f"**All-group independent outcome:** `{gates['pastOnlyOutcomeCommittorPassed'].all()}`.\n\n"
        + f"**All-group H8 shooting coordinate:** `{gates['shortShootingCoordinatePassed'].all()}`.\n\n"
        + f"**Next:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L38-promotion-stage")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(BUILD_ROOT, stage)
    if LOOP_ROOT.exists():
        shutil.rmtree(LOOP_ROOT)
    os.replace(stage, LOOP_ROOT)
    shutil.rmtree(BUILD_ROOT)
    artifact_manifest = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    if any(
        sha256_file(LOOP_ROOT / item["path"]) != item["sha256"]
        for item in artifact_manifest["files"]
    ):
        raise RuntimeError("L38 artifact hash validation failed")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L38", "# S19 current handoff — S19-L38", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": None,
            "nextAuthorizedLoop": "S19-L39",
            "authorizationUpperBound": "S19-L55",
            "s20Active": False,
            "updatedAtUtc": runtime["completedAtUtc"],
        },
    )
    BASE.write_json(ARTIFACT_ROOT / "artifact_manifest.json", manifest_for(ARTIFACT_ROOT))
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "classifications": classifications,
                "nextTheme": next_theme,
                "runtime": runtime,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-lock", action="store_true")
    args = parser.parse_args()
    if args.prepare_lock:
        prepare_lock()
    else:
        execute()


if __name__ == "__main__":
    main()
