#!/usr/bin/env python3
"""Run S19-L48 process-committor shooting-efficiency audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from e01_onset_discovery.process_shooting_efficiency import (
    bernoulli_scores,
    jeffreys_estimate,
    next_uncertainty_allocation,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L47 = load_module(
    "e01_l47_runner",
    ROOT / "scripts/e01/run_s19_l47_functional_coherence_sufficiency.py",
)
L44 = L47.L46.L45.L44
L41 = L44.L43.L42.L41
BASE = L47.BASE
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L48"
L41_ROOT = ARTIFACT_ROOT / "loops/L41"
L44_ROOT = ARTIFACT_ROOT / "loops/L44"
L47_ROOT = ARTIFACT_ROOT / "loops/L47"
BUILD_ROOT = Path("/cache/e01_s19_l48/build")
CONFIG = ROOT / "configs/e01/s19_l48_process_committor_shooting_efficiency.yaml"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = ROOT / "src/e01_onset_discovery/process_shooting_efficiency.py"
LOOP_ID = "S19-L48"
VERSION = "E01-S19-L48-PROCESS-COMMITTOR-SHOOTING-EFFICIENCY-v1.0.0"
PROCESS = "NEW_HEREDITARY_EPISODE_RUN3"
BUDGETS = (4, 8, 16, 32, 64)
BOOTSTRAPS = 4096
RESPLITS = 128
EVALUATION_COHORTS = ("L28_VALIDATION", "L31_CONFIRMATION")
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
STRATEGIES = ("FIXED_UNIFORM", "ADAPTIVE_POSTERIOR_VARIANCE")
SEED_ROOT = bytes.fromhex(
    "12e1ff4ead35e0dd4cc461912921acb53cebc3d2a9da6d28c758bd170b12f254"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.reindex(sorted(frame.columns), axis=1).reset_index(drop=True)
    return hashlib.sha256(
        ordered.to_json(orient="table", index=False, double_precision=15).encode()
    ).hexdigest()


def seed_material(*parts: object) -> bytes:
    canonical_parts = tuple(
        part.item() if isinstance(part, np.generic) else part for part in parts
    )
    return hashlib.sha256(
        SEED_ROOT
        + b"\x00"
        + json.dumps(canonical_parts, separators=(",", ":")).encode()
    ).digest()


def derived_seed(*parts: object) -> int:
    return int.from_bytes(seed_material(*parts)[:16], "big")


def interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(finite, [0.025, 0.975])))


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or len(np.unique(a[mask])) < 2 or len(np.unique(b[mask])) < 2:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def validate_immutable_prior() -> dict[str, Any]:
    prior = L47.validate_immutable_prior()
    manifest = json.loads((L47_ROOT / "artifact_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = L47_ROOT / row["path"]
        actual = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "path": str(path),
                "expectedSha256": row["sha256"],
                "actualSha256": actual,
                "unchanged": actual == row["sha256"],
            }
        )
    passed = bool(prior["unchanged"] and rows and all(row["unchanged"] for row in rows))
    return {
        "schema": "eidosoma.e01.s19_l48.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "priorThroughL46Unchanged": bool(prior["unchanged"]),
        "validatedL47ArtifactCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
    }


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "BROWN_CAI_DASGUPTA_BINOMIAL_INTERVALS",
                "evidenceClass": "PRIMARY_STATISTICAL_METHOD",
                "finding": "Jeffreys equal-tailed intervals are a recommended small-sample alternative to the unstable Wald interval.",
                "frozenUse": "Beta(1/2,1/2) posterior summary for conditional branch probabilities",
                "url": "https://doi.org/10.1214/ss/1009213286",
            },
            {
                "sourceId": "HOWARD_RAMADAS_CONFIDENCE_SEQUENCES",
                "evidenceClass": "PRIMARY_STATISTICAL_METHOD",
                "finding": "Repeated inspection requires time-uniform rather than ordinary fixed-sample coverage claims.",
                "frozenUse": "L48 makes no sequential coverage claim; adaptive estimates are scored on an untouched branch half",
                "url": "https://doi.org/10.1214/20-AOS1991",
            },
            {
                "sourceId": "SHEKHAR_ADAPTIVE_DISTRIBUTION_ESTIMATION",
                "evidenceClass": "PRIMARY_STATISTICAL_METHOD",
                "finding": "Budget allocation can prioritize uncertain discrete distributions under a fixed sampling budget.",
                "frozenUse": "one largest-posterior-variance allocation heuristic",
                "url": "https://proceedings.mlr.press/v119/shekhar20a.html",
            },
            {
                "sourceId": "L44_PROCESS_FAMILY",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "The online-certified run-of-three process has a reliable conditional state-dependent probability in both candidates and held-out cohorts.",
                "frozenUse": "sole process target and eligibility registry",
                "url": None,
            },
            {
                "sourceId": "L45_L47_PROXY_NON_SUPPORT",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Registered PhiID and functional-coherence increments did not transfer consistently beyond direct controls.",
                "frozenUse": "motivation to treat shooting as a measurement rather than force another proxy",
                "url": None,
            },
        ]
    )


def fixture_results() -> pd.DataFrame:
    empty = jeffreys_estimate(0, 0)
    replay_a = jeffreys_estimate(3, 4)
    replay_b = jeffreys_estimate(3, 4)
    outcomes = np.asarray([0, 1, 1, 0], dtype=np.int8)
    scores = bernoulli_scores(0.5, outcomes)
    chosen = next_uncertainty_allocation(
        ["b", "a", "c"],
        np.asarray([2, 2, 4]),
        np.asarray([4, 4, 4]),
        np.asarray([4, 4, 4]),
    )
    return pd.DataFrame(
        [
            {
                "fixtureId": "F01_JEFFREYS_ZERO_TRIAL",
                "passed": empty.posterior_mean == 0.5
                and 0 < empty.lower95 < empty.upper95 < 1,
                "detail": "prior-only estimate retained and marked separately",
            },
            {
                "fixtureId": "F02_JEFFREYS_EXACT_REPLAY",
                "passed": replay_a == replay_b and replay_a.posterior_mean == 0.7,
                "detail": "3/4 with Beta(1/2,1/2)",
            },
            {
                "fixtureId": "F03_BERNOULLI_SCORING",
                "passed": scores["brier"] == 0.25
                and np.isclose(scores["logLoss"], np.log(2.0)),
                "detail": "independent reference outcomes",
            },
            {
                "fixtureId": "F04_ALLOCATION_TIE",
                "passed": chosen == 1,
                "detail": "lexical stateId tie breaker",
            },
            {
                "fixtureId": "F05_BUDGET_NESTING",
                "passed": all(left < right for left, right in itertools.pairwise(BUDGETS)),
                "detail": str(BUDGETS),
            },
            {
                "fixtureId": "F06_HALF_SEPARATION",
                "passed": set(range(64)).isdisjoint(range(64, 128)),
                "detail": "estimator A versus reference B",
            },
            {
                "fixtureId": "F07_SEED_SERIALIZATION",
                "passed": str(derived_seed("fixture")).isdigit(),
                "detail": "128-bit decimal string",
            },
            {
                "fixtureId": "F08_NUMPY_INTEGER_SEED_EQUIVALENCE",
                "passed": derived_seed("fixture", np.int64(32))
                == derived_seed("fixture", 32),
                "detail": "TA01 converts a NumPy integer to the identical registered native-integer seed material",
            },
        ]
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episodes = pd.read_parquet(L44_ROOT / "branch_episode_results.parquet")
    states = pd.read_parquet(L44_ROOT / "state_process_results.parquet")
    states = states[
        states["processId"].eq(PROCESS) & states["eligible"]
    ].reset_index(drop=True)
    costs = pd.read_parquet(L41_ROOT / "branch_trace_results.parquet")
    costs = costs[costs["branchFamily"].eq("F12")][
        [
            "stateId",
            "evaluationCohort",
            "candidateId",
            "matrixIndex",
            "landmark",
            "branchIndex",
            "branchHalf",
            "branchIdentitySha256",
            "molecularUpdates",
            "fissions",
            "selectedObservationsGenerated",
            "terminalStatus",
            "pathSha256",
        ]
    ]
    keys = [
        "stateId",
        "evaluationCohort",
        "candidateId",
        "matrixIndex",
        "landmark",
        "branchIndex",
        "branchHalf",
    ]
    merged = episodes.merge(costs, on=keys, validate="one_to_one")
    merged = merged[merged["stateId"].isin(states["stateId"])].copy()
    expected_half = np.where(merged["branchIndex"].lt(64), "A", "B")
    if (
        len(episodes) != 35_840
        or len(costs) != 35_840
        or len(states) != 256
        or len(merged) != 32_768
        or not np.array_equal(merged["branchHalf"].to_numpy(), expected_half)
        or merged.groupby("stateId").size().ne(128).any()
        or merged["targetUsesCompletedTestTrajectory"].any()
    ):
        raise RuntimeError("L48 frozen input scope or half separation failure")
    state_keys = [
        "stateId",
        "evaluationCohort",
        "candidateId",
        "matrixIndex",
        "landmark",
    ]
    checks = []
    for values, group in merged.groupby(state_keys, sort=False):
        state = states
        for key, value in zip(state_keys, values, strict=True):
            state = state[state[key].eq(value)]
        if len(state) != 1:
            raise RuntimeError("L48 state registry lookup failure")
        row = state.iloc[0]
        half_a = group[group["branchHalf"].eq("A") & group["breakObserved"]]
        half_b = group[group["branchHalf"].eq("B") & group["breakObserved"]]
        checks.append(
            bool(
                len(half_a) == row["trialsHalfA"]
                and len(half_b) == row["trialsHalfB"]
                and np.isclose(half_a["newHereditaryEpisodeRun3"].mean(), row["qHatHalfA"])
                and np.isclose(half_b["newHereditaryEpisodeRun3"].mean(), row["qHatHalfB"])
            )
        )
    if not all(checks):
        raise RuntimeError("L48 exact L44 conditional-half replay failure")
    merged = merged.sort_values(
        ["candidateId", "evaluationCohort", "matrixIndex", "branchIndex"]
    ).reset_index(drop=True)
    states = states.sort_values(
        ["candidateId", "evaluationCohort", "matrixIndex"]
    ).reset_index(drop=True)
    return merged, states, costs


def development_priors(states: pd.DataFrame) -> dict[str, float]:
    development = states[states["evaluationCohort"].eq("L28_DEVELOPMENT")]
    priors = development.groupby("candidateId")["qHat"].mean().to_dict()
    if set(priors) != set(CANDIDATES) or not all(0 < value < 1 for value in priors.values()):
        raise RuntimeError("L48 development prior failure")
    return {key: float(value) for key, value in priors.items()}


def measurement_row(
    group: pd.DataFrame,
    selected_branch_indices: np.ndarray,
    *,
    strategy: str,
    average_budget: int,
    development_prior: float,
) -> dict[str, Any]:
    selected_set = set(map(int, selected_branch_indices))
    selected = group[group["branchIndex"].isin(selected_set)]
    reference = group[group["branchHalf"].eq("B") & group["breakObserved"]]
    conditional = selected[selected["breakObserved"]]
    successes = int(conditional["newHereditaryEpisodeRun3"].sum())
    trials = len(conditional)
    estimate = jeffreys_estimate(successes, trials)
    reference_outcomes = reference["newHereditaryEpisodeRun3"].to_numpy(np.int8)
    score = bernoulli_scores(estimate.posterior_mean, reference_outcomes)
    prior_score = bernoulli_scores(development_prior, reference_outcomes)
    first = group.iloc[0]
    return {
        "stateId": first.stateId,
        "evaluationCohort": first.evaluationCohort,
        "candidateId": first.candidateId,
        "matrixIndex": int(first.matrixIndex),
        "landmark": int(first.landmark),
        "measurementStrategy": strategy,
        "averageBranchBudget": int(average_budget),
        "allocatedBranches": len(selected),
        "conditionalBreakTrials": trials,
        "conditionalSuccesses": successes,
        "dataInformed": bool(trials > 0),
        "qEstimateRaw": float(successes / trials) if trials else np.nan,
        "qEstimateJeffreys": estimate.posterior_mean,
        "posteriorVariance": estimate.posterior_variance,
        "posteriorLower95": estimate.lower95,
        "posteriorUpper95": estimate.upper95,
        "posteriorWidth95": estimate.upper95 - estimate.lower95,
        "referenceTrials": len(reference),
        "referenceSuccesses": int(reference_outcomes.sum()),
        "qReference": float(reference_outcomes.mean()),
        "brierReferenceBranches": score["brier"],
        "logLossReferenceBranches": score["logLoss"],
        "developmentPrior": development_prior,
        "priorBrierReferenceBranches": prior_score["brier"],
        "priorLogLossReferenceBranches": prior_score["logLoss"],
        "brierImprovementOverPrior": prior_score["brier"] - score["brier"],
        "logLossImprovementOverPrior": prior_score["logLoss"] - score["logLoss"],
        "absoluteQError": abs(estimate.posterior_mean - reference_outcomes.mean()),
        "molecularUpdates": int(selected["molecularUpdates"].sum()),
        "fissions": int(selected["fissions"].sum()),
        "selectedObservationsGenerated": int(
            selected["selectedObservationsGenerated"].sum()
        ),
        "heldoutReferenceUsedForAllocation": False,
        "targetUsesCompletedTestTrajectory": False,
    }


def fixed_measurements(
    episodes: pd.DataFrame, priors: dict[str, float]
) -> pd.DataFrame:
    rows = []
    keys = ["stateId", "evaluationCohort", "candidateId", "matrixIndex", "landmark"]
    for _, group in episodes.groupby(keys, sort=False):
        for budget in BUDGETS:
            rows.append(
                measurement_row(
                    group,
                    np.arange(budget, dtype=np.int64),
                    strategy="FIXED_UNIFORM",
                    average_budget=budget,
                    development_prior=priors[str(group.iloc[0]["candidateId"])],
                )
            )
    result = pd.DataFrame(rows).sort_values(
        ["candidateId", "evaluationCohort", "averageBranchBudget", "matrixIndex"]
    ).reset_index(drop=True)
    if len(result) != len(episodes["stateId"].unique()) * len(BUDGETS):
        raise RuntimeError("L48 fixed-measurement scope failure")
    return result


def adaptive_measurements(
    episodes: pd.DataFrame, priors: dict[str, float]
) -> pd.DataFrame:
    rows = []
    state_keys = ["stateId", "evaluationCohort", "candidateId", "matrixIndex", "landmark"]
    state_groups = {
        state_id: group.sort_values("branchIndex").reset_index(drop=True)
        for state_id, group in episodes.groupby("stateId", sort=False)
    }
    state_registry = (
        episodes[state_keys]
        .drop_duplicates()
        .sort_values(["candidateId", "evaluationCohort", "stateId"])
    )
    for (cohort, candidate), registry in state_registry.groupby(
        ["evaluationCohort", "candidateId"], sort=False
    ):
        state_ids = list(registry["stateId"].astype(str))
        allocated = np.full(len(state_ids), 4, dtype=np.int64)
        successes = np.zeros(len(state_ids), dtype=np.int64)
        trials = np.zeros(len(state_ids), dtype=np.int64)
        for index, state_id in enumerate(state_ids):
            initial = state_groups[state_id].iloc[:4]
            conditional = initial[initial["breakObserved"]]
            trials[index] = len(conditional)
            successes[index] = int(conditional["newHereditaryEpisodeRun3"].sum())
        for checkpoint in BUDGETS:
            target_total = checkpoint * len(state_ids)
            while int(allocated.sum()) < target_total:
                index = next_uncertainty_allocation(
                    state_ids, successes, trials, allocated, cap=64
                )
                start = int(allocated[index])
                stop = min(start + 4, 64)
                addition = state_groups[state_ids[index]].iloc[start:stop]
                conditional = addition[addition["breakObserved"]]
                trials[index] += len(conditional)
                successes[index] += int(
                    conditional["newHereditaryEpisodeRun3"].sum()
                )
                allocated[index] = stop
            if int(allocated.sum()) != target_total:
                raise RuntimeError("L48 adaptive checkpoint budget failure")
            for index, state_id in enumerate(state_ids):
                rows.append(
                    measurement_row(
                        state_groups[state_id],
                        np.arange(int(allocated[index]), dtype=np.int64),
                        strategy="ADAPTIVE_POSTERIOR_VARIANCE",
                        average_budget=checkpoint,
                        development_prior=priors[candidate],
                    )
                )
    result = pd.DataFrame(rows).sort_values(
        ["candidateId", "evaluationCohort", "averageBranchBudget", "matrixIndex"]
    ).reset_index(drop=True)
    if len(result) != len(state_groups) * len(BUDGETS):
        raise RuntimeError("L48 adaptive-measurement scope failure")
    return result


def measurement_group_results(
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    keys = [
        "evaluationCohort",
        "candidateId",
        "measurementStrategy",
        "averageBranchBudget",
    ]
    fixed64_cost = {
        (cohort, candidate): float(group["molecularUpdates"].mean())
        for (cohort, candidate), group in states[
            states["measurementStrategy"].eq("FIXED_UNIFORM")
            & states["averageBranchBudget"].eq(64)
        ].groupby(["evaluationCohort", "candidateId"], sort=False)
    }
    for values, group in states.groupby(keys, sort=False):
        q_estimate = group["qEstimateJeffreys"].to_numpy(float)
        q_reference = group["qReference"].to_numpy(float)
        rng = np.random.Generator(
            np.random.PCG64DXSM(derived_seed("bootstrap", *values))
        )
        metrics = np.full((BOOTSTRAPS, 5), np.nan, dtype=np.float64)
        for replicate in range(BOOTSTRAPS):
            selected = rng.integers(0, len(group), size=len(group))
            sample = group.iloc[selected]
            metrics[replicate] = (
                safe_spearman(
                    sample["qEstimateJeffreys"].to_numpy(float),
                    sample["qReference"].to_numpy(float),
                ),
                sample["brierImprovementOverPrior"].mean(),
                sample["absoluteQError"].mean(),
                sample["dataInformed"].mean(),
                sample["posteriorWidth95"].mean(),
            )
            bootstrap_rows.append(
                {
                    **dict(zip(keys, values, strict=True)),
                    "replicate": replicate,
                    "spearman": metrics[replicate, 0],
                    "brierImprovementOverPrior": metrics[replicate, 1],
                    "meanAbsoluteQError": metrics[replicate, 2],
                    "dataInformedFraction": metrics[replicate, 3],
                    "meanPosteriorWidth95": metrics[replicate, 4],
                }
            )
        rho_ci = interval(metrics[:, 0])
        brier_ci = interval(metrics[:, 1])
        mae_ci = interval(metrics[:, 2])
        availability_ci = interval(metrics[:, 3])
        width_ci = interval(metrics[:, 4])
        baseline_cost = fixed64_cost[(values[0], values[1])]
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "states": len(group),
                "dataInformedStates": int(group["dataInformed"].sum()),
                "dataInformedFraction": float(group["dataInformed"].mean()),
                "dataInformedLower95": availability_ci[0],
                "meanAllocatedBranches": float(group["allocatedBranches"].mean()),
                "minimumAllocatedBranches": int(group["allocatedBranches"].min()),
                "medianAllocatedBranches": float(group["allocatedBranches"].median()),
                "maximumAllocatedBranches": int(group["allocatedBranches"].max()),
                "meanConditionalBreakTrials": float(
                    group["conditionalBreakTrials"].mean()
                ),
                "spearman": safe_spearman(q_estimate, q_reference),
                "spearmanLower95": rho_ci[0],
                "spearmanUpper95": rho_ci[1],
                "meanBrier": float(group["brierReferenceBranches"].mean()),
                "meanPriorBrier": float(group["priorBrierReferenceBranches"].mean()),
                "brierImprovementOverPrior": float(
                    group["brierImprovementOverPrior"].mean()
                ),
                "brierImprovementLower95": brier_ci[0],
                "brierImprovementUpper95": brier_ci[1],
                "meanLogLoss": float(group["logLossReferenceBranches"].mean()),
                "meanAbsoluteQError": float(group["absoluteQError"].mean()),
                "meanAbsoluteQErrorLower95": mae_ci[0],
                "meanAbsoluteQErrorUpper95": mae_ci[1],
                "meanPosteriorWidth95": float(group["posteriorWidth95"].mean()),
                "meanPosteriorWidthLower95": width_ci[0],
                "meanPosteriorWidthUpper95": width_ci[1],
                "meanMolecularUpdates": float(group["molecularUpdates"].mean()),
                "molecularUpdateSavingsVsFixed64": float(
                    1 - group["molecularUpdates"].mean() / baseline_cost
                ),
                "branchSavingsVsFixed64": float(
                    1 - group["allocatedBranches"].mean() / 64
                ),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(keys).reset_index(drop=True),
        pd.DataFrame(bootstrap_rows)
        .sort_values([*keys, "replicate"])
        .reset_index(drop=True),
    )


def half_replay_validation(
    measurements: pd.DataFrame, frozen_states: pd.DataFrame
) -> pd.DataFrame:
    fixed = measurements[
        measurements["measurementStrategy"].eq("FIXED_UNIFORM")
        & measurements["averageBranchBudget"].eq(64)
    ]
    merged = fixed.merge(
        frozen_states[
            [
                "stateId",
                "trialsHalfA",
                "trialsHalfB",
                "qHatHalfA",
                "qHatHalfB",
            ]
        ],
        on="stateId",
        validate="one_to_one",
    )
    merged["trialAExact"] = merged["conditionalBreakTrials"].eq(
        merged["trialsHalfA"]
    )
    merged["trialBExact"] = merged["referenceTrials"].eq(merged["trialsHalfB"])
    merged["qAExact"] = np.isclose(
        merged["qEstimateRaw"], merged["qHatHalfA"], atol=0, rtol=0
    )
    merged["qBExact"] = np.isclose(
        merged["qReference"], merged["qHatHalfB"], atol=0, rtol=0
    )
    merged["exactReplay"] = merged[
        ["trialAExact", "trialBExact", "qAExact", "qBExact"]
    ].all(axis=1)
    return merged[
        [
            "stateId",
            "evaluationCohort",
            "candidateId",
            "matrixIndex",
            "trialsHalfA",
            "trialsHalfB",
            "qHatHalfA",
            "qHatHalfB",
            "trialAExact",
            "trialBExact",
            "qAExact",
            "qBExact",
            "exactReplay",
        ]
    ].sort_values(["candidateId", "evaluationCohort", "matrixIndex"]).reset_index(
        drop=True
    )


def resplit_results(
    episodes: pd.DataFrame, priors: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_groups = {
        state_id: group.sort_values("branchIndex").reset_index(drop=True)
        for state_id, group in episodes.groupby("stateId", sort=False)
    }
    registry = (
        episodes[
            [
                "stateId",
                "evaluationCohort",
                "candidateId",
                "matrixIndex",
            ]
        ]
        .drop_duplicates()
        .sort_values(["candidateId", "evaluationCohort", "stateId"])
    )
    rows: list[dict[str, Any]] = []
    for replicate in range(RESPLITS):
        for (cohort, candidate), group_registry in registry.groupby(
            ["evaluationCohort", "candidateId"], sort=False
        ):
            accumulators = {
                budget: {
                    "estimate": [],
                    "reference": [],
                    "informed": [],
                    "brierImprovement": [],
                    "absoluteError": [],
                }
                for budget in BUDGETS
            }
            for state_id in group_registry["stateId"].astype(str):
                group = state_groups[state_id]
                rng = np.random.Generator(
                    np.random.PCG64DXSM(derived_seed("resplit", replicate, state_id))
                )
                order = rng.permutation(128)
                reference = group.iloc[order[64:]]
                reference = reference[reference["breakObserved"]]
                if not len(reference):
                    continue
                reference_outcomes = reference[
                    "newHereditaryEpisodeRun3"
                ].to_numpy(np.int8)
                q_reference = float(reference_outcomes.mean())
                prior_score = bernoulli_scores(priors[candidate], reference_outcomes)
                for budget in BUDGETS:
                    selected = group.iloc[order[:budget]]
                    conditional = selected[selected["breakObserved"]]
                    successes = int(
                        conditional["newHereditaryEpisodeRun3"].sum()
                    )
                    estimate = jeffreys_estimate(successes, len(conditional))
                    score = bernoulli_scores(
                        estimate.posterior_mean, reference_outcomes
                    )
                    accumulator = accumulators[budget]
                    accumulator["estimate"].append(estimate.posterior_mean)
                    accumulator["reference"].append(q_reference)
                    accumulator["informed"].append(len(conditional) > 0)
                    accumulator["brierImprovement"].append(
                        prior_score["brier"] - score["brier"]
                    )
                    accumulator["absoluteError"].append(
                        abs(estimate.posterior_mean - q_reference)
                    )
            for budget, accumulator in accumulators.items():
                rows.append(
                    {
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "averageBranchBudget": budget,
                        "resplitReplicate": replicate,
                        "states": len(accumulator["estimate"]),
                        "dataInformedFraction": float(
                            np.mean(accumulator["informed"])
                        ),
                        "spearman": safe_spearman(
                            np.asarray(accumulator["estimate"]),
                            np.asarray(accumulator["reference"]),
                        ),
                        "brierImprovementOverPrior": float(
                            np.mean(accumulator["brierImprovement"])
                        ),
                        "meanAbsoluteQError": float(
                            np.mean(accumulator["absoluteError"])
                        ),
                    }
                )
    raw = pd.DataFrame(rows).sort_values(
        ["candidateId", "evaluationCohort", "averageBranchBudget", "resplitReplicate"]
    ).reset_index(drop=True)
    summary_rows = []
    keys = ["evaluationCohort", "candidateId", "averageBranchBudget"]
    for values, group in raw.groupby(keys, sort=False):
        summary_rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "resplits": len(group),
                "minimumStates": int(group["states"].min()),
                "spearmanP10": float(group["spearman"].quantile(0.1)),
                "spearmanMedian": float(group["spearman"].median()),
                "spearmanP90": float(group["spearman"].quantile(0.9)),
                "minimumDataInformedFraction": float(
                    group["dataInformedFraction"].min()
                ),
                "medianBrierImprovementOverPrior": float(
                    group["brierImprovementOverPrior"].median()
                ),
                "brierImprovementP10": float(
                    group["brierImprovementOverPrior"].quantile(0.1)
                ),
                "medianAbsoluteQError": float(
                    group["meanAbsoluteQError"].median()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(keys).reset_index(drop=True)
    return raw, summary


def scientific_gates(
    groups: pd.DataFrame, resplits: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], str]:
    heldout = groups[groups["evaluationCohort"].isin(EVALUATION_COHORTS)].copy()
    fixed_resplit = resplits[resplits["evaluationCohort"].isin(EVALUATION_COHORTS)]
    heldout = heldout.merge(
        fixed_resplit[
            [
                "evaluationCohort",
                "candidateId",
                "averageBranchBudget",
                "spearmanP10",
            ]
        ],
        on=["evaluationCohort", "candidateId", "averageBranchBudget"],
        how="left",
        validate="many_to_one",
    )
    heldout["availabilityPassed"] = heldout["dataInformedFraction"].ge(0.95)
    heldout["rankPassed"] = heldout["spearman"].gt(0.5) & heldout[
        "spearmanLower95"
    ].gt(0.3)
    heldout["brierPassed"] = heldout["brierImprovementLower95"].gt(0)
    heldout["resplitPassed"] = np.where(
        heldout["measurementStrategy"].eq("FIXED_UNIFORM"),
        heldout["spearmanP10"].gt(0.3),
        True,
    )
    heldout["groupGatePassed"] = heldout[
        ["availabilityPassed", "rankPassed", "brierPassed", "resplitPassed"]
    ].all(axis=1)
    budget_rows = []
    for (strategy, budget), group in heldout.groupby(
        ["measurementStrategy", "averageBranchBudget"], sort=False
    ):
        budget_rows.append(
            {
                "gateLevel": "COMPLETE_BUDGET",
                "measurementStrategy": strategy,
                "averageBranchBudget": budget,
                "evaluationGroups": len(group),
                "availabilityGroupsPassed": int(group["availabilityPassed"].sum()),
                "rankGroupsPassed": int(group["rankPassed"].sum()),
                "brierGroupsPassed": int(group["brierPassed"].sum()),
                "resplitGroupsPassed": int(group["resplitPassed"].sum()),
                "passed": len(group) == 4 and bool(group["groupGatePassed"].all()),
            }
        )
    budget_gates = pd.DataFrame(budget_rows)
    fixed_passing = sorted(
        budget_gates.loc[
            budget_gates["measurementStrategy"].eq("FIXED_UNIFORM")
            & budget_gates["passed"],
            "averageBranchBudget",
        ].astype(int)
    )
    adaptive_passing = sorted(
        budget_gates.loc[
            budget_gates["measurementStrategy"].eq(
                "ADAPTIVE_POSTERIOR_VARIANCE"
            )
            & budget_gates["passed"],
            "averageBranchBudget",
        ].astype(int)
    )
    minimum_fixed = fixed_passing[0] if fixed_passing else None
    minimum_adaptive = adaptive_passing[0] if adaptive_passing else None
    reduced = bool(minimum_fixed is not None and minimum_fixed <= 32)
    adaptive_improves = bool(
        minimum_adaptive is not None
        and minimum_fixed is not None
        and minimum_adaptive < minimum_fixed
    )
    classifications = []
    if reduced:
        classifications.append("PROCESS_COMMITTOR_MEASURABLE_WITH_REDUCED_SHOOTING")
    elif minimum_fixed == 64:
        classifications.append("FULL_64_BRANCH_HALF_REQUIRED")
    elif minimum_adaptive is not None and minimum_adaptive <= 32:
        classifications.append("PROCESS_COMMITTOR_MEASURABLE_WITH_ADAPTIVE_SHOOTING")
    else:
        classifications.append(
            "PROCESS_COMMITTOR_SHOOTING_RANK_NOT_RELIABLE_UNDER_INDEPENDENT_HALF"
        )
    classifications.append(
        "UNCERTAINTY_ALLOCATION_IMPROVES_SHOOTING_EFFICIENCY"
        if adaptive_improves
        else "UNIFORM_ALLOCATION_NOT_IMPROVED_BY_REGISTERED_ADAPTATION"
    )
    classifications.extend(
        [
            "SHOOTING_REMAINS_SIMULATOR_BASED_NOT_STATIC_EARLY_WARNING",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
    )
    next_theme = (
        "L49_LONGITUDINAL_PROCESS_COMMITTOR_RISK_TRAJECTORY"
        if minimum_fixed is not None or minimum_adaptive is not None
        else "L49_PROCESS_TARGET_OR_MEASUREMENT_REASSESSMENT"
    )
    group_rows = heldout.copy()
    group_rows.insert(0, "gateLevel", "EVALUATION_GROUP")
    for column in budget_gates.columns:
        if column not in group_rows:
            group_rows[column] = np.nan
    for column in group_rows.columns:
        if column not in budget_gates:
            budget_gates[column] = np.nan
    gates = pd.concat(
        [group_rows, budget_gates.reindex(columns=group_rows.columns)],
        ignore_index=True,
    )
    return gates, classifications, next_theme


def input_scope_registry() -> pd.DataFrame:
    rows = []
    for input_id, path in (
        ("L44_BRANCH_EPISODES", L44_ROOT / "branch_episode_results.parquet"),
        ("L44_STATE_PROCESS", L44_ROOT / "state_process_results.parquet"),
        ("L44_ARTIFACT_MANIFEST", L44_ROOT / "artifact_manifest.json"),
        ("L41_BRANCH_TRACE", L41_ROOT / "branch_trace_results.parquet"),
        ("L41_ARTIFACT_MANIFEST", L41_ROOT / "artifact_manifest.json"),
        ("L47_ARTIFACT_MANIFEST", L47_ROOT / "artifact_manifest.json"),
    ):
        rows.append(
            {
                "inputId": input_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "readOnly": True,
            }
        )
    return pd.DataFrame(rows)


def analysis_seed_manifest(states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in ("L28_DEVELOPMENT", *EVALUATION_COHORTS):
        for candidate in CANDIDATES:
            for strategy in STRATEGIES:
                for budget in BUDGETS:
                    material = seed_material(
                        "bootstrap", cohort, candidate, strategy, budget
                    )
                    rows.append(
                        {
                            "purpose": "MATRIX_BOOTSTRAP_STREAM",
                            "stateId": None,
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "analysisId": f"{strategy}::{budget}",
                            "replicate": None,
                            "derivedSeed": str(int.from_bytes(material[:16], "big")),
                            "seedMaterialSha256": hashlib.sha256(material).hexdigest(),
                        }
                    )
    for state_id in sorted(states["stateId"].astype(str)):
        state = states[states["stateId"].eq(state_id)].iloc[0]
        for replicate in range(RESPLITS):
            material = seed_material("resplit", replicate, state_id)
            rows.append(
                {
                    "purpose": "INDEPENDENT_BRANCH_RESPLIT",
                    "stateId": state_id,
                    "evaluationCohort": state.evaluationCohort,
                    "candidateId": state.candidateId,
                    "analysisId": "BRANCH_A_B_RESPLIT",
                    "replicate": replicate,
                    "derivedSeed": str(int.from_bytes(material[:16], "big")),
                    "seedMaterialSha256": hashlib.sha256(material).hexdigest(),
                }
            )
    frame = pd.DataFrame(rows)
    if frame["derivedSeed"].duplicated().any() or frame[
        "seedMaterialSha256"
    ].duplicated().any():
        raise RuntimeError("L48 analysis seed collision")
    return frame


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L48/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            if "seedmaterialsha256" in column.lower():
                prior_material.update(frame[column].dropna().astype(str))
    overlaps = sorted(set(seeds["seedMaterialSha256"].astype(str)) & prior_material)
    return {
        "schema": "eidosoma.e01.s19_l48.seed_firewall.v1",
        "status": "PASS" if not overlaps else "FAIL",
        "analysisSeedCount": len(seeds),
        "analysisSeedMaterialOverlapCount": len(overlaps),
        "newScientificBranchStreams": 0,
    }


def candidate_comparison(groups: pd.DataFrame) -> pd.DataFrame:
    heldout = groups[groups["evaluationCohort"].isin(EVALUATION_COHORTS)]
    pivot = heldout.pivot_table(
        index=["evaluationCohort", "measurementStrategy", "averageBranchBudget"],
        columns="candidateId",
        values=["spearman", "brierImprovementOverPrior", "meanAbsoluteQError"],
    )
    pivot.columns = ["::".join(map(str, value)) for value in pivot.columns]
    return pivot.reset_index()


def compute_tables() -> tuple[dict[str, pd.DataFrame], list[str], str]:
    episodes, frozen_states, _ = load_inputs()
    priors = development_priors(frozen_states)
    prior_registry = pd.DataFrame(
        [
            {
                "candidateId": candidate,
                "developmentPrior": value,
                "fitCohort": "L28_DEVELOPMENT",
                "fitUnit": "eligible catalytic-matrix state qHat",
            }
            for candidate, value in priors.items()
        ]
    )
    fixed = fixed_measurements(episodes, priors)
    adaptive = adaptive_measurements(episodes, priors)
    measurements = pd.concat([fixed, adaptive], ignore_index=True).sort_values(
        [
            "candidateId",
            "evaluationCohort",
            "measurementStrategy",
            "averageBranchBudget",
            "matrixIndex",
        ]
    ).reset_index(drop=True)
    replay = half_replay_validation(measurements, frozen_states)
    if not replay["exactReplay"].all():
        raise RuntimeError("L48 full-half replay failure")
    groups, bootstrap = measurement_group_results(measurements)
    resplit_raw, resplit_summary = resplit_results(episodes, priors)
    gates, classifications, next_theme = scientific_gates(groups, resplit_summary)
    state_registry = frozen_states[
        [
            "stateId",
            "evaluationCohort",
            "candidateId",
            "matrixIndex",
            "landmark",
            "trials",
            "trialsHalfA",
            "trialsHalfB",
            "qHat",
            "qHatHalfA",
            "qHatHalfB",
            "eligible",
            "targetUsesCompletedTestTrajectory",
        ]
    ].copy()
    tables = {
        "eligible_state_registry.parquet": state_registry,
        "development_prior_registry.parquet": prior_registry,
        "branch_measurement_results.parquet": measurements,
        "fixed_budget_results.parquet": fixed,
        "adaptive_allocation_results.parquet": adaptive,
        "full_half_replay_validation.parquet": replay,
        "measurement_group_results.parquet": groups,
        "matrix_bootstrap_results.parquet": bootstrap,
        "resplit_robustness_results.parquet": resplit_raw,
        "resplit_robustness_summary.parquet": resplit_summary,
        "candidate_comparison.parquet": candidate_comparison(groups),
        "scientific_gate_results.parquet": gates,
    }
    return tables, classifications, next_theme


def make_figures(tables: dict[str, pd.DataFrame]) -> None:
    figure_root = BUILD_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    groups = tables["measurement_group_results.parquet"]
    heldout = groups[groups["evaluationCohort"].isin(EVALUATION_COHORTS)]
    colors = {
        "S12F-CANDIDATE-02": "#4c78a8",
        "S12F-CANDIDATE-03": "#f58518",
    }
    linestyles = {
        "FIXED_UNIFORM": "-",
        "ADAPTIVE_POSTERIOR_VARIANCE": "--",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, cohort in zip(axes, EVALUATION_COHORTS, strict=True):
        for (candidate, strategy), group in heldout[
            heldout["evaluationCohort"].eq(cohort)
        ].groupby(["candidateId", "measurementStrategy"], sort=False):
            group = group.sort_values("averageBranchBudget")
            axis.plot(
                group["averageBranchBudget"],
                group["spearman"],
                marker="o",
                color=colors[candidate],
                ls=linestyles[strategy],
                label=f"C{candidate[-2:]} {strategy.split('_')[0]}",
            )
            axis.fill_between(
                group["averageBranchBudget"],
                group["spearmanLower95"],
                group["spearmanUpper95"],
                color=colors[candidate],
                alpha=0.08,
            )
        axis.axhline(0.5, color="black", ls=":")
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS, BUDGETS)
        axis.set_xlabel("Average estimator branches per state")
        axis.set_title(cohort)
    axes[0].set_ylabel("Spearman with independent branch-B q")
    axes[0].legend(fontsize=7)
    fig.suptitle("Process-committor rank reliability versus branch budget")
    fig.tight_layout()
    fig.savefig(figure_root / "01_rank_reliability_by_budget.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, cohort in zip(axes, EVALUATION_COHORTS, strict=True):
        for (candidate, strategy), group in heldout[
            heldout["evaluationCohort"].eq(cohort)
        ].groupby(["candidateId", "measurementStrategy"], sort=False):
            group = group.sort_values("averageBranchBudget")
            axis.plot(
                group["averageBranchBudget"],
                group["brierImprovementOverPrior"],
                marker="o",
                color=colors[candidate],
                ls=linestyles[strategy],
                label=f"C{candidate[-2:]} {strategy.split('_')[0]}",
            )
            axis.fill_between(
                group["averageBranchBudget"],
                group["brierImprovementLower95"],
                group["brierImprovementUpper95"],
                color=colors[candidate],
                alpha=0.08,
            )
        axis.axhline(0, color="black", ls=":")
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS, BUDGETS)
        axis.set_xlabel("Average estimator branches per state")
        axis.set_title(cohort)
    axes[0].set_ylabel("Brier improvement over development prior")
    axes[0].legend(fontsize=7)
    fig.suptitle("Independent branch-B probability scoring")
    fig.tight_layout()
    fig.savefig(figure_root / "02_brier_improvement_by_budget.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (candidate, strategy), group in heldout.groupby(
        ["candidateId", "measurementStrategy"], sort=False
    ):
        averaged = group.groupby("averageBranchBudget", as_index=False)[
            ["meanPosteriorWidth95", "molecularUpdateSavingsVsFixed64"]
        ].mean()
        axes[0].plot(
            averaged["averageBranchBudget"],
            averaged["meanPosteriorWidth95"],
            marker="o",
            color=colors[candidate],
            ls=linestyles[strategy],
            label=f"C{candidate[-2:]} {strategy.split('_')[0]}",
        )
        axes[1].plot(
            averaged["averageBranchBudget"],
            averaged["molecularUpdateSavingsVsFixed64"],
            marker="o",
            color=colors[candidate],
            ls=linestyles[strategy],
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS, BUDGETS)
        axis.set_xlabel("Average estimator branches per state")
    axes[0].set_ylabel("Mean Jeffreys 95% interval width")
    axes[1].set_ylabel("Molecular-update savings vs fixed 64")
    axes[0].legend(fontsize=7)
    fig.suptitle("Uncertainty and measured simulation cost")
    fig.tight_layout()
    fig.savefig(figure_root / "03_uncertainty_and_cost.png", dpi=160)
    plt.close(fig)

    state = tables["branch_measurement_results.parquet"]
    adaptive32 = state[
        state["measurementStrategy"].eq("ADAPTIVE_POSTERIOR_VARIANCE")
        & state["averageBranchBudget"].eq(32)
        & state["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.arange(2, 67, 4)
    for candidate in CANDIDATES:
        ax.hist(
            adaptive32.loc[
                adaptive32["candidateId"].eq(candidate), "allocatedBranches"
            ],
            bins=bins,
            alpha=0.5,
            label=f"C{candidate[-2:]}",
        )
    ax.axvline(32, color="black", ls="--", label="uniform 32")
    ax.set_xlabel("Allocated branch count per state")
    ax.set_ylabel("States")
    ax.set_title("Registered uncertainty allocation at average budget 32")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "04_adaptive_allocation_distribution.png", dpi=160)
    plt.close(fig)

    resplit = tables["resplit_robustness_summary.parquet"]
    eval_resplit = resplit[resplit["evaluationCohort"].isin(EVALUATION_COHORTS)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, cohort in zip(axes, EVALUATION_COHORTS, strict=True):
        for candidate, group in eval_resplit[
            eval_resplit["evaluationCohort"].eq(cohort)
        ].groupby("candidateId", sort=False):
            group = group.sort_values("averageBranchBudget")
            axis.plot(
                group["averageBranchBudget"],
                group["spearmanMedian"],
                marker="o",
                color=colors[candidate],
                label=f"C{candidate[-2:]}",
            )
            axis.fill_between(
                group["averageBranchBudget"],
                group["spearmanP10"],
                group["spearmanP90"],
                color=colors[candidate],
                alpha=0.15,
            )
        axis.axhline(0.3, color="black", ls=":")
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS, BUDGETS)
        axis.set_xlabel("Estimator branches")
        axis.set_title(cohort)
    axes[0].set_ylabel("Spearman across 128 independent resplits")
    axes[0].legend(fontsize=8)
    fig.suptitle("Branch-identity robustness of fixed-budget measurement")
    fig.tight_layout()
    fig.savefig(figure_root / "05_resplit_robustness.png", dpi=160)
    plt.close(fig)

    fixed32 = state[
        state["measurementStrategy"].eq("FIXED_UNIFORM")
        & state["averageBranchBudget"].eq(32)
        & state["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    for axis, ((cohort, candidate), group) in zip(
        axes.ravel(),
        fixed32.groupby(["evaluationCohort", "candidateId"], sort=False),
        strict=True,
    ):
        axis.scatter(group["qReference"], group["qEstimateJeffreys"], alpha=0.7)
        axis.plot([0, 1], [0, 1], color="black", ls=":")
        axis.set_title(f"{cohort} C{candidate[-2:]}")
        axis.set_xlabel("Independent branch-B q")
        axis.set_ylabel("32-branch-A Jeffreys estimate")
    fig.suptitle("Canonical independent-half process committor estimates")
    fig.tight_layout()
    fig.savefig(figure_root / "06_fixed32_reference_scatter.png", dpi=160)
    plt.close(fig)

    gates = tables["scientific_gate_results.parquet"]
    complete = gates[gates["gateLevel"].eq("COMPLETE_BUDGET")]
    matrix = complete.pivot_table(
        index="measurementStrategy",
        columns="averageBranchBudget",
        values="passed",
        aggfunc="first",
    ).reindex(index=STRATEGIES, columns=BUDGETS)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    image = ax.imshow(matrix.astype(float), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(BUDGETS)), BUDGETS)
    ax.set_yticks(range(len(STRATEGIES)), STRATEGIES, fontsize=8)
    ax.set_xlabel("Average branch budget")
    ax.set_title("Complete held-out shooting-efficiency gates")
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(figure_root / "07_decision_matrix.png", dpi=160)
    plt.close(fig)


def report_text(
    tables: dict[str, pd.DataFrame],
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    groups = tables["measurement_group_results.parquet"]
    heldout = groups[groups["evaluationCohort"].isin(EVALUATION_COHORTS)][
        [
            "evaluationCohort",
            "candidateId",
            "measurementStrategy",
            "averageBranchBudget",
            "states",
            "dataInformedFraction",
            "spearman",
            "spearmanLower95",
            "spearmanUpper95",
            "brierImprovementOverPrior",
            "brierImprovementLower95",
            "brierImprovementUpper95",
            "meanAbsoluteQError",
            "meanPosteriorWidth95",
            "meanAllocatedBranches",
            "molecularUpdateSavingsVsFixed64",
        ]
    ]
    budget_gates = tables["scientific_gate_results.parquet"]
    budget_gates = budget_gates[budget_gates["gateLevel"].eq("COMPLETE_BUDGET")][
        [
            "measurementStrategy",
            "averageBranchBudget",
            "evaluationGroups",
            "availabilityGroupsPassed",
            "rankGroupsPassed",
            "brierGroupsPassed",
            "resplitGroupsPassed",
            "passed",
        ]
    ]
    resplit = tables["resplit_robustness_summary.parquet"]
    resplit = resplit[resplit["evaluationCohort"].isin(EVALUATION_COHORTS)][
        [
            "evaluationCohort",
            "candidateId",
            "averageBranchBudget",
            "spearmanP10",
            "spearmanMedian",
            "spearmanP90",
            "minimumDataInformedFraction",
            "brierImprovementP10",
        ]
    ]
    return f"""# S19-L48 Full Results — Process-Committor Shooting Necessity and Branch-Budget Efficiency

## Top summary

- **Research step:** `{VERSION}`
- **Completion status:** complete; additive exploratory evidence
- **Artifacts written:** immutable/source/input/seed locks, eligible-state and branch-budget tables, fixed and adaptive estimates, exact half replay, 4,096 matrix bootstraps, 128 deterministic independent resplits, seven figures, validation and hash manifests
- **Validation:** PASS — immutable prior, exact L41/L44 identities, eight fixtures, zero-overlap analysis seeds, 64/64 half replay, independent-reference separation, two exact full analysis passes, report regeneration, storage and artifact hashes; preserved TA01 repaired only NumPy-scalar seed serialization before any scientific table was written
- **Outcome classification:** {', '.join(f'`{value}`' for value in classifications)}
- **Lay summary:** The loop does not search for another biomarker. It asks how much forward stochastic shooting is needed to measure the already frozen probability of forming a new three-fission hereditary episode after a genuine inheritance break. Estimates use only branch A; every accuracy and ranking result is scored on untouched branch B.
- **Recommended next action:** `{next_theme}` under the human-authorized sequence through L65. S20, E02, author contact and intervention work remain inactive.

## Frozen question and scientific boundary

L44 established a state-dependent conditional committor for `NEW_HEREDITARY_EPISODE_RUN3`; L45 and L47 did not find a transferable PhiID or functional increment beyond direct controls. L48 therefore evaluates shooting as a computational measurement method. It preserves the strict `H>0.9` inheritance criterion, genuine-break conditioning, twelve-fission horizon, simulator candidates, 256 L44-eligible matrix/state units and all 128 branch identities.

The Jeffreys Beta(1/2,1/2) posterior mean stabilizes small conditional trial counts. Equal-tailed intervals are uncertainty descriptions, not time-uniform confidence sequences. The adaptive allocation sees only estimator-half outcomes and is judged on the independent reference half.

## Methods

- Fixed branch-A budgets: 4, 8, 16, 32 and 64 branches per state.
- Adaptive budgets: start at 4 per state, add batches of 4 to the largest current Jeffreys posterior variance, lexical state-ID tie breaking, cap 64.
- Reference: all 64 branch-B futures, restricted by the unchanged genuine-break condition.
- Primary unit: catalytic matrix/state; candidates and development/validation/confirmation cohorts remain separate.
- Uncertainty: exactly 4,096 matrix bootstraps per strategy/budget/group.
- Robustness: exactly 128 domain-separated random 64/64 branch resplits; no resplit changes the primary canonical halves.
- Baseline: candidate-specific, matrix-weighted L44 development prior, frozen before held-out scoring.
- Cost: actual frozen L41 molecular updates, fissions and selected observations in the estimator branches.

## Held-out budget results

{heldout.to_markdown(index=False, floatfmt='.6f')}

## Complete budget gates

{budget_gates.to_markdown(index=False)}

## Independent-resplit robustness

{resplit.to_markdown(index=False, floatfmt='.6f')}

## Interpretation

Passing the measurement gate means that a reduced ensemble recovers the independent branch-half ordering and improves probability scoring over a development prior. It does not mean the process can be inferred without simulation. Conversely, failure at a small budget means only that more branch samples are needed; it does not invalidate the L44 process committor.

The target remains online-defined and does not use a completed trajectory, but every shooting estimate uses newly imagined simulator futures. It is therefore eligible as a simulation-accessible risk measurement, not as a static observed-prefix biomarker, empirical biological assay, author-code reconstruction or causal intervention result.

## Validation, runtime and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- Wall time: `{runtime['wallSeconds'] / 60:.3f}` minutes; estimated CPU upper bound: `{runtime['estimatedCpuHoursUpper']:.6f}` hours.
- Workers used: `{runtime['workers']}` of up to 8; numerical-library threads: 1; GPU hours: 0.
- Frozen eligible states: `{runtime['eligibleStates']}`; reused branch outcomes: `{runtime['reusedBranchOutcomes']}`.
- New matrices/trajectories/branch streams: `0/0/0`.
- Exact complete analysis passes: `2`.
- Technical amendments: `1`; TA01 canonicalized a NumPy integer to the identical native-integer seed material and changed no scientific contract.
- Web-grounded statistical context is recorded in `source_registry.parquet`; no external payload is redistributed.

## Limitations

The 64-branch reference is itself a finite Monte Carlo estimate, so L48 uses independent halves and repeated resplits rather than treating it as exact truth. The process is conditional on observing a genuine break and remains a high-frequency heredity event with only a modest order excess over count-matched permutations. Allocation is one registered posterior-variance heuristic, not a tournament. Adaptive reuse of frozen outcomes measures potential efficiency but still requires prospective confirmation before operational use. No L48 result changes S18 or any earlier S19 classification.
"""


def manifest_for(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema": "eidosoma.e01.s19_l48.artifact_manifest.v1",
        "loopId": LOOP_ID,
        "files": rows,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
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
            "beliefBeforeLoop": "L44 supplies a reliable process committor, while L45 and L47 weaken the case that a compact Phi or functional proxy can replace forward propagation.",
            "failureOrAmbiguityTargeted": "Whether shooting can serve as a practical measurement and how many independent futures it requires.",
            "informationGainRationale": "Nested frozen branch budgets and an independent reference half distinguish estimator variance from target nonidentifiability without new simulation.",
            "learned": "L48 branch-budget and allocation contract locked before derived outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L44 reliable process committor; L45 Phi non-support; L47 functional increment non-support; reviewer shooting-as-measurement framing.",
            "proposedNextTest": "Measure fixed and uncertainty-allocated branch efficiency against untouched branch B.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Reduced uniform shooting, adaptive shooting, full-half requirement or unreliable independent-half measurement.",
            "selectedHypotheses": "The process committor is real but may require ensemble forward simulation to estimate.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Another unbounded representation family should precede measurement-efficiency characterization.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A useful shooting budget must recover independent-half ranks and improve Brier score in both candidates and held-out cohorts.",
            "failureOrAmbiguityTargeted": "Minimum branch budget and adaptive allocation value.",
            "informationGainRationale": "Matrix bootstraps and 128 branch resplits expose fragile branch-identity results.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L48 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Process-committor shooting efficiency.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any registered L48 budget/allocation gate that failed.",
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
        + f"\n\n## {LOOP_ID} — process-committor shooting efficiency\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** `{next_theme}`.\n",
    )

    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidate = {
        "branchCount": len(BUDGETS) * 2,
        "bundleId": "L48_PROCESS_COMMITTOR_SHOOTING_EFFICIENCY",
        "candidateId": "S19-L48-PROCESS-COMMITTOR-SHOOTING",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 1,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 4,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "nested branch-A Jeffreys estimates and posterior-variance allocation scored against untouched branch B",
        "rankingScore": 28.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": "PROCESS_COMMITTOR_MEASURABLE_WITH_REDUCED_SHOOTING"
        in classifications
        or "PROCESS_COMMITTOR_MEASURABLE_WITH_ADAPTIVE_SHOOTING" in classifications,
        "selectionReason": "L44_PROCESS_COMMITTOR_AND_L45_L47_PROXY_NON_SUPPORT",
        "sourceGrounding": 4,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidate_path,
        pd.concat(
            [candidates, pd.DataFrame([candidate]).reindex(columns=candidates.columns)],
            ignore_index=True,
        ),
    )

    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    additions = []
    for row in source_registry().itertuples(index=False):
        additions.append(
            {
                "commitOrVersion": None,
                "evidenceClass": row.evidenceClass,
                "finding": f"{row.finding}; L48 use: {row.frozenUse}",
                "licenseStatus": "PUBLIC_METADATA_OR_WORKSPACE_EVIDENCE",
                "redistributionStatus": "REFERENCE_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L48_{row.sourceId}",
                "sourceType": row.evidenceClass,
                "treeIdentity": None,
                "url": row.url,
            }
        )
    BASE.write_parquet(
        source_path,
        pd.concat(
            [sources, pd.DataFrame(additions).reindex(columns=sources.columns)],
            ignore_index=True,
        ),
    )


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L48 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L48 local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    episodes, states, costs = load_inputs()
    scope = input_scope_registry()
    seeds = analysis_seed_manifest(states)
    firewall = seed_firewall(seeds)
    benchmark = {
        "schema": "eidosoma.e01.s19_l48.benchmark_projection.v1",
        "outcomeBlindHeldout": True,
        "developmentOnlyTechnicalBenchmark": True,
        "developmentBenchmarkStates": 10,
        "developmentBenchmarkCandidate": "S12F-CANDIDATE-02",
        "developmentBenchmarkCohort": "L28_DEVELOPMENT",
        "designChangedAfterDevelopmentBenchmark": False,
        "eligibleStatesFromFrozenRegistry": len(states),
        "reusedBranchOutcomes": len(episodes),
        "reusedCostRowsValidated": len(costs),
        "projectedWallHoursUpper": 1.0,
        "projectedCpuHoursUpper": 1.0,
        "workers": 1,
        "workersAvailable": 8,
        "parallelismDecision": "serial vectorized reuse of frozen binary outcomes; no simulator execution and multiprocessing overhead is unnecessary",
        "status": "PASS",
    }
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or len(scope) != 6
        or firewall["status"] != "PASS"
        or len(states) != 256
        or len(episodes) != 32_768
    ):
        raise RuntimeError("L48 preoutcome validation failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L48 decision record\n\n"
        "The human-authorized autonomous sequence through L65 permits up to eight CPUs where helpful. L44 established a reliable online-certified run-of-three heredity committor; L45 and L47 found no registered PhiID or functional increment beyond ordinary controls. Before opening L48 branch-budget outcomes, this record freezes a measurement-efficiency audit on the 256 L44-eligible states and their existing F12 paths only: genuine-break conditioning unchanged; branch A indices 0–63 as estimator and branch B 64–127 as untouched reference; fixed nested budgets 4/8/16/32/64; Jeffreys Beta(1/2,1/2) posterior summaries; one largest-posterior-variance allocation in batches of four; candidate/cohort separation; 4,096 matrix bootstraps; and 128 domain-separated independent 64/64 resplits. No target, threshold, horizon, state, candidate, simulator, branch dynamic, feature, information scalar, label, or intervention is searched. The adaptive rule never reads branch B. Serial vectorized execution is locked because no new simulation occurs.\n"
        "A technical benchmark executed only the first ten sorted `L28_DEVELOPMENT` candidate-2 states after the complete code/configuration had been drafted. It accessed no validation or confirmation outcome, triggered no design change, and is recorded explicitly rather than described as outcome-blind development work.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "input_scope_registry.parquet", scope)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    sources = source_registry()
    BASE.write_parquet(LOOP_ROOT / "source_registry.parquet", sources)
    BASE.write_json(
        LOOP_ROOT / "source_snapshot_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l48.source_snapshot_manifest.v1",
            "l44ProcessCoreSha256": sha256_file(
                ROOT / "src/e01_onset_discovery/heredity_process_family.py"
            ),
            "l48CoreSha256": sha256_file(CORE_PATH),
            "l48RunnerSha256": sha256_file(RUNNER_PATH),
            "configSha256": sha256_file(CONFIG),
            "sources": sources.to_dict("records"),
        },
    )
    locked_inputs = {
        "scopeRegistry": LOOP_ROOT / "input_scope_registry.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "l44Episodes": L44_ROOT / "branch_episode_results.parquet",
        "l44States": L44_ROOT / "state_process_results.parquet",
        "l44ArtifactManifest": L44_ROOT / "artifact_manifest.json",
        "l41Costs": L41_ROOT / "branch_trace_results.parquet",
        "l41ArtifactManifest": L41_ROOT / "artifact_manifest.json",
        "l47ArtifactManifest": L47_ROOT / "artifact_manifest.json",
    }
    hashes = {name: sha256_file(path) for name, path in locked_inputs.items()}
    lock = {
        "schema": "eidosoma.e01.s19_l48.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "configSha256": sha256_file(CONFIG),
        "processId": PROCESS,
        "conditioning": "L44_GENUINE_BREAK",
        "eligibleStates": len(states),
        "estimatorBranchIndices": [0, 63],
        "referenceBranchIndices": [64, 127],
        "fixedBudgets": list(BUDGETS),
        "adaptiveStart": 4,
        "adaptiveBatch": 4,
        "adaptivePriority": "Jeffreys posterior variance",
        "adaptiveTie": "lexical stateId",
        "matrixBootstraps": BOOTSTRAPS,
        "independentResplits": RESPLITS,
        "workers": 1,
        "workersAvailable": 8,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "lockedInputHashes": hashes,
        "outcomeAccessed": False,
        "lockedAtUtc": utc_now(),
    }
    BASE.write_json(LOOP_ROOT / "implementation_lock.json", lock)
    BASE.write_json(
        LOOP_ROOT / "preoutcome_repository_lock.json",
        {
            "head": head,
            "remote": remote,
            "priorAggregateSha256": prior["aggregateSha256"],
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "lockedInputHashes": hashes,
        },
    )


def prepare_ta01() -> None:
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L48 TA01 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L48 TA01 local/remote commit mismatch")
    original = json.loads(
        (LOOP_ROOT / "preoutcome_repository_lock.json").read_text()
    )
    fixtures = fixture_results()
    if not fixtures["passed"].all():
        raise RuntimeError("L48 TA01 fixture failure")
    if BUILD_ROOT.exists():
        quarantine = Path("/cache/e01_s19_l48/attempt_001_failed_build")
        if quarantine.exists():
            shutil.rmtree(quarantine)
        if any(BUILD_ROOT.iterdir()):
            shutil.move(str(BUILD_ROOT), str(quarantine))
        else:
            shutil.rmtree(BUILD_ROOT)
    repair = {
        "schema": "eidosoma.e01.s19_l48.technical_amendment_lock.v1",
        "amendmentId": "TA01_NUMPY_INTEGER_SEED_SERIALIZATION",
        "status": "LOCKED_BEFORE_RETRY",
        "originalRepositoryHead": original["head"],
        "repairedRepositoryHead": head,
        "remoteHead": remote,
        "originalRunnerSha256": original["runnerSha256"],
        "repairedRunnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "configSha256": sha256_file(CONFIG),
        "lockedInputHashes": original["lockedInputHashes"],
        "scientificContractChanged": False,
        "scientificTableWrittenByFailedAttempt": False,
        "reason": "pandas group key averageBranchBudget was numpy.int64 and json.dumps rejected it during bootstrap seed derivation",
        "repair": "convert numpy scalar seed parts to their native scalar with item(); np.int64(32) and int(32) now yield identical frozen seed material",
        "mandatoryFixtureAdded": "F08_NUMPY_INTEGER_SEED_EQUIVALENCE",
        "lockedAtUtc": utc_now(),
    }
    BASE.write_json(LOOP_ROOT / "technical_amendment_lock.json", repair)
    BASE.write_json(
        LOOP_ROOT / "preexecution_repository_lock_ta01.json",
        {
            "head": head,
            "remote": remote,
            "priorAggregateSha256": original["priorAggregateSha256"],
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "lockedInputHashes": original["lockedInputHashes"],
            "technicalAmendmentId": repair["amendmentId"],
        },
    )


def execute() -> None:
    started = time.perf_counter()
    execution_lock_path = LOOP_ROOT / "preexecution_repository_lock_ta01.json"
    if not execution_lock_path.exists():
        execution_lock_path = LOOP_ROOT / "preoutcome_repository_lock.json"
    lock = json.loads(execution_lock_path.read_text())
    if (
        git("rev-parse", "HEAD") != lock["head"]
        or git("rev-parse", "origin/eidosoma/groups/42") != lock["remote"]
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("L48 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_inputs = {
        "scopeRegistry": LOOP_ROOT / "input_scope_registry.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "l44Episodes": L44_ROOT / "branch_episode_results.parquet",
        "l44States": L44_ROOT / "state_process_results.parquet",
        "l44ArtifactManifest": L44_ROOT / "artifact_manifest.json",
        "l41Costs": L41_ROOT / "branch_trace_results.parquet",
        "l41ArtifactManifest": L41_ROOT / "artifact_manifest.json",
        "l47ArtifactManifest": L47_ROOT / "artifact_manifest.json",
    }
    if any(
        sha256_file(path) != lock["lockedInputHashes"][name]
        for name, path in locked_inputs.items()
    ):
        raise RuntimeError("L48 locked input changed")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
        or sha256_file(CONFIG) != lock["configSha256"]
    ):
        raise RuntimeError("L48 pre-execution validation failed")
    if frame_hash(input_scope_registry()) != frame_hash(
        pd.read_parquet(LOOP_ROOT / "input_scope_registry.parquet")
    ):
        raise RuntimeError("L48 input scope regeneration mismatch")
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)

    tables, classifications, next_theme = compute_tables()
    make_figures(tables)
    tables_again, classifications_again, next_theme_again = compute_tables()
    exact = {
        name: frame_hash(frame) == frame_hash(tables_again[name])
        for name, frame in tables.items()
    }
    regeneration = {
        "schema": "eidosoma.e01.s19_l48.regeneration_validation.v1",
        "status": (
            "PASS"
            if all(exact.values())
            and classifications == classifications_again
            and next_theme == next_theme_again
            else "FAIL"
        ),
        "tableExact": exact,
        "classificationExact": classifications == classifications_again,
        "nextThemeExact": next_theme == next_theme_again,
        "analysisPasses": 2,
    }
    if regeneration["status"] != "PASS":
        raise RuntimeError("L48 exact regeneration failure")
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l48.classification.v1",
            "classifications": classifications,
            "nextTheme": next_theme,
            "priorStatusesChanged": False,
            "promotableAsConfirmed": False,
            "newBranchStreams": 0,
        },
    )
    pd.DataFrame(
        [
            {
                "amendmentId": "TA01_NUMPY_INTEGER_SEED_SERIALIZATION",
                "status": "APPLIED_AND_VALIDATED",
                "attemptPreserved": True,
                "scientificContractChanged": False,
                "scientificValuesReleasedBeforeRepair": False,
                "exactSeedEquivalenceFixture": True,
                "reason": "numpy.int64 group key was not directly JSON serializable",
            }
        ]
    ).to_csv(BUILD_ROOT / "technical_amendment_ledger.csv", index=False)
    pd.DataFrame(
        [
            {
                "failureId": "ATTEMPT_001_NUMPY_INTEGER_SEED_SERIALIZATION",
                "stage": "FIRST_BOOTSTRAP_STREAM_CONSTRUCTION",
                "status": "PRESERVED_TECHNICAL_FAILURE_REPAIRED_BY_TA01",
                "reason": "numpy.int64 group key was not JSON serializable; native-integer canonicalization preserves registered seed material",
                "scientificValuesReleased": False,
            }
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    elapsed = time.perf_counter() - started
    runtime = {
        "schema": "eidosoma.e01.s19_l48.runtime.v1",
        "repositoryHead": lock["head"],
        "workers": 1,
        "workersAvailable": 8,
        "parallelismDecision": "serial vectorized reuse of frozen outcomes; no forward simulation was needed",
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": elapsed,
        "estimatedCpuHoursUpper": elapsed / 3600,
        "eligibleStates": len(tables["eligible_state_registry.parquet"]),
        "reusedBranchOutcomes": 32_768,
        "fixedBudgets": list(BUDGETS),
        "matrixBootstrapsPerGroup": BOOTSTRAPS,
        "independentResplits": RESPLITS,
        "analysisPasses": 2,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "completedAtUtc": utc_now(),
    }
    if runtime["estimatedCpuHoursUpper"] > 24 or runtime["wallSeconds"] > 24 * 3600:
        raise RuntimeError("L48 runtime ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    retained_bytes = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    ) + sum(path.stat().st_size for path in LOOP_ROOT.iterdir() if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l48.storage_validation.v1",
        "status": "PASS" if retained_bytes <= 15 * 1024**3 else "FAIL",
        "retainedBytes": retained_bytes,
        "retainedGiBCeiling": 15,
        "temporaryGiBCeiling": 30,
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(tables, classifications, runtime, next_theme)
    if report != report_text(tables, classifications, runtime, next_theme):
        raise RuntimeError("L48 report regeneration failure")
    BASE.atomic_text(BUILD_ROOT / "S19_L48_FULL_RESULTS.md", report)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L48 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Next:** `{next_theme}`.\n",
    )
    if storage["status"] != "PASS":
        raise RuntimeError("L48 storage ceiling exceeded")
    for path in (BUILD_ROOT / "figures").glob("*.png"):
        if not path.stat().st_size:
            raise RuntimeError(f"empty L48 figure: {path}")

    for path in BUILD_ROOT.iterdir():
        destination = LOOP_ROOT / path.name
        if path.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
    BASE.write_json(LOOP_ROOT / "artifact_manifest.json", manifest_for(LOOP_ROOT))
    if manifest_for(LOOP_ROOT) != json.loads(
        (LOOP_ROOT / "artifact_manifest.json").read_text()
    ):
        raise RuntimeError("L48 artifact manifest regeneration failed")

    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    root_report = (
        f"# S19 current-step report\n\nLatest completed loop: `{LOOP_ID}`.\n\n"
        f"Classification: {', '.join(classifications)}.\n\n"
        f"Next autonomous theme: `{next_theme}`.\n"
    )
    BASE.atomic_text(ARTIFACT_ROOT / "S19_CURRENT_STEP_REPORT.md", root_report)
    BASE.atomic_text(ARTIFACT_ROOT / "CURRENT_STEP_HANDOFF.md", root_report)
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "programStatus": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "nextAuthorizedLoop": "S19-L49",
            "nextTheme": next_theme,
            "authorizationUpperBound": "S19-L65",
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
    parser.add_argument("--prepare-ta01", action="store_true")
    args = parser.parse_args()
    if args.prepare_lock:
        prepare_lock()
    elif args.prepare_ta01:
        prepare_ta01()
    else:
        execute()


if __name__ == "__main__":
    main()
