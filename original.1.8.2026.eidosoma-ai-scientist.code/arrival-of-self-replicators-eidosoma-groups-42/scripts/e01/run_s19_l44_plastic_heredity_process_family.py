#!/usr/bin/env python3
"""Run S19-L44 plastic-heredity process-family identifiability audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from e01_onset_discovery.heredity_process_family import (
    crossfit_markov_gain_bits,
    summarize_binary_episode,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L44"
L41_ROOT = ARTIFACT_ROOT / "loops/L41"
L42_ROOT = ARTIFACT_ROOT / "loops/L42"
L43_ROOT = ARTIFACT_ROOT / "loops/L43"
BUILD_ROOT = Path("/cache/e01_s19_l44")
CONFIG = ROOT / "configs/e01/s19_l44_plastic_heredity_process_family.yaml"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = ROOT / "src/e01_onset_discovery/heredity_process_family.py"
LOOP_ID = "S19-L44"
VERSION = "E01-S19-L44-PLASTIC-HEREDITY-PROCESS-FAMILY-IDENTIFIABILITY-v1.0.0"
BOOTSTRAPS = 4096
EVALUATION_COHORTS = ("L28_VALIDATION", "L31_CONFIRMATION")
PROCESSES = (
    "INHERITANCE_BREAK_WITH_DEPARTURE",
    "INHERITANCE_RESUMPTION_RUN2",
    "NEW_HEREDITARY_EPISODE_RUN3",
    "PERSISTENT_HEREDITARY_EPISODE_RUN5",
    "OLD_NEIGHBOURHOOD_RECOVERY_RUN2",
    "POSITIVE_OLD_ANCHOR_GAIN_AT_RESUMPTION",
    "REPEATED_CROSS_GENERATION_RETURN",
)
ORDER_PROCESSES = {
    "INHERITANCE_RESUMPTION_RUN2": "run2OrderNullProbability",
    "NEW_HEREDITARY_EPISODE_RUN3": "run3OrderNullProbability",
    "PERSISTENT_HEREDITARY_EPISODE_RUN5": "run5OrderNullProbability",
}
SEED_ROOT = bytes.fromhex(
    "bcce5b5a5312551086b3aeed8f64f52ba84ecdd6f39c9bbe2e49941c9de6fb0a"
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L43 = load_module(
    "e01_l43_runner",
    ROOT / "scripts/e01/run_s19_l43_continuous_homeostatic_recovery_gain.py",
)
BASE = L43.BASE


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.sort_index(axis=1)
        .sort_values(list(frame.columns), na_position="last")
        .reset_index(drop=True)
        .to_json(orient="records", double_precision=15)
        .encode()
    ).hexdigest()


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if (
        mask.sum() < 3
        or len(np.unique(left[mask])) < 2
        or len(np.unique(right[mask])) < 2
    ):
        return float("nan")
    return float(spearmanr(left[mask], right[mask]).statistic)


def interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(finite, [0.025, 0.975])))


def seed_material(*parts: object) -> bytes:
    return hashlib.sha256(
        SEED_ROOT + b"\x00" + json.dumps(parts, separators=(",", ":")).encode()
    ).digest()


def derived_seed(*parts: object) -> int:
    return int.from_bytes(seed_material(*parts)[:16], "big")


def validate_immutable_prior() -> dict[str, Any]:
    prior = L43.validate_immutable_prior()
    rows = []
    for root in (L42_ROOT, L43_ROOT):
        manifest = json.loads((root / "artifact_manifest.json").read_text())
        for row in manifest["files"]:
            path = root / row["path"]
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
        "schema": "eidosoma.e01.s19_l44.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "priorThroughL42Unchanged": bool(prior["unchanged"]),
        "validatedArtifactCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
    }


def fixture_results() -> pd.DataFrame:
    episode = summarize_binary_episode([False, True, True, True, True, True])
    sticky = crossfit_markov_gain_bits(
        [[False, False, False, True, True, True]] * 20,
        [[False, False, False, True, True, True]] * 20,
    )
    alternating = crossfit_markov_gain_bits(
        [[False, True, False, True]] * 20,
        [[False, True, False, True]] * 20,
    )
    rows = [
        ("RUN2_CERTIFICATION", episode.run2_certification_one_based == 3),
        ("RUN3_CERTIFICATION", episode.run3_certification_one_based == 4),
        ("RUN5_CERTIFICATION", episode.run5_certification_one_based == 6),
        ("FIXED_COUNT_NULL_BOUNDED", 0 <= episode.run3_order_null_probability <= 1),
        ("STICKY_MARKOV_GAIN", sticky["markovGainBitsPerTransition"] > 0),
        ("ORDER_SENSITIVE_MARKOV_GAIN", alternating["markovGainBitsPerTransition"] > 0),
        ("EMPTY_SEQUENCE_STATUS", summarize_binary_episode([]).opportunities == 0),
        (
            "EXACT_REPLAY",
            episode == summarize_binary_episode([False, True, True, True, True, True]),
        ),
    ]
    return pd.DataFrame(
        [{"fixtureId": name, "passed": bool(passed)} for name, passed in rows]
    )


def analysis_seed_manifest() -> pd.DataFrame:
    rows = []
    for cohort in ("L28_DEVELOPMENT", *EVALUATION_COHORTS):
        for candidate in ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03"):
            for process in PROCESSES:
                parts = ("process_bootstrap", cohort, candidate, process)
                material = seed_material(*parts)
                rows.append(
                    {
                        "purpose": parts[0],
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "processId": process,
                        "partsJson": json.dumps(parts),
                        "rootHex": SEED_ROOT.hex(),
                        "derivedSeed": str(int.from_bytes(material[:16], "big")),
                        "seedMaterialSha256": material.hex(),
                    }
                )
            for comparison in ("MARKOV_GAIN", *ORDER_PROCESSES):
                parts = ("baseline_bootstrap", cohort, candidate, comparison)
                material = seed_material(*parts)
                rows.append(
                    {
                        "purpose": parts[0],
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "processId": comparison,
                        "partsJson": json.dumps(parts),
                        "rootHex": SEED_ROOT.hex(),
                        "derivedSeed": str(int.from_bytes(material[:16], "big")),
                        "seedMaterialSha256": material.hex(),
                    }
                )
    frame = (
        pd.DataFrame(rows)
        .sort_values(list(pd.DataFrame(rows).columns))
        .reset_index(drop=True)
    )
    if (
        frame["derivedSeed"].duplicated().any()
        or frame["seedMaterialSha256"].duplicated().any()
    ):
        raise RuntimeError("L44 analysis seed collision")
    return frame


def seed_firewall(analysis: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L44/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            lowered = column.lower()
            if "seedmaterialsha256" in lowered:
                prior_material.update(frame[column].dropna().astype(str))
            if lowered == "derivedseed" or lowered.endswith("derivedseed"):
                prior_derived.update(frame[column].dropna().astype(str))
    overlap_m = sorted(set(analysis["seedMaterialSha256"]) & prior_material)
    overlap_d = sorted(set(analysis["derivedSeed"]) & prior_derived)
    return {
        "schema": "eidosoma.e01.s19_l44.seed_firewall.v1",
        "status": "PASS" if not overlap_m and not overlap_d else "FAIL",
        "analysisSeedCount": len(analysis),
        "newBranchStreams": 0,
        "seedMaterialOverlapCount": len(overlap_m),
        "derivedSeedOverlapCount": len(overlap_d),
    }


def source_grounding_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L44_REVIEWER_PLASTIC_HEREDITY",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Map inheritance frequency, breaks, resumption, old return, new hereditary regimes and persistence separately; use renewal and Markov baselines.",
                "frozenUse": "seven-process family and IID/Markov/order-null audit",
                "url": None,
            },
            {
                "sourceId": "L44_L43_DIRECTIONAL_DIVERGENCE",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "At inheritance resumption, old-anchor H changed by about -0.26 while arbitrary-anchor changes were about -0.02 to -0.03.",
                "frozenUse": "plastic-heredity rather than restorative-homeostasis framing",
                "url": None,
            },
            {
                "sourceId": "L44_ALTERNATING_RENEWAL",
                "evidenceClass": "PUBLIC_METHOD_SOURCE",
                "finding": "Alternating renewal processes represent binary regimes through random dwell-time distributions.",
                "frozenUse": "positive/negative inheritance dwell and renewal summaries",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4286718/",
            },
            {
                "sourceId": "L44_RECURRENCE_TIME_REGIME_SWITCHING",
                "evidenceClass": "PUBLIC_METHOD_SOURCE",
                "finding": "Recurrence-time distributions can distinguish one regime from switching mixtures.",
                "frozenUse": "time-since-break and episode recurrence diagnostics",
                "url": "https://doi.org/10.1016/j.spl.2009.08.025",
            },
            {
                "sourceId": "L44_GARD_HOMEOSTATIC_INHERITANCE",
                "evidenceClass": "PRIMARY_LITERATURE",
                "finding": "GARD homeostatic growth and fission transfer compositional information, while privileged compositional states may vary.",
                "frozenUse": "inheritance process is primary; exact old-state restoration is separate",
                "url": "https://pubmed.ncbi.nlm.nih.gov/30045888/",
            },
        ]
    )


def load_branch_family() -> tuple[pd.DataFrame, pd.DataFrame]:
    l42 = pd.read_parquet(L42_ROOT / "branch_recovery_results.parquet")
    l42 = l42[
        (l42["branchFamily"] == "F12")
        & (l42["targetId"] == "PRIMARY_PREBREAK_DAUGHTER")
    ].copy()
    l43 = pd.read_parquet(L43_ROOT / "branch_gain_results.parquet")
    l43 = l43[
        (l43["branchFamily"] == "F12")
        & (l43["targetId"] == "PRIMARY_PREBREAK_DAUGHTER")
    ][
        [
            "stateId",
            "branchIndex",
            "resumptionObserved",
            "recoveryGain",
            "breakAnchorH",
            "certificationAnchorH",
            "inheritedFlags",
            "pathSha256",
        ]
    ].rename(
        columns={
            "inheritedFlags": "inheritanceFlagsL43",
            "pathSha256": "pathSha256L43",
        }
    )
    l41 = pd.read_parquet(L41_ROOT / "branch_outcome_results.parquet")
    l41 = l41[
        (l41["branchFamily"] == "F12") & (l41["targetId"] == "PRIMARY_PREFIX_HISTORY")
    ][["stateId", "branchIndex", "event", "pathSha256"]].rename(
        columns={"event": "repeatedReturnEvent", "pathSha256": "pathSha256L41"}
    )
    branch = l42.merge(
        l43, on=["stateId", "branchIndex"], how="inner", suffixes=("", "_l43")
    )
    branch = branch.merge(l41, on=["stateId", "branchIndex"], how="inner")
    if len(branch) != 35_840:
        raise RuntimeError(f"L44 branch scope mismatch: {len(branch)}")
    if not (
        branch["pathSha256"].eq(branch["pathSha256L43"]).all()
        and branch["pathSha256"].eq(branch["pathSha256L41"]).all()
        and branch["resumptionFlags"].eq(branch["inheritanceFlagsL43"]).all()
    ):
        raise RuntimeError("L44 frozen branch replay identity mismatch")
    episode_rows = []
    dwell_rows = []
    for row in branch.itertuples(index=False):
        flags = json.loads(row.resumptionFlags)
        summary = summarize_binary_episode(flags)
        episode_rows.append(
            {
                "stateId": row.stateId,
                "evaluationCohort": row.evaluationCohort,
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "landmark": int(row.landmark),
                "branchIndex": int(row.branchIndex),
                "branchHalf": row.branchHalf,
                "breakObserved": bool(row.breakObserved),
                "inheritanceResumptionRun2": summary.run2_event,
                "newHereditaryEpisodeRun3": summary.run3_event,
                "persistentHereditaryEpisodeRun5": summary.run5_event,
                "oldNeighbourhoodRecoveryRun2": bool(row.event),
                "positiveOldAnchorGainAtResumption": bool(
                    row.resumptionObserved
                    and pd.notna(row.recoveryGain)
                    and row.recoveryGain > 0
                ),
                "repeatedCrossGenerationReturn": bool(row.repeatedReturnEvent),
                "postbreakOpportunities": summary.opportunities,
                "inheritedPostbreakCount": summary.positives,
                "postbreakInheritanceFraction": summary.positive_fraction,
                "maximumInheritanceRun": summary.maximum_positive_run,
                "maximumNoninheritanceRun": summary.maximum_negative_run,
                "run2CertificationOneBased": summary.run2_certification_one_based,
                "run3CertificationOneBased": summary.run3_certification_one_based,
                "run5CertificationOneBased": summary.run5_certification_one_based,
                "run2OrderNullProbability": summary.run2_order_null_probability,
                "run3OrderNullProbability": summary.run3_order_null_probability,
                "run5OrderNullProbability": summary.run5_order_null_probability,
                "transition00": summary.transition_00,
                "transition01": summary.transition_01,
                "transition10": summary.transition_10,
                "transition11": summary.transition_11,
                "recoveryGain": row.recoveryGain,
                "breakAnchorH": row.breakAnchorH,
                "certificationAnchorH": row.certificationAnchorH,
                "inheritanceFlags": json.dumps(flags),
                "targetUsesCompletedTestTrajectory": False,
            }
        )
        run_records: dict[str, list[tuple[int, bool, bool]]] = {
            "INHERITED": [],
            "NONINHERITED": [],
        }
        if flags:
            run_start = 0
            run_value = bool(flags[0])
            for position in range(1, len(flags) + 1):
                if position == len(flags) or bool(flags[position]) != run_value:
                    run_records["INHERITED" if run_value else "NONINHERITED"].append(
                        (
                            position - run_start,
                            run_start == 0,
                            position == len(flags),
                        )
                    )
                    if position < len(flags):
                        run_start = position
                        run_value = bool(flags[position])
        for regime, records in run_records.items():
            for run_index, (length, left_censored, right_censored) in enumerate(
                records
            ):
                dwell_rows.append(
                    {
                        "stateId": row.stateId,
                        "evaluationCohort": row.evaluationCohort,
                        "candidateId": row.candidateId,
                        "matrixIndex": int(row.matrixIndex),
                        "landmark": int(row.landmark),
                        "branchIndex": int(row.branchIndex),
                        "branchHalf": row.branchHalf,
                        "regime": regime,
                        "runIndex": run_index,
                        "dwellFissions": int(length),
                        "leftCensored": bool(left_censored),
                        "rightCensored": bool(right_censored),
                    }
                )
    episodes = (
        pd.DataFrame(episode_rows)
        .sort_values(["candidateId", "matrixIndex", "landmark", "branchIndex"])
        .reset_index(drop=True)
    )
    dwells = (
        pd.DataFrame(dwell_rows)
        .sort_values(
            [
                "candidateId",
                "matrixIndex",
                "landmark",
                "branchIndex",
                "regime",
                "runIndex",
            ]
        )
        .reset_index(drop=True)
    )
    return episodes, dwells


PROCESS_COLUMNS = {
    "INHERITANCE_BREAK_WITH_DEPARTURE": "breakObserved",
    "INHERITANCE_RESUMPTION_RUN2": "inheritanceResumptionRun2",
    "NEW_HEREDITARY_EPISODE_RUN3": "newHereditaryEpisodeRun3",
    "PERSISTENT_HEREDITARY_EPISODE_RUN5": "persistentHereditaryEpisodeRun5",
    "OLD_NEIGHBOURHOOD_RECOVERY_RUN2": "oldNeighbourhoodRecoveryRun2",
    "POSITIVE_OLD_ANCHOR_GAIN_AT_RESUMPTION": "positiveOldAnchorGainAtResumption",
    "REPEATED_CROSS_GENERATION_RETURN": "repeatedCrossGenerationReturn",
}


def process_trials(group: pd.DataFrame, process: str) -> pd.DataFrame:
    if process == "INHERITANCE_BREAK_WITH_DEPARTURE":
        return group
    if process == "POSITIVE_OLD_ANCHOR_GAIN_AT_RESUMPTION":
        return group[group["inheritanceResumptionRun2"]]
    if process == "REPEATED_CROSS_GENERATION_RETURN":
        return group
    return group[group["breakObserved"]]


def state_process_results(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["stateId", "evaluationCohort", "candidateId", "matrixIndex", "landmark"]
    for state_keys, raw in episodes.groupby(keys, sort=False):
        for process in PROCESSES:
            group = process_trials(raw, process)
            column = PROCESS_COLUMNS[process]
            halves = group.groupby("branchHalf")[column].agg(["count", "sum", "mean"])
            trials_a = int(halves.loc["A", "count"]) if "A" in halves.index else 0
            trials_b = int(halves.loc["B", "count"]) if "B" in halves.index else 0
            q_a = (
                float(halves.loc["A", "mean"]) if "A" in halves.index else float("nan")
            )
            q_b = (
                float(halves.loc["B", "mean"]) if "B" in halves.index else float("nan")
            )
            order_column = ORDER_PROCESSES.get(process)
            order_null = (
                float(group[order_column].mean())
                if order_column and len(group)
                else float("nan")
            )
            rows.append(
                {
                    **dict(zip(keys, state_keys, strict=True)),
                    "processId": process,
                    "trials": len(group),
                    "trialsHalfA": trials_a,
                    "trialsHalfB": trials_b,
                    "successes": int(group[column].sum()),
                    "qHat": float(group[column].mean()) if len(group) else float("nan"),
                    "qHatHalfA": q_a,
                    "qHatHalfB": q_b,
                    "meanOrderNullProbability": order_null,
                    "actualMinusOrderNull": (
                        float(group[column].mean()) - order_null
                        if order_column and len(group)
                        else float("nan")
                    ),
                    "meanPostbreakOpportunities": float(
                        group["postbreakOpportunities"].mean()
                    )
                    if len(group)
                    else float("nan"),
                    "meanInheritanceFraction": float(
                        group["postbreakInheritanceFraction"].mean()
                    )
                    if len(group)
                    else float("nan"),
                    "eligible": bool(
                        len(group) >= 32 and trials_a >= 16 and trials_b >= 16
                    ),
                    "targetUsesCompletedTestTrajectory": False,
                }
            )
    result = pd.DataFrame(rows).sort_values([*keys, "processId"]).reset_index(drop=True)
    if len(result) != 1_960:
        raise RuntimeError(f"L44 state-process scope mismatch: {len(result)}")
    return result


def corrected_variance(group: pd.DataFrame) -> tuple[float, float, float]:
    q = group["qHat"].to_numpy(float)
    trials = group["trials"].to_numpy(float)
    observed = float(np.var(q, ddof=1)) if len(q) > 1 else float("nan")
    noise = float(np.mean(q * (1 - q) / trials)) if len(q) else float("nan")
    return observed, noise, observed - noise


def resample_indices_by_matrix(
    group: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    matrix_rows = [
        indices for _, indices in group.groupby("matrixIndex").indices.items()
    ]
    selected = rng.integers(0, len(matrix_rows), size=len(matrix_rows))
    return np.concatenate([np.asarray(matrix_rows[index]) for index in selected])


def process_reliability(
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    boot_rows = []
    for keys, raw in states.groupby(
        ["evaluationCohort", "candidateId", "processId"], sort=False
    ):
        group = raw[raw["eligible"]].reset_index(drop=True)
        observed, noise, corrected = corrected_variance(group)
        rho = safe_spearman(
            group["qHatHalfA"].to_numpy(float), group["qHatHalfB"].to_numpy(float)
        )
        rng = np.random.default_rng(derived_seed("process_bootstrap", *keys))
        metrics = np.full((BOOTSTRAPS, 4), np.nan)
        if len(group):
            for replicate in range(BOOTSTRAPS):
                indices = resample_indices_by_matrix(group, rng)
                sample = group.iloc[indices]
                metrics[replicate, 0] = float(sample["qHat"].mean())
                metrics[replicate, 1] = corrected_variance(sample)[2]
                metrics[replicate, 2] = safe_spearman(
                    sample["qHatHalfA"].to_numpy(float),
                    sample["qHatHalfB"].to_numpy(float),
                )
                metrics[replicate, 3] = float(sample["actualMinusOrderNull"].mean())
        for replicate, values in enumerate(metrics):
            boot_rows.append(
                {
                    "evaluationCohort": keys[0],
                    "candidateId": keys[1],
                    "processId": keys[2],
                    "replicate": replicate,
                    "meanQ": values[0],
                    "correctedVariance": values[1],
                    "splitHalfSpearman": values[2],
                    "meanOrderExcess": values[3],
                }
            )
        q_ci = interval(metrics[:, 0])
        var_ci = interval(metrics[:, 1])
        rho_ci = interval(metrics[:, 2])
        order_ci = interval(metrics[:, 3])
        intermediate = int(((group["qHat"] > 0.1) & (group["qHat"] < 0.9)).sum())
        rows.append(
            {
                "evaluationCohort": keys[0],
                "candidateId": keys[1],
                "processId": keys[2],
                "states": len(raw),
                "eligibleStates": len(group),
                "meanQ": float(group["qHat"].mean()) if len(group) else float("nan"),
                "meanQLower95": q_ci[0],
                "meanQUpper95": q_ci[1],
                "intermediateStates": intermediate,
                "observedBetweenStateVariance": observed,
                "estimatedBinomialNoiseVariance": noise,
                "correctedBetweenStateVariance": corrected,
                "correctedVarianceLower95": var_ci[0],
                "correctedVarianceUpper95": var_ci[1],
                "splitHalfSpearman": rho,
                "splitHalfLower95": rho_ci[0],
                "splitHalfUpper95": rho_ci[1],
                "meanOrderExcess": float(group["actualMinusOrderNull"].mean())
                if len(group)
                else float("nan"),
                "orderExcessLower95": order_ci[0],
                "orderExcessUpper95": order_ci[1],
                "reliabilityGatePassed": bool(
                    len(group) >= 32
                    and intermediate >= 20
                    and var_ci[0] > 0
                    and rho > 0.5
                    and rho_ci[0] > 0.3
                ),
                "orderExcessGatePassed": bool(
                    keys[2] in ORDER_PROCESSES and order_ci[0] > 0
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


def order_baseline_results(
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for process, null_column in ORDER_PROCESSES.items():
        outcome_column = PROCESS_COLUMNS[process]
        eligible = episodes[episodes["breakObserved"]].copy()
        eligible["observed"] = eligible[outcome_column].astype(float)
        eligible["predictedByFixedCountOrderNull"] = eligible[null_column].astype(float)
        probabilities = np.clip(
            eligible["predictedByFixedCountOrderNull"].to_numpy(float),
            1e-12,
            1 - 1e-12,
        )
        eligible["orderNullBrier"] = (
            eligible["observed"].to_numpy(float) - probabilities
        ) ** 2
        eligible["orderNullLogLoss"] = -(
            eligible["observed"].to_numpy(float) * np.log(probabilities)
            + (1 - eligible["observed"].to_numpy(float)) * np.log(1 - probabilities)
        )
        eligible["orderExcess"] = (
            eligible["observed"] - eligible["predictedByFixedCountOrderNull"]
        )
        for keys, group in eligible.groupby(
            ["evaluationCohort", "candidateId"], sort=False
        ):
            rng = np.random.default_rng(
                derived_seed("baseline_bootstrap", *keys, process)
            )
            matrix_groups = [
                indices for _, indices in group.groupby("matrixIndex").indices.items()
            ]
            values = np.empty((BOOTSTRAPS, 4), dtype=np.float64)
            for replicate in range(BOOTSTRAPS):
                chosen = rng.integers(0, len(matrix_groups), size=len(matrix_groups))
                sample = group.iloc[
                    np.concatenate(
                        [np.asarray(matrix_groups[index]) for index in chosen]
                    )
                ]
                values[replicate] = (
                    sample["observed"].mean(),
                    sample["predictedByFixedCountOrderNull"].mean(),
                    sample["orderExcess"].mean(),
                    sample["orderNullBrier"].mean(),
                )
            for replicate, value in enumerate(values):
                bootstrap_rows.append(
                    {
                        "evaluationCohort": keys[0],
                        "candidateId": keys[1],
                        "processId": process,
                        "replicate": replicate,
                        "observedPrevalence": value[0],
                        "orderNullPrevalence": value[1],
                        "orderExcess": value[2],
                        "orderNullBrier": value[3],
                    }
                )
            excess_ci = interval(values[:, 2])
            rows.append(
                {
                    "evaluationCohort": keys[0],
                    "candidateId": keys[1],
                    "processId": process,
                    "branches": len(group),
                    "matrices": group["matrixIndex"].nunique(),
                    "observedPrevalence": float(group["observed"].mean()),
                    "orderNullPrevalence": float(
                        group["predictedByFixedCountOrderNull"].mean()
                    ),
                    "orderExcess": float(group["orderExcess"].mean()),
                    "orderExcessLower95": excess_ci[0],
                    "orderExcessUpper95": excess_ci[1],
                    "orderNullBrier": float(group["orderNullBrier"].mean()),
                    "orderNullLogLoss": float(group["orderNullLogLoss"].mean()),
                    "positiveOrderExcessGatePassed": bool(excess_ci[0] > 0),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def markov_state_results(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["stateId", "evaluationCohort", "candidateId", "matrixIndex", "landmark"]
    for state_keys, raw in episodes.groupby(keys, sort=False):
        group = raw[raw["breakObserved"]]
        half_a = [
            json.loads(value)
            for value in group.loc[group["branchHalf"].eq("A"), "inheritanceFlags"]
        ]
        half_b = [
            json.loads(value)
            for value in group.loc[group["branchHalf"].eq("B"), "inheritanceFlags"]
        ]
        crossfit = crossfit_markov_gain_bits(half_a, half_b)
        n00 = int(group["transition00"].sum())
        n01 = int(group["transition01"].sum())
        n10 = int(group["transition10"].sum())
        n11 = int(group["transition11"].sum())
        p01 = (n01 + 0.5) / (n00 + n01 + 1.0)
        p11 = (n11 + 0.5) / (n10 + n11 + 1.0)
        rows.append(
            {
                **dict(zip(keys, state_keys, strict=True)),
                "breakBranches": len(group),
                "breakBranchesHalfA": len(half_a),
                "breakBranchesHalfB": len(half_b),
                "transition00": n00,
                "transition01": n01,
                "transition10": n10,
                "transition11": n11,
                "probabilityInheritanceAfterNoninheritance": p01,
                "probabilityInheritanceAfterInheritance": p11,
                "persistenceContrast": p11 - p01,
                **crossfit,
                "eligible": bool(
                    len(group) >= 32
                    and len(half_a) >= 16
                    and len(half_b) >= 16
                    and crossfit["transitions"] > 0
                ),
                "targetUsesCompletedTestTrajectory": False,
            }
        )
    result = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    if len(result) != 280:
        raise RuntimeError(f"L44 Markov-state scope mismatch: {len(result)}")
    return result


def markov_group_results(
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for keys, raw in states.groupby(["evaluationCohort", "candidateId"], sort=False):
        group = raw[raw["eligible"]].reset_index(drop=True)
        rng = np.random.default_rng(
            derived_seed("baseline_bootstrap", *keys, "MARKOV_GAIN")
        )
        values = np.full((BOOTSTRAPS, 4), np.nan, dtype=np.float64)
        if len(group):
            for replicate in range(BOOTSTRAPS):
                sample = group.iloc[resample_indices_by_matrix(group, rng)]
                values[replicate] = (
                    sample["markovGainBitsPerTransition"].mean(),
                    sample["probabilityInheritanceAfterNoninheritance"].mean(),
                    sample["probabilityInheritanceAfterInheritance"].mean(),
                    sample["persistenceContrast"].mean(),
                )
        for replicate, value in enumerate(values):
            bootstrap_rows.append(
                {
                    "evaluationCohort": keys[0],
                    "candidateId": keys[1],
                    "replicate": replicate,
                    "markovGainBitsPerTransition": value[0],
                    "probabilityInheritanceAfterNoninheritance": value[1],
                    "probabilityInheritanceAfterInheritance": value[2],
                    "persistenceContrast": value[3],
                }
            )
        gain_ci = interval(values[:, 0])
        contrast_ci = interval(values[:, 3])
        rows.append(
            {
                "evaluationCohort": keys[0],
                "candidateId": keys[1],
                "states": len(raw),
                "eligibleStates": len(group),
                "markovGainBitsPerTransition": float(
                    group["markovGainBitsPerTransition"].mean()
                )
                if len(group)
                else float("nan"),
                "markovGainLower95": gain_ci[0],
                "markovGainUpper95": gain_ci[1],
                "probabilityInheritanceAfterNoninheritance": float(
                    group["probabilityInheritanceAfterNoninheritance"].mean()
                )
                if len(group)
                else float("nan"),
                "probabilityInheritanceAfterInheritance": float(
                    group["probabilityInheritanceAfterInheritance"].mean()
                )
                if len(group)
                else float("nan"),
                "persistenceContrast": float(group["persistenceContrast"].mean())
                if len(group)
                else float("nan"),
                "persistenceContrastLower95": contrast_ci[0],
                "persistenceContrastUpper95": contrast_ci[1],
                "markovBeyondIidGatePassed": bool(len(group) >= 32 and gain_ci[0] > 0),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def dwell_and_hazard_results(
    dwells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    hazard_rows: list[dict[str, Any]] = []
    for keys, group in dwells.groupby(
        ["evaluationCohort", "candidateId", "regime"], sort=False
    ):
        values = group["dwellFissions"].to_numpy(float)
        summary_rows.append(
            {
                "evaluationCohort": keys[0],
                "candidateId": keys[1],
                "regime": keys[2],
                "runs": len(group),
                "meanDwellFissions": float(np.mean(values)),
                "medianDwellFissions": float(np.median(values)),
                "p90DwellFissions": float(np.quantile(values, 0.9)),
                "rightCensoredFraction": float(group["rightCensored"].mean()),
                "leftCensoredFraction": float(group["leftCensored"].mean()),
            }
        )
        for elapsed in range(1, int(values.max()) + 1):
            at_risk = group[group["dwellFissions"] >= elapsed]
            ended = at_risk[
                (at_risk["dwellFissions"] == elapsed) & ~at_risk["rightCensored"]
            ]
            hazard_rows.append(
                {
                    "evaluationCohort": keys[0],
                    "candidateId": keys[1],
                    "regime": keys[2],
                    "elapsedFissions": elapsed,
                    "atRiskRuns": len(at_risk),
                    "observedEnds": len(ended),
                    "empiricalEndHazard": len(ended) / len(at_risk),
                    "empiricalSurvival": len(at_risk) / len(group),
                    "rightBoundaryCensoringRetained": True,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(hazard_rows)


def prefix_control_registry() -> pd.DataFrame:
    boundaries = pd.read_parquet(L42_ROOT / "prefix_boundary_registry.parquet")
    summaries = pd.read_parquet(L42_ROOT / "prefix_state_summary.parquet")
    prefix = L43.L42.L41.L40.L39.prefix_controls(boundaries, summaries)
    response = pd.read_parquet(L42_ROOT / "response_registry.parquet")
    return prefix.merge(
        response[
            [
                "stateId",
                "currentMass",
                "currentGenerationLocalStep",
                "currentCompletedFissions",
            ]
        ],
        on="stateId",
        how="left",
        validate="one_to_one",
    )


def process_control_relationships(
    states: pd.DataFrame, prefix: pd.DataFrame
) -> pd.DataFrame:
    prefix_columns = [
        "stateId",
        "prefixInheritanceFraction",
        "prefixTrailingInheritanceRun",
        "prefixMaximumInheritanceRun",
        "prefixBoundaryCount",
        "currentMass",
        "currentGenerationLocalStep",
    ]
    merged = states.merge(
        prefix[prefix_columns], on="stateId", how="left", validate="many_to_one"
    )
    features = {
        "PREFIX_INHERITANCE_FRACTION": (
            "prefixInheritanceFraction",
            "PAST_OBSERVABLE",
        ),
        "PREFIX_TRAILING_STREAK": (
            "prefixTrailingInheritanceRun",
            "PAST_OBSERVABLE",
        ),
        "PREFIX_MAXIMUM_STREAK": (
            "prefixMaximumInheritanceRun",
            "PAST_OBSERVABLE",
        ),
        "PREFIX_OPPORTUNITIES": ("prefixBoundaryCount", "PAST_OBSERVABLE"),
        "CURRENT_MASS": ("currentMass", "PAST_OBSERVABLE"),
        "GENERATION_PHASE": (
            "currentGenerationLocalStep",
            "PAST_OBSERVABLE",
        ),
        "FUTURE_FISSION_OPPORTUNITIES": (
            "meanPostbreakOpportunities",
            "FUTURE_OUTCOME_DECOMPOSITION_ONLY",
        ),
        "FUTURE_INHERITANCE_FREQUENCY": (
            "meanInheritanceFraction",
            "FUTURE_OUTCOME_DECOMPOSITION_ONLY",
        ),
    }
    rows: list[dict[str, Any]] = []
    for keys, group in merged[merged["eligible"]].groupby(
        ["evaluationCohort", "candidateId", "processId"], sort=False
    ):
        for feature_id, (column, timing) in features.items():
            rows.append(
                {
                    "evaluationCohort": keys[0],
                    "candidateId": keys[1],
                    "processId": keys[2],
                    "controlId": feature_id,
                    "timingClass": timing,
                    "states": int(group[["qHat", column]].dropna().shape[0]),
                    "spearmanWithProcessQ": safe_spearman(
                        group["qHat"].to_numpy(float), group[column].to_numpy(float)
                    ),
                    "targetUsesCompletedTestTrajectory": False,
                }
            )
    return pd.DataFrame(rows)


def process_relationship_matrix(states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = states[states["eligible"]]
    for keys, group in eligible.groupby(
        ["evaluationCohort", "candidateId"], sort=False
    ):
        pivot = group.pivot(index="stateId", columns="processId", values="qHat")
        for left_index, left in enumerate(PROCESSES):
            for right in PROCESSES[left_index + 1 :]:
                rows.append(
                    {
                        "evaluationCohort": keys[0],
                        "candidateId": keys[1],
                        "leftProcessId": left,
                        "rightProcessId": right,
                        "states": int(pivot[[left, right]].dropna().shape[0]),
                        "spearman": safe_spearman(
                            pivot[left].to_numpy(float), pivot[right].to_numpy(float)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plasticity_results(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, raw in episodes.groupby(["evaluationCohort", "candidateId"], sort=False):
        break_group = raw[raw["breakObserved"]]
        resumed = break_group[break_group["inheritanceResumptionRun2"]]
        rows.append(
            {
                "evaluationCohort": keys[0],
                "candidateId": keys[1],
                "branches": len(raw),
                "breakPrevalence": float(raw["breakObserved"].mean()),
                "resumptionGivenBreak": float(
                    break_group["inheritanceResumptionRun2"].mean()
                ),
                "oldNeighbourhoodRecoveryGivenBreak": float(
                    break_group["oldNeighbourhoodRecoveryRun2"].mean()
                ),
                "positiveOldAnchorGainGivenResumption": float(
                    resumed["positiveOldAnchorGainAtResumption"].mean()
                ),
                "meanOldAnchorGainGivenResumption": float(
                    resumed["recoveryGain"].mean()
                ),
                "medianOldAnchorGainGivenResumption": float(
                    resumed["recoveryGain"].median()
                ),
                "interpretation": "GENERIC_HEREDITY_RESUMPTION_WITH_OLD_COMPOSITION_DIVERGENCE",
            }
        )
    return pd.DataFrame(rows)


def _all_evaluation_groups_pass(
    frame: pd.DataFrame, process: str | None, column: str
) -> bool:
    subset = frame[frame["evaluationCohort"].isin(EVALUATION_COHORTS)]
    if process is not None and "processId" in subset.columns:
        subset = subset[subset["processId"].eq(process)]
    return bool(len(subset) == 4 and subset[column].astype(bool).all())


def scientific_gates(
    reliability: pd.DataFrame,
    order_results: pd.DataFrame,
    markov: pd.DataFrame,
    plasticity: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str, str | None]:
    gate_rows: list[dict[str, Any]] = []
    reliable: dict[str, bool] = {}
    ordered: dict[str, bool] = {}
    for process in PROCESSES:
        reliable[process] = _all_evaluation_groups_pass(
            reliability, process, "reliabilityGatePassed"
        )
        ordered[process] = (
            _all_evaluation_groups_pass(
                order_results, process, "positiveOrderExcessGatePassed"
            )
            if process in ORDER_PROCESSES
            else False
        )
        gate_rows.append(
            {
                "gateId": f"PROCESS::{process}",
                "processId": process,
                "reliableAllEvaluationGroups": reliable[process],
                "positiveOrderExcessAllEvaluationGroups": ordered[process],
                "passed": bool(
                    reliable[process]
                    and (ordered[process] if process in ORDER_PROCESSES else True)
                ),
            }
        )

    markov_pass = _all_evaluation_groups_pass(markov, None, "markovBeyondIidGatePassed")
    eval_plasticity = plasticity[
        plasticity["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    plasticity_direction = bool(
        len(eval_plasticity) == 4
        and (eval_plasticity["resumptionGivenBreak"] > 0.5).all()
        and (eval_plasticity["oldNeighbourhoodRecoveryGivenBreak"] < 0.1).all()
        and (eval_plasticity["meanOldAnchorGainGivenResumption"] < 0).all()
    )
    ordered_episode = any(
        reliable[process] and ordered[process] for process in ORDER_PROCESSES
    )
    classifications: list[str] = []
    if any(reliable.values()):
        classifications.append("PROCESS_FAMILY_IDENTIFIED")
    else:
        classifications.append("PROCESS_FAMILY_NONIDENTIFIABLE")
    if plasticity_direction:
        classifications.append(
            "PLASTIC_HEREDITY_REGIME_SWITCHING_DIRECTIONALLY_SUPPORTED"
        )
    if markov_pass:
        classifications.append("STICKY_HEREDITARY_EPISODES_BEYOND_IID")
    if ordered_episode:
        classifications.append("ORDERED_HEREDITY_EPISODE_SIGNAL")
    else:
        classifications.append("INHERITANCE_FREQUENCY_SUFFICIENT_FOR_ORDERED_EPISODES")
    classifications.append("NOT_PROMOTABLE_AS_CONFIRMED")

    if (
        reliable["NEW_HEREDITARY_EPISODE_RUN3"]
        and ordered["NEW_HEREDITARY_EPISODE_RUN3"]
    ):
        selected = "NEW_HEREDITARY_EPISODE_RUN3"
        next_theme = "L45_PHI_INCREMENTAL_VALUE_FOR_NEW_HEREDITARY_EPISODE"
    elif (
        reliable["INHERITANCE_RESUMPTION_RUN2"]
        and ordered["INHERITANCE_RESUMPTION_RUN2"]
    ):
        selected = "INHERITANCE_RESUMPTION_RUN2"
        next_theme = "L45_PHI_INCREMENTAL_VALUE_FOR_INHERITANCE_RESUMPTION"
    elif reliable["INHERITANCE_BREAK_WITH_DEPARTURE"]:
        selected = "INHERITANCE_BREAK_WITH_DEPARTURE"
        next_theme = "L45_PHI_INCREMENTAL_VALUE_FOR_INHERITANCE_BREAK"
    else:
        selected = None
        next_theme = "L45_PHI_PROCESS_IDENTIFIABILITY_ONLY"

    gate_rows.extend(
        [
            {
                "gateId": "MARKOV_BEYOND_IID_ALL_EVALUATION_GROUPS",
                "processId": None,
                "reliableAllEvaluationGroups": False,
                "positiveOrderExcessAllEvaluationGroups": False,
                "passed": markov_pass,
            },
            {
                "gateId": "PLASTIC_HEREDITY_DIRECTION_ALL_EVALUATION_GROUPS",
                "processId": None,
                "reliableAllEvaluationGroups": False,
                "positiveOrderExcessAllEvaluationGroups": False,
                "passed": plasticity_direction,
            },
        ]
    )
    return pd.DataFrame(gate_rows), classifications, next_theme, selected


def benchmark_projection() -> dict[str, Any]:
    return {
        "schema": "eidosoma.e01.s19_l44.benchmark_projection.v1",
        "basis": "35,840 frozen F12 branches; analysis-only binary summaries and 4,096 matrix bootstraps",
        "projectedWallSeconds": 3600,
        "projectedCpuHoursConservative": 8,
        "wallHoursCeiling": 24,
        "cpuHoursCeiling": 40,
        "workersMaximum": 8,
        "newScientificSimulation": False,
        "status": "PASS",
    }


def make_figures(
    reliability: pd.DataFrame,
    order_results: pd.DataFrame,
    markov: pd.DataFrame,
    hazards: pd.DataFrame,
    relationships: pd.DataFrame,
    plasticity: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    figure_root = BUILD_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    short = {
        "INHERITANCE_BREAK_WITH_DEPARTURE": "break",
        "INHERITANCE_RESUMPTION_RUN2": "resume-2",
        "NEW_HEREDITARY_EPISODE_RUN3": "episode-3",
        "PERSISTENT_HEREDITARY_EPISODE_RUN5": "persist-5",
        "OLD_NEIGHBOURHOOD_RECOVERY_RUN2": "old-return",
        "POSITIVE_OLD_ANCHOR_GAIN_AT_RESUMPTION": "positive-gain",
        "REPEATED_CROSS_GENERATION_RETURN": "repeat-return",
    }
    evaluation = reliability[
        reliability["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [
        f"{row.evaluationCohort.split('_')[-1][:4]}-{row.candidateId[-2:]}\n{short[row.processId]}"
        for row in evaluation.itertuples(index=False)
    ]
    x = np.arange(len(evaluation))
    ax.bar(x, evaluation["meanQ"], color="#4c78a8")
    ax.vlines(
        x,
        evaluation["meanQLower95"],
        evaluation["meanQUpper95"],
        color="black",
        lw=0.8,
    )
    ax.set_xticks(x, labels, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("Branch probability")
    ax.set_title("Seven distinct heredity-process probabilities")
    fig.tight_layout()
    fig.savefig(figure_root / "01_process_family_prevalence.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for process, group in evaluation.groupby("processId"):
        ax.scatter(
            group["splitHalfSpearman"],
            group["correctedBetweenStateVariance"],
            label=short[process],
        )
    ax.axvline(0.5, color="grey", ls="--")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Split-half state-rank Spearman")
    ax.set_ylabel("Corrected between-state variance")
    ax.set_title("Process committor identifiability")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_root / "02_process_reliability.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = order_results[order_results["evaluationCohort"].isin(EVALUATION_COHORTS)]
    x = np.arange(len(ordered))
    ax.bar(x, ordered["orderExcess"], color="#f58518")
    ax.vlines(
        x, ordered["orderExcessLower95"], ordered["orderExcessUpper95"], color="black"
    )
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(
        x,
        [
            f"{r.evaluationCohort.split('_')[-1][:4]}-{r.candidateId[-2:]}\n{short[r.processId]}"
            for r in ordered.itertuples(index=False)
        ],
        rotation=55,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Observed minus fixed-count order-null probability")
    ax.set_title("Does temporal ordering add beyond inheritance count?")
    fig.tight_layout()
    fig.savefig(figure_root / "03_fixed_count_order_excess.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    markov_eval = markov[markov["evaluationCohort"].isin(EVALUATION_COHORTS)]
    labels = [
        f"{r.evaluationCohort.split('_')[-1][:4]}-{r.candidateId[-2:]}"
        for r in markov_eval.itertuples(index=False)
    ]
    axes[0].scatter(
        markov_eval["probabilityInheritanceAfterNoninheritance"],
        markov_eval["probabilityInheritanceAfterInheritance"],
    )
    for label, row in zip(labels, markov_eval.itertuples(index=False), strict=True):
        axes[0].annotate(
            label,
            (
                row.probabilityInheritanceAfterNoninheritance,
                row.probabilityInheritanceAfterInheritance,
            ),
            fontsize=7,
        )
    low = min(
        markov_eval["probabilityInheritanceAfterNoninheritance"].min(),
        markov_eval["probabilityInheritanceAfterInheritance"].min(),
    )
    high = max(
        markov_eval["probabilityInheritanceAfterNoninheritance"].max(),
        markov_eval["probabilityInheritanceAfterInheritance"].max(),
    )
    axes[0].plot([low, high], [low, high], color="grey", ls="--")
    axes[0].set_xlabel("P(inherit | previous noninherit)")
    axes[0].set_ylabel("P(inherit | previous inherit)")
    axes[0].set_title("First-order persistence")
    axes[1].bar(np.arange(len(markov_eval)), markov_eval["markovGainBitsPerTransition"])
    axes[1].vlines(
        np.arange(len(markov_eval)),
        markov_eval["markovGainLower95"],
        markov_eval["markovGainUpper95"],
        color="black",
    )
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xticks(np.arange(len(markov_eval)), labels, rotation=35, ha="right")
    axes[1].set_ylabel("Cross-fit gain (bits/transition)")
    axes[1].set_title("Markov versus IID inheritance")
    fig.tight_layout()
    fig.savefig(figure_root / "04_markov_renewal_baseline.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for keys, group in hazards[
        hazards["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].groupby(["candidateId", "regime"]):
        ax.step(
            group["elapsedFissions"],
            group["empiricalSurvival"],
            where="post",
            label=f"C{keys[0][-2:]} {keys[1].lower()}",
        )
    ax.set_xlabel("Dwell time (future fissions)")
    ax.set_ylabel("Empirical survival of episode")
    ax.set_title("Inheritance and noninheritance dwell structure")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_root / "05_dwell_survival.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    relation_eval = relationships[
        relationships["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    pivot = relation_eval.pivot_table(
        index="leftProcessId",
        columns="rightProcessId",
        values="spearman",
        aggfunc="mean",
    ).reindex(index=PROCESSES, columns=PROCESSES)
    image = ax.imshow(
        pivot.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto"
    )
    ax.set_xticks(
        range(len(PROCESSES)), [short[p] for p in PROCESSES], rotation=45, ha="right"
    )
    ax.set_yticks(range(len(PROCESSES)), [short[p] for p in PROCESSES])
    ax.set_title("Cross-process state-probability relationships")
    fig.colorbar(image, ax=ax, label="Mean Spearman")
    fig.tight_layout()
    fig.savefig(figure_root / "06_process_relationships.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    p = plasticity[plasticity["evaluationCohort"].isin(EVALUATION_COHORTS)]
    x = np.arange(len(p))
    width = 0.25
    ax.bar(x - width, p["resumptionGivenBreak"], width, label="generic resumption")
    ax.bar(
        x,
        p["oldNeighbourhoodRecoveryGivenBreak"],
        width,
        label="old-neighbourhood return",
    )
    ax.bar(
        x + width,
        p["positiveOldAnchorGainGivenResumption"],
        width,
        label="positive old-anchor gain",
    )
    ax.set_xticks(
        x,
        [
            f"{r.evaluationCohort.split('_')[-1][:4]}-{r.candidateId[-2:]}"
            for r in p.itertuples(index=False)
        ],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Plastic heredity: resumption is not restoration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "07_plasticity_decomposition.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    matrix = gates.set_index("gateId")[["passed"]].astype(int)
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks([0], ["passed"])
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
    ax.set_title("L44 preregistered gate matrix")
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(figure_root / "08_decision_matrix.png", dpi=160)
    plt.close(fig)


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
        "schema": "eidosoma.e01.s19_l44.artifact_manifest.v1",
        "loopId": LOOP_ID,
        "files": rows,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def append_ledgers(
    classifications: list[str],
    timestamp: str,
    next_theme: str,
    selected_process: str | None,
) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L43 showed a strong directional separation: generic inheritance resumes, but similarity to the old pre-break composition decreases by about 0.26 rather than recovering.",
            "failureOrAmbiguityTargeted": "Whether inheritance, breaks, resumption, episode ordering, persistence, exact return and old-anchor gain form an identifiable process family beyond inheritance frequency and opportunities.",
            "informationGainRationale": "Seven outcomes are frozen jointly, with exact fixed-count order nulls, cross-fit IID/Markov baselines and no new branch simulation.",
            "learned": "L44 process-family and renewal contract locked before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L41-L43 frozen paths; reviewer plastic-heredity/regime-switching framing; renewal and Markov baselines.",
            "proposedNextTest": "Audit the frozen F12 branch summaries as distinct processes.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Sticky inheritance, plastic regime switching, count-sufficient apparent streaks, or no transferable process committor.",
            "selectedHypotheses": "A process-level family may be identifiable even though exact old-state restoration is not.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "A single fixed-composition attractor is the only meaningful organizational object.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A usable process must be nondegenerate, split-half reliable in both evaluation cohorts and candidates, and exceed frequency/order baselines when temporal ordering is claimed.",
            "failureOrAmbiguityTargeted": "Process identifiability and side-quest target selection.",
            "informationGainRationale": "The exact same branch ensemble supports all outcomes, isolating definition and temporal order from simulator variance.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L44 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": selected_process or "NO_PROCESS_SELECTED",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Inheritance streaks alone necessarily demonstrate memory beyond ordinary inheritance frequency.",
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
        + f"\n\n## {LOOP_ID} — plastic-heredity process-family identifiability\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Selected L45 process:** `{selected_process or 'NONE'}`.\n"
        + f"- **Next:** `{next_theme}`.\n",
    )

    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    candidate = {
        "branchCount": 7,
        "bundleId": "L44_PLASTIC_HEREDITY_PROCESS_FAMILY",
        "candidateId": "S19-L44-PROCESS-FAMILY-IDENTIFIABILITY",
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
        "proposedSpecification": "seven frozen fission-clock inheritance/break/resumption/return processes with exact count-order and cross-fit Markov baselines",
        "rankingScore": 29.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": selected_process is not None,
        "selectionReason": "L43_PLASTIC_HEREDITY_DIRECTION_AND_REVIEWER_PROCESS_FAMILY_AUDIT",
        "sourceGrounding": 5,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidates_path,
        pd.concat(
            [
                candidates,
                pd.DataFrame([candidate]).reindex(columns=candidates.columns),
            ],
            ignore_index=True,
        ),
    )

    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    source_rows = [
        {
            "commitOrVersion": None,
            "evidenceClass": row.evidenceClass,
            "finding": f"{row.finding}; L44 use: {row.frozenUse}",
            "licenseStatus": "PUBLIC_METADATA_OR_WORKSPACE_EVIDENCE",
            "redistributionStatus": "REFERENCE_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": row.sourceId,
            "sourceType": row.evidenceClass,
            "treeIdentity": None,
            "url": row.url,
        }
        for row in source_grounding_registry().itertuples(index=False)
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
            "selectedDiscoveryLead": selected_process,
            "newMatrices": 0,
            "newTrajectories": 0,
            "newBranchStreams": 0,
            "nextStepActive": True,
        }
    )
    registry["proposedNextLoopTheme"] = next_theme
    registry["proposedNextLoopActive"] = True
    registry["authorizationUpperBound"] = "S19-L65"
    BASE.atomic_text(registry_path, yaml.safe_dump(registry, sort_keys=False))

    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["history"].append(
        {
            "decision": "S19_L44_COMPLETE_AUTONOMOUS_CONTINUATION",
            "loopId": LOOP_ID,
            "nextLoopAuthorized": True,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "s20Activated": False,
            "scope": VERSION,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L65"
    BASE.write_json(history_path, history)


def report_text(
    reliability: pd.DataFrame,
    order_results: pd.DataFrame,
    markov: pd.DataFrame,
    dwell_summary: pd.DataFrame,
    plasticity: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
    selected_process: str | None,
) -> str:
    evaluation_reliability = reliability[
        reliability["evaluationCohort"].isin(EVALUATION_COHORTS)
    ][
        [
            "evaluationCohort",
            "candidateId",
            "processId",
            "meanQ",
            "intermediateStates",
            "correctedBetweenStateVariance",
            "correctedVarianceLower95",
            "splitHalfSpearman",
            "splitHalfLower95",
            "reliabilityGatePassed",
        ]
    ]
    evaluation_order = order_results[
        order_results["evaluationCohort"].isin(EVALUATION_COHORTS)
    ][
        [
            "evaluationCohort",
            "candidateId",
            "processId",
            "observedPrevalence",
            "orderNullPrevalence",
            "orderExcess",
            "orderExcessLower95",
            "orderExcessUpper95",
            "positiveOrderExcessGatePassed",
        ]
    ]
    evaluation_markov = markov[markov["evaluationCohort"].isin(EVALUATION_COHORTS)]
    evaluation_plasticity = plasticity[
        plasticity["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    return f"""# S19-L44 — Plastic-Heredity Process-Family Identifiability

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the extended L19–L65 autonomous sequence.
- **Classifications:** {", ".join(f"`{item}`" for item in classifications)}
- **Validation:** immutable L43-and-earlier baseline; eight deterministic fixtures; exact reuse of 35,840 frozen F12 branches; seven separately named process outcomes; 4,096 catalytic-matrix bootstraps; exact fixed-count order nulls; cross-fit IID/Markov scoring; exact full regeneration; storage and artifact hashes.
- **Selected L45 side-quest process:** `{selected_process or "NONE — identifiability-only"}`.
- **Recommended next action:** `{next_theme}`.

## Frozen question

Which among ordinary inheritance break, generic resumption, three- and five-fission inheritance episodes, strict old-neighbourhood recovery, positive old-anchor recovery gain, and repeated cross-generation return is a nondegenerate, state-dependent process across both simulator candidates and independent cohorts? For run outcomes, is the actual temporal order more informative than the exact count-matched random-order expectation? For the underlying inheritance sequence, does a cross-fitted first-order Markov model improve on an IID inheritance-frequency baseline?

No new matrix, trajectory, branch stream, threshold, horizon, run length or outcome was generated or searched. `NEW_HEREDITARY_EPISODE_RUN3` and `PERSISTENT_HEREDITARY_EPISODE_RUN5` are explicitly operational inheritance-only constructs, not claims that a new composition or function has been identified.

## Anchor results

### Process probabilities and committor reliability

{evaluation_reliability.to_markdown(index=False)}

### Exact fixed-count temporal-order baselines

{evaluation_order.to_markdown(index=False)}

### Cross-fit first-order Markov versus IID inheritance

{evaluation_markov.to_markdown(index=False)}

### Plasticity decomposition

{evaluation_plasticity.to_markdown(index=False)}

### Dwell summaries

{dwell_summary[dwell_summary["evaluationCohort"].isin(EVALUATION_COHORTS)].to_markdown(index=False)}

### Scientific gates

{gates.to_markdown(index=False)}

## Interpretation

This loop treats inheritance frequency, temporal ordering, and restoration to an old composition as different scientific objects. Generic resumption after a genuine break is not called homeostasis unless the lineage also returns toward its online pre-break composition. Likewise, a run of inherited fissions is not treated as memory unless it exceeds the exact probability expected from the same number of inherited fissions placed in random order.

The Markov comparison measures whether knowing the immediately preceding inheritance state improves held-out transition log loss over the marginal inheritance rate. It is a local temporal-dependence diagnostic, not proof of collective memory, error correction, a stable attractor, or biological replication. Dwell estimates preserve right-boundary censoring and are descriptive within the twelve-fission branch horizon.

## Provenance and validation

- Repository lock: `{runtime["repositoryHead"]}`.
- Analysis execution: vectorized/serial controller with one numerical-library thread; up to eight CPUs were permitted, but no new branch simulation was required.
- GPU hours: `0`.
- New matrices/trajectories/branch streams: `0/0/0`.
- Reused frozen F12 branches: `{runtime["reusedBranchStreams"]}`.
- Wall time: `{runtime["wallSeconds"]:.2f}` seconds.
- S01–S18, V1/V2 and S19-L01–L43 remain unchanged.

## Reproduction

```bash
PYTHONPATH=src pytest -q tests/e01/test_s19_l44.py
python -m ruff check src/e01_onset_discovery/heredity_process_family.py scripts/e01/run_s19_l44_plastic_heredity_process_family.py tests/e01/test_s19_l44.py
python scripts/e01/run_s19_l44_plastic_heredity_process_family.py --prepare-lock
python scripts/e01/run_s19_l44_plastic_heredity_process_family.py
```
"""


def compute_all() -> tuple[dict[str, pd.DataFrame], list[str], str, str | None]:
    episodes, dwells = load_branch_family()
    states = state_process_results(episodes)
    reliability, reliability_bootstrap = process_reliability(states)
    order_results, order_bootstrap = order_baseline_results(episodes)
    markov_states = markov_state_results(episodes)
    markov_groups, markov_bootstrap = markov_group_results(markov_states)
    dwell_summary, dwell_hazards = dwell_and_hazard_results(dwells)
    prefix = prefix_control_registry()
    controls = process_control_relationships(states, prefix)
    relationships = process_relationship_matrix(states)
    plasticity = plasticity_results(episodes)
    gates, classifications, next_theme, selected_process = scientific_gates(
        reliability, order_results, markov_groups, plasticity
    )
    tables = {
        "branch_episode_results.parquet": episodes,
        "episode_dwell_results.parquet": dwells,
        "state_process_results.parquet": states,
        "process_reliability_results.parquet": reliability,
        "process_reliability_bootstrap.parquet": reliability_bootstrap,
        "fixed_count_order_baseline_results.parquet": order_results,
        "fixed_count_order_baseline_bootstrap.parquet": order_bootstrap,
        "markov_state_results.parquet": markov_states,
        "markov_group_results.parquet": markov_groups,
        "markov_group_bootstrap.parquet": markov_bootstrap,
        "dwell_summary.parquet": dwell_summary,
        "dwell_hazard_results.parquet": dwell_hazards,
        "prefix_control_registry.parquet": prefix,
        "process_control_relationships.parquet": controls,
        "process_relationship_matrix.parquet": relationships,
        "plasticity_decomposition.parquet": plasticity,
        "scientific_gate_results.parquet": gates,
    }
    return tables, classifications, next_theme, selected_process


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L44 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L44 local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    analysis = analysis_seed_manifest()
    firewall = seed_firewall(analysis)
    benchmark = benchmark_projection()
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
    ):
        raise RuntimeError("L44 preoutcome validation failed")

    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L44 decision record\n\n"
        "The human authorized continuation through L65 and directed L44 to preserve the latest process framing before a bounded L45 information-dynamics side quest. L43 established that generic inheritance resumes after disruption while similarity to the online pre-break composition decreases strongly. Before opening any L44 result, this loop freezes seven distinct fission-clock process outcomes and exact count-order, IID, Markov, opportunity, phase and frequency controls. It does not call any one outcome the replicator label, does not search run lengths, and does not generate new branches. Up to eight CPUs are authorized when useful; the analysis-only implementation uses one numerical-library thread and retains catalytic matrix as the independent unit.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", analysis)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_parquet(
        LOOP_ROOT / "source_grounding_registry.parquet",
        source_grounding_registry(),
    )

    locked_inputs = {
        "l41BranchOutcome": L41_ROOT / "branch_outcome_results.parquet",
        "l42BranchRecovery": L42_ROOT / "branch_recovery_results.parquet",
        "l42PrefixBoundaries": L42_ROOT / "prefix_boundary_registry.parquet",
        "l42PrefixSummary": L42_ROOT / "prefix_state_summary.parquet",
        "l42ResponseRegistry": L42_ROOT / "response_registry.parquet",
        "l43BranchGain": L43_ROOT / "branch_gain_results.parquet",
        "l41ArtifactManifest": L41_ROOT / "artifact_manifest.json",
        "l42ArtifactManifest": L42_ROOT / "artifact_manifest.json",
        "l43ArtifactManifest": L43_ROOT / "artifact_manifest.json",
    }
    hashes = {name: sha256_file(path) for name, path in locked_inputs.items()}
    lock = {
        "schema": "eidosoma.e01.s19_l44.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "processes": list(PROCESSES),
        "orderedProcesses": ORDER_PROCESSES,
        "futureFissionHorizon": 12,
        "branchCountPerState": 128,
        "matrixBootstraps": BOOTSTRAPS,
        "selectionHierarchyForL45": [
            "NEW_HEREDITARY_EPISODE_RUN3",
            "INHERITANCE_RESUMPTION_RUN2",
            "INHERITANCE_BREAK_WITH_DEPARTURE",
            "NONE_IDENTIFIABILITY_ONLY",
        ],
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "completedTestTrajectoryUsed": False,
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
            "lockedInputHashes": hashes,
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
        raise RuntimeError("L44 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_inputs = {
        "l41BranchOutcome": L41_ROOT / "branch_outcome_results.parquet",
        "l42BranchRecovery": L42_ROOT / "branch_recovery_results.parquet",
        "l42PrefixBoundaries": L42_ROOT / "prefix_boundary_registry.parquet",
        "l42PrefixSummary": L42_ROOT / "prefix_state_summary.parquet",
        "l42ResponseRegistry": L42_ROOT / "response_registry.parquet",
        "l43BranchGain": L43_ROOT / "branch_gain_results.parquet",
        "l41ArtifactManifest": L41_ROOT / "artifact_manifest.json",
        "l42ArtifactManifest": L42_ROOT / "artifact_manifest.json",
        "l43ArtifactManifest": L43_ROOT / "artifact_manifest.json",
    }
    if any(
        sha256_file(path) != lock["lockedInputHashes"][name]
        for name, path in locked_inputs.items()
    ):
        raise RuntimeError("L44 locked input changed")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
    ):
        raise RuntimeError("L44 pre-execution validation failed")

    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    tables, classifications, next_theme, selected_process = compute_all()
    make_figures(
        tables["process_reliability_results.parquet"],
        tables["fixed_count_order_baseline_results.parquet"],
        tables["markov_group_results.parquet"],
        tables["dwell_hazard_results.parquet"],
        tables["process_relationship_matrix.parquet"],
        tables["plasticity_decomposition.parquet"],
        tables["scientific_gate_results.parquet"],
    )
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)

    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l44.classification.v1",
            "classifications": classifications,
            "selectedL45Process": selected_process,
            "selectedL45Theme": next_theme,
            "outcomeSearchPerformed": False,
            "targetUsesCompletedTestTrajectory": False,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=[
            "failureId",
            "stage",
            "status",
            "reason",
            "scientificValuesReleased",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)

    # Independent deterministic regeneration from the immutable L41-L43 tables.
    tables_again, classifications_again, next_again, selected_again = compute_all()
    exact = {
        name: frame_hash(frame) == frame_hash(tables_again[name])
        for name, frame in tables.items()
    }
    regeneration = {
        "schema": "eidosoma.e01.s19_l44.regeneration_validation.v1",
        "status": "PASS"
        if all(exact.values())
        and classifications == classifications_again
        and next_theme == next_again
        and selected_process == selected_again
        else "FAIL",
        "tableExact": exact,
        "classificationExact": classifications == classifications_again,
        "nextThemeExact": next_theme == next_again,
        "selectedProcessExact": selected_process == selected_again,
        "reusedBranchRows": len(tables_again["branch_episode_results.parquet"]),
    }
    if regeneration["status"] != "PASS":
        raise RuntimeError("L44 exact regeneration failure")

    runtime = {
        "schema": "eidosoma.e01.s19_l44.runtime.v1",
        "repositoryHead": lock["head"],
        "workersPermittedMaximum": 8,
        "workersUsedForScientificSimulation": 0,
        "analysisController": "SERIAL_VECTORIZED_ANALYSIS",
        "numericalLibraryThreads": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started,
        "controllerCpuHours": (time.process_time() - started_cpu) / 3600,
        "states": 280,
        "reusedBranchStreams": len(tables["branch_episode_results.parquet"]),
        "newBranchStreams": 0,
        "newMatrices": 0,
        "newTrajectories": 0,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    retained_bytes = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    )
    storage = {
        "schema": "eidosoma.e01.s19_l44.storage_validation.v1",
        "status": "PASS" if retained_bytes <= 15 * 1024**3 else "FAIL",
        "retainedBytes": retained_bytes,
        "retainedGiBCeiling": 15,
        "temporaryBytes": retained_bytes,
        "temporaryGiBCeiling": 30,
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        tables["process_reliability_results.parquet"],
        tables["fixed_count_order_baseline_results.parquet"],
        tables["markov_group_results.parquet"],
        tables["dwell_summary.parquet"],
        tables["plasticity_decomposition.parquet"],
        tables["scientific_gate_results.parquet"],
        classifications,
        runtime,
        next_theme,
        selected_process,
    )
    BASE.atomic_text(BUILD_ROOT / "S19_L44_FULL_RESULTS.md", report)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L44 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n"
        f"**Selected L45 process:** `{selected_process or 'NONE'}`.\n\n"
        f"**Next:** `{next_theme}`.\n",
    )
    for path in (BUILD_ROOT / "figures").glob("*.png"):
        if not path.stat().st_size:
            raise RuntimeError(f"empty L44 figure: {path}")
    if storage["status"] != "PASS":
        raise RuntimeError("L44 storage ceiling exceeded")

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
        raise RuntimeError("L44 artifact manifest regeneration failed")

    append_ledgers(
        classifications,
        runtime["completedAtUtc"],
        next_theme,
        selected_process,
    )
    root_report = (
        f"# S19 current-step report\n\nLatest completed loop: `{LOOP_ID}`.\n\n"
        f"Classification: {', '.join(classifications)}.\n\n"
        f"Selected L45 process: `{selected_process or 'NONE'}`.\n\n"
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
            "selectedDiscoveryLead": selected_process,
            "nextAuthorizedLoop": "S19-L45",
            "authorizationUpperBound": "S19-L65",
            "s20Active": False,
            "updatedAtUtc": runtime["completedAtUtc"],
        },
    )
    BASE.write_json(
        ARTIFACT_ROOT / "artifact_manifest.json", manifest_for(ARTIFACT_ROOT)
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "classifications": classifications,
                "selectedL45Process": selected_process,
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
