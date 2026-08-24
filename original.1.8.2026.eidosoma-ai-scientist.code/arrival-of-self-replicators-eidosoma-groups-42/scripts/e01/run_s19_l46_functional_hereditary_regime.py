#!/usr/bin/env python3
"""Run S19-L46 functional hereditary-regime transition audit."""

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
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

from e01_onset_discovery.empirical_committor import RestoredState
from e01_onset_discovery.functional_heredity_regime import (
    cosine,
    functional_profile,
    growth_signature,
    mean_pairwise_cosine,
    mean_pairwise_distance,
    simulate_functional_fission_clock,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L45 = load_module(
    "e01_l45_runner",
    ROOT / "scripts/e01/run_s19_l45_phi_incremental_hereditary_episode.py",
)
L44 = L45.L44
L43 = L44.L43
L42 = L43.L42
L41 = L42.L41
L28 = L41.L28
BASE = L45.BASE

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L46"
L41_ROOT = ARTIFACT_ROOT / "loops/L41"
L43_ROOT = ARTIFACT_ROOT / "loops/L43"
L44_ROOT = ARTIFACT_ROOT / "loops/L44"
L45_ROOT = ARTIFACT_ROOT / "loops/L45"
BUILD_ROOT = Path("/cache/e01_s19_l46/build")
CONFIG = ROOT / "configs/e01/s19_l46_functional_hereditary_regime.yaml"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = ROOT / "src/e01_onset_discovery/functional_heredity_regime.py"
LOOP_ID = "S19-L46"
VERSION = "E01-S19-L46-FUNCTIONAL-HEREDITARY-REGIME-TRANSITION-AUDIT-v1.0.0"
WORKERS = 8
BOOTSTRAPS = 4096
FAMILY = "F12"
HORIZON = 12
EVALUATION_COHORTS = ("L28_VALIDATION", "L31_CONFIRMATION")
PROFILE_DOMAINS = ("CATALYTIC_ACTIVATION", "EXPECTED_NET_EXCHANGE")
GROWTH_COLUMNS = (
    "logMolecularUpdates",
    "logGrossSampledEvents",
    "nonzeroReactionTypesPerUpdate",
    "daughterMassFraction",
)
PRIMARY_METRICS = (
    "CATALYTIC_ACTIVATION_OLD_MINUS_BREAK",
    "CATALYTIC_ACTIVATION_OLD_MINUS_PERMUTED",
    "CATALYTIC_ACTIVATION_OLD_MINUS_UNRELATED",
    "CATALYTIC_ACTIVATION_ORDERED_COHERENCE_EXCESS",
    "EXPECTED_NET_EXCHANGE_OLD_MINUS_BREAK",
    "EXPECTED_NET_EXCHANGE_OLD_MINUS_PERMUTED",
    "EXPECTED_NET_EXCHANGE_OLD_MINUS_UNRELATED",
    "EXPECTED_NET_EXCHANGE_ORDERED_COHERENCE_EXCESS",
    "GROWTH_OLD_CLOSENESS_EXCESS",
    "GROWTH_ORDERED_COHERENCE_EXCESS",
    "COMPOSITION_OLD_MINUS_BREAK",
)
SEED_ROOT = bytes.fromhex(
    "c447821733334bf91710dc4af1ac26ba826062798401ecebb122602220cd9f56"
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
    return hashlib.sha256(
        SEED_ROOT + b"\x00" + json.dumps(parts, separators=(",", ":")).encode()
    ).digest()


def derived_seed(*parts: object) -> int:
    return int.from_bytes(seed_material(*parts)[:16], "big")


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or len(np.unique(a[mask])) < 2 or len(np.unique(b[mask])) < 2:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(finite, [0.025, 0.975])))


def validate_immutable_prior() -> dict[str, Any]:
    prior = L45.validate_immutable_prior()
    rows: list[dict[str, Any]] = []
    for root in (L43_ROOT, L44_ROOT, L45_ROOT):
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
        "schema": "eidosoma.e01.s19_l46.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "priorThroughL44Unchanged": bool(prior["unchanged"]),
        "validatedArtifactCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
    }


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "GARD_SYSTEMS_PROTOBIOLOGY",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6073634/",
                "evidenceClass": "DIRECT_PUBLICATION",
                "finding": "GARD treats composition as inherited information and matrix-conditioned dynamic functionality as a rudimentary phenotype.",
                "frozenUse": "beta-conditioned activation and exchange profiles",
            },
            {
                "sourceId": "GARD_ATTRACTOR_DYNAMICS",
                "url": "https://doi.org/10.1016/j.xcrp.2023.101384",
                "evidenceClass": "DIRECT_PUBLICATION",
                "finding": "Compositionally reproducing GARD states are described as dynamic attractors with homeostatic growth.",
                "frozenUse": "separate old-state restoration from new-regime coherence",
            },
            {
                "sourceId": "COMPOSITIONAL_HEREDITY_ROBUSTNESS",
                "url": "https://arxiv.org/abs/2211.03155",
                "evidenceClass": "DIRECT_PUBLICATION",
                "finding": "Compositional heredity depends on growth/division dynamics and catalytic efficiency.",
                "frozenUse": "growth/division signature companion",
            },
            {
                "sourceId": "L43_L45_FROZEN_EVIDENCE",
                "url": None,
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Old composition does not recover; run-3 heredity is ordered beyond count; registered PhiID adds no held-out value.",
                "frozenUse": "functional-equivalence question without Phi or label search",
            },
        ]
    )


def fixture_results() -> pd.DataFrame:
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    beta = np.full((100, 100), 0.02, dtype=np.float64)
    restored = RestoredState(
        tuple(map(int, state)),
        "initial_selected_state",
        0,
        1,
        0,
        0,
    )
    definition = L28.definition("S12F-CANDIDATE-02")
    parts = ("fixture", "exact")
    canonical = L41.simulate_fission_clock(
        restored=restored,
        beta=beta,
        definition=definition,
        event_rng=np.random.Generator(np.random.PCG64DXSM(derived_seed(*parts, "event"))),
        trim_rng=np.random.Generator(np.random.PCG64DXSM(derived_seed(*parts, "trim"))),
        fission_rng=np.random.Generator(np.random.PCG64DXSM(derived_seed(*parts, "fission"))),
        daughter_rng=np.random.Generator(np.random.PCG64DXSM(derived_seed(*parts, "daughter"))),
        future_fissions=3,
    )
    audited = simulate_functional_fission_clock(
        restored=restored,
        beta=beta,
        definition=definition,
        event_rng=np.random.Generator(np.random.PCG64DXSM(derived_seed(*parts, "event"))),
        trim_rng=np.random.Generator(np.random.PCG64DXSM(derived_seed(*parts, "trim"))),
        fission_rng=np.random.Generator(np.random.PCG64DXSM(derived_seed(*parts, "fission"))),
        daughter_rng=np.random.Generator(np.random.PCG64DXSM(derived_seed(*parts, "daughter"))),
        future_fissions=3,
    )
    rng = np.random.default_rng(46)
    random_state = rng.integers(1, 5, size=100, dtype=np.int64)
    random_beta = np.exp(rng.normal(-4, 1, size=(100, 100)))
    permutation = rng.permutation(100)
    original = functional_profile(random_state, random_beta)
    permuted = functional_profile(
        random_state[permutation], random_beta[permutation][:, permutation]
    )
    vectors = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    rows = [
        {
            "fixtureId": "F01_EXACT_L41_PATH",
            "passed": audited.path_sha256 == canonical.path_sha256,
            "detail": audited.path_sha256,
        },
        {
            "fixtureId": "F02_EXACT_FUTURE_STATES",
            "passed": audited.future_states == canonical.future_states,
            "detail": str(len(audited.future_states)),
        },
        {
            "fixtureId": "F03_EXACT_PARENT_DAUGHTER_H",
            "passed": audited.parent_daughter_h == canonical.parent_daughter_h,
            "detail": str(audited.parent_daughter_h),
        },
        {
            "fixtureId": "F04_INTERVAL_ACCOUNTING",
            "passed": sum(row.molecular_updates for row in audited.intervals)
            == audited.molecular_updates,
            "detail": str(audited.molecular_updates),
        },
        {
            "fixtureId": "F05_ACTIVATION_PERMUTATION",
            "passed": np.allclose(
                permuted.catalytic_activation,
                original.catalytic_activation[permutation],
                atol=1e-12,
                rtol=1e-12,
            ),
            "detail": "joint state/beta feature permutation",
        },
        {
            "fixtureId": "F06_EXCHANGE_PERMUTATION",
            "passed": np.allclose(
                permuted.expected_net_exchange,
                original.expected_net_exchange[permutation],
                atol=1e-12,
                rtol=1e-12,
            ),
            "detail": "joint state/beta feature permutation",
        },
        {
            "fixtureId": "F07_PAIRWISE_COHERENCE",
            "passed": np.isclose(mean_pairwise_cosine(vectors), 1 / 3),
            "detail": str(mean_pairwise_cosine(vectors)),
        },
        {
            "fixtureId": "F08_EXACT_REPLAY",
            "passed": audited.final_state_sha256 == canonical.final_state_sha256
            and audited.selected_observations_generated
            == canonical.selected_observations_generated,
            "detail": audited.final_state_sha256,
        },
        {
            "fixtureId": "F09_SEED_DECIMAL_SERIALIZATION",
            "passed": str(derived_seed("fixture", "seed")).isdigit(),
            "detail": "128-bit analysis seeds are serialized losslessly as decimal strings",
        },
    ]
    return pd.DataFrame(rows)


def build_payloads() -> list[dict[str, Any]]:
    payloads = L42.build_payloads()
    l43 = pd.read_parquet(L43_ROOT / "branch_gain_results.parquet")
    l43 = l43[
        (l43["branchFamily"].eq(FAMILY))
        & (l43["targetId"].eq("PRIMARY_PREBREAK_DAUGHTER"))
    ][
        [
            "stateId",
            "branchIndex",
            "breakObserved",
            "breakBoundaryOneBased",
            "inheritedFlags",
            "pathSha256",
        ]
    ].rename(columns={"pathSha256": "pathSha256L43"})
    l44 = pd.read_parquet(L44_ROOT / "branch_episode_results.parquet")
    l44 = l44[
        [
            "stateId",
            "branchIndex",
            "newHereditaryEpisodeRun3",
            "run3CertificationOneBased",
            "inheritanceFlags",
        ]
    ]
    metadata = l43.merge(l44, on=["stateId", "branchIndex"], how="inner")
    if len(metadata) != 35_840 or not metadata["inheritedFlags"].eq(
        metadata["inheritanceFlags"]
    ).all():
        raise RuntimeError("L46 L43/L44 branch metadata mismatch")
    by_state = {
        state_id: {
            int(row.branchIndex): {
                "breakObserved": bool(row.breakObserved),
                "breakBoundaryOneBased": (
                    None
                    if pd.isna(row.breakBoundaryOneBased)
                    else int(row.breakBoundaryOneBased)
                ),
                "newHereditaryEpisodeRun3": bool(row.newHereditaryEpisodeRun3),
                "run3CertificationOneBased": (
                    None
                    if pd.isna(row.run3CertificationOneBased)
                    else int(row.run3CertificationOneBased)
                ),
                "inheritanceFlags": row.inheritanceFlags,
                "pathSha256": row.pathSha256L43,
            }
            for row in group.itertuples(index=False)
        }
        for state_id, group in metadata.groupby("stateId", sort=False)
    }
    output = []
    for payload in payloads:
        row = dict(payload)
        row["l46BranchMetadata"] = by_state[payload["stateId"]]
        output.append(row)
    if len(output) != 280 or any(len(row["l46BranchMetadata"]) != 128 for row in output):
        raise RuntimeError("L46 payload scope failure")
    return output


def analysis_seed_manifest() -> pd.DataFrame:
    rows = []
    for cohort in ("L28_DEVELOPMENT", *EVALUATION_COHORTS):
        for candidate in ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03"):
            for metric in PRIMARY_METRICS:
                for replicate in range(BOOTSTRAPS):
                    material = seed_material("bootstrap", cohort, candidate, metric, replicate)
                    rows.append(
                        {
                            "purpose": "MATRIX_BOOTSTRAP",
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "metricId": metric,
                            "replicate": replicate,
                            "derivedSeed": str(int.from_bytes(material[:16], "big")),
                            "seedMaterialSha256": hashlib.sha256(material).hexdigest(),
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame["derivedSeed"].duplicated().any() or frame["seedMaterialSha256"].duplicated().any():
        raise RuntimeError("L46 analysis seed collision")
    return frame


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L46/" in str(path):
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
        "schema": "eidosoma.e01.s19_l46.seed_firewall.v1",
        "status": "PASS" if not overlaps else "FAIL",
        "analysisSeedCount": len(seeds),
        "analysisSeedMaterialOverlapCount": len(overlaps),
        "reusedBranchStreamCount": 35_840,
        "newBranchStreamCount": 0,
    }


def benchmark_projection() -> dict[str, Any]:
    prior_runtime = json.loads((L41_ROOT / "runtime_manifest.json").read_text())
    projected_wall = float(prior_runtime["wallSeconds"]) * (2 * 35_840 / 53_760) * 1.5
    projected_cpu = projected_wall * WORKERS / 3600
    return {
        "schema": "eidosoma.e01.s19_l46.benchmark_projection.v1",
        "outcomeBlindHistoricalProjection": True,
        "sourceRuntimeManifest": str(L41_ROOT / "runtime_manifest.json"),
        "projectedFullReplayPasses": 2,
        "projectedWallHoursUpper": projected_wall / 3600,
        "projectedCpuHoursUpper": projected_cpu,
        "workers": WORKERS,
        "cpuHoursCeiling": 100,
        "wallHoursCeiling": 72,
        "status": "PASS" if projected_cpu < 85 and projected_wall < 72 * 3600 * 0.85 else "FAIL",
    }


def _profile_vectors(state: np.ndarray, beta: np.ndarray) -> dict[str, np.ndarray]:
    profile = functional_profile(state, beta)
    return {
        "CATALYTIC_ACTIVATION": profile.catalytic_activation,
        "EXPECTED_NET_EXCHANGE": profile.expected_net_exchange,
    }


def _worker(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    beta = L28.generate_beta(
        L28.derive_seed(
            L28.L23_ROOT_HEX,
            L28.L23_PHASE,
            "catalytic_matrix",
            int(payload["matrixIndex"]),
        )
    )
    if L28.simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError(f"L46 beta replay failure: {payload['stateId']}")
    restored = RestoredState(
        tuple(payload["state"]),
        payload["currentObservationKind"],
        int(payload["currentCompletedFissions"]),
        int(payload["currentGrowthGeneration"]),
        int(payload["currentGenerationLocalStep"]),
        int(payload["currentBatchStep"]),
    )
    prefix_latest = np.asarray(payload["prefixStates"][-1], dtype=np.int64)
    unrelated = np.asarray(payload["unrelatedPrefixStates"][-1], dtype=np.int64)
    permutation = np.asarray(payload["l42SpeciesPermutation"], dtype=np.int64)
    replay_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for branch in range(128):
        trace = simulate_functional_fission_clock(
            restored=restored,
            beta=beta,
            definition=L28.definition(payload["candidateId"]),
            event_rng=L41.generator(*L41._stream_parts(payload, FAMILY, branch, "event")),
            trim_rng=L41.generator(*L41._stream_parts(payload, FAMILY, branch, "trim")),
            fission_rng=L41.generator(*L41._stream_parts(payload, FAMILY, branch, "fission")),
            daughter_rng=L41.generator(*L41._stream_parts(payload, FAMILY, branch, "daughter")),
            future_fissions=HORIZON,
        )
        expected = payload["l42ExpectedBranches"][f"{FAMILY}:{branch}"]
        metadata = payload["l46BranchMetadata"][branch]
        exact = bool(
            trace.path_sha256 == expected["pathSha256"]
            and trace.path_sha256 == metadata["pathSha256"]
            and trace.final_state_sha256 == expected["finalStateSha256"]
            and trace.fissions == expected["fissions"]
            and trace.selected_observations_generated
            == expected["selectedObservationsGenerated"]
            and trace.terminal_status == expected["terminalStatus"]
        )
        if not exact:
            raise RuntimeError(
                f"L46 frozen path replay failure: {payload['stateId']} {branch}"
            )
        common = {
            "stateId": payload["stateId"],
            "evaluationCohort": payload["evaluationCohort"],
            "candidateId": payload["candidateId"],
            "matrixIndex": int(payload["matrixIndex"]),
            "landmark": int(payload["landmark"]),
            "branchIndex": branch,
            "branchHalf": "A" if branch < 64 else "B",
        }
        replay_rows.append(
            {
                **common,
                "expectedBranchIdentitySha256": expected["branchIdentitySha256"],
                "pathSha256": trace.path_sha256,
                "finalStateSha256": trace.final_state_sha256,
                "fissions": trace.fissions,
                "selectedObservationsGenerated": trace.selected_observations_generated,
                "molecularUpdates": trace.molecular_updates,
                "terminalStatus": trace.terminal_status,
                "exactL41Replay": exact,
                "intervalAccountingExact": sum(
                    interval.molecular_updates for interval in trace.intervals
                )
                == trace.molecular_updates,
            }
        )
        if not metadata["newHereditaryEpisodeRun3"]:
            continue
        if not metadata["breakObserved"]:
            raise RuntimeError("L46 run3 event without frozen break")
        break_boundary = int(metadata["breakBoundaryOneBased"])
        relative_certification = int(metadata["run3CertificationOneBased"])
        certification_boundary = break_boundary + relative_certification
        flags = np.asarray(json.loads(metadata["inheritanceFlags"]), dtype=np.bool_)
        parent_h = np.asarray(trace.parent_daughter_h, dtype=np.float64)
        if not np.array_equal(flags, parent_h[break_boundary:] > 0.9):
            raise RuntimeError("L46 frozen inheritance-flag mismatch")
        states = np.asarray(trace.future_states, dtype=np.int64)
        if certification_boundary > len(states) or relative_certification < 3:
            raise RuntimeError("L46 certification indexing failure")
        new_indices = np.arange(certification_boundary - 3, certification_boundary)
        inherited_indices = break_boundary + np.flatnonzero(flags)
        if len(inherited_indices) < 3 or not np.all(flags[relative_certification - 3 : relative_certification]):
            raise RuntimeError("L46 registered run3 window failure")
        old_anchor = prefix_latest if break_boundary == 1 else states[break_boundary - 2]
        break_state = states[break_boundary - 1]
        permuted_anchor = old_anchor[permutation]
        new_states = states[new_indices]
        inherited_states = states[inherited_indices]
        profiles = {
            "old": _profile_vectors(old_anchor, beta),
            "break": _profile_vectors(break_state, beta),
            "permuted": _profile_vectors(permuted_anchor, beta),
            "unrelated": _profile_vectors(unrelated, beta),
            "new": [_profile_vectors(state, beta) for state in new_states],
            "inherited": [
                _profile_vectors(state, beta) for state in inherited_states
            ],
        }
        old_relative = old_anchor / old_anchor.sum()
        break_relative = break_state / break_state.sum()
        permuted_relative = permuted_anchor / permuted_anchor.sum()
        unrelated_relative = unrelated / unrelated.sum()
        new_relative = np.asarray(
            [state / state.sum() for state in new_states], dtype=np.float64
        )
        new_relative_mean = new_relative.mean(axis=0)
        row: dict[str, Any] = {
            **common,
            "breakBoundaryOneBased": break_boundary,
            "run3CertificationRelativeOneBased": relative_certification,
            "run3CertificationBoundaryOneBased": certification_boundary,
            "postbreakOpportunities": len(flags),
            "inheritedPostbreakCount": int(flags.sum()),
            "oldAnchorMass": int(old_anchor.sum()),
            "breakStateMass": int(break_state.sum()),
            "oldNewCompositionH": cosine(old_relative, new_relative_mean),
            "breakNewCompositionH": cosine(break_relative, new_relative_mean),
            "permutedNewCompositionH": cosine(permuted_relative, new_relative_mean),
            "unrelatedNewCompositionH": cosine(unrelated_relative, new_relative_mean),
            "newTripleCompositionCoherence": mean_pairwise_cosine(new_relative),
            "allInheritedCompositionCoherence": mean_pairwise_cosine(
                np.asarray(
                    [state / state.sum() for state in inherited_states],
                    dtype=np.float64,
                )
            ),
            "oldGrowthComplete": bool(trace.intervals[break_boundary - 1].complete_growth_interval),
            "oldGrowthSignature": json.dumps(
                growth_signature(trace.intervals[break_boundary - 1]).tolist(),
                separators=(",", ":"),
            ),
            "newGrowthSignatures": json.dumps(
                [growth_signature(trace.intervals[index]).tolist() for index in new_indices],
                separators=(",", ":"),
            ),
            "allInheritedGrowthSignatures": json.dumps(
                [
                    growth_signature(trace.intervals[index]).tolist()
                    for index in inherited_indices
                ],
                separators=(",", ":"),
            ),
            "pathSha256": trace.path_sha256,
            "targetUsesCompletedTestTrajectory": False,
        }
        for domain in PROFILE_DOMAINS:
            new_vectors = np.asarray(
                [profile[domain] for profile in profiles["new"]], dtype=np.float64
            )
            inherited_vectors = np.asarray(
                [profile[domain] for profile in profiles["inherited"]],
                dtype=np.float64,
            )
            new_mean = new_vectors.mean(axis=0)
            stem = (
                "activation" if domain == "CATALYTIC_ACTIVATION" else "netExchange"
            )
            row[f"{stem}OldNew"] = cosine(profiles["old"][domain], new_mean)
            row[f"{stem}BreakNew"] = cosine(profiles["break"][domain], new_mean)
            row[f"{stem}PermutedNew"] = cosine(
                profiles["permuted"][domain], new_mean
            )
            row[f"{stem}UnrelatedNew"] = cosine(
                profiles["unrelated"][domain], new_mean
            )
            row[f"{stem}NewTripleCoherence"] = mean_pairwise_cosine(new_vectors)
            row[f"{stem}AllInheritedCoherence"] = mean_pairwise_cosine(
                inherited_vectors
            )
        episode_rows.append(row)
    return {"replay": replay_rows, "episodes": episode_rows}


def execute_paths(payloads: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    replay_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            replay_rows.extend(result["replay"])
            episode_rows.extend(result["episodes"])
    keys = ["candidateId", "matrixIndex", "landmark", "branchIndex"]
    replay = pd.DataFrame(replay_rows).sort_values(keys).reset_index(drop=True)
    episodes = pd.DataFrame(episode_rows).sort_values(keys).reset_index(drop=True)
    if (
        len(replay) != 35_840
        or not replay["exactL41Replay"].all()
        or not replay["intervalAccountingExact"].all()
        or len(episodes) != int(
            pd.read_parquet(L44_ROOT / "branch_episode_results.parquet")[
                "newHereditaryEpisodeRun3"
            ].sum()
        )
    ):
        raise RuntimeError("L46 replay/episode scope failure")
    return replay, episodes


def growth_scale_registry(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    development = episodes[episodes["evaluationCohort"].eq("L28_DEVELOPMENT")]
    for candidate, group in development.groupby("candidateId", sort=False):
        vectors: list[list[float]] = []
        for row in group.itertuples(index=False):
            vectors.append(json.loads(row.oldGrowthSignature))
            vectors.extend(json.loads(row.newGrowthSignatures))
            vectors.extend(json.loads(row.allInheritedGrowthSignatures))
        values = np.asarray(vectors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(GROWTH_COLUMNS):
            raise RuntimeError("L46 growth-scale source shape failure")
        median = np.median(values, axis=0)
        raw_mad = np.median(np.abs(values - median), axis=0)
        scale = 1.4826 * raw_mad
        replaced = scale <= 0
        scale[replaced] = 1.0
        for index, name in enumerate(GROWTH_COLUMNS):
            rows.append(
                {
                    "candidateId": candidate,
                    "featureId": name,
                    "developmentValueCount": len(values),
                    "median": float(median[index]),
                    "rawMAD": float(raw_mad[index]),
                    "scaledMAD": float(scale[index]),
                    "zeroScaleReplacedByOne": bool(replaced[index]),
                    "fitCohort": "L28_DEVELOPMENT",
                }
            )
    return pd.DataFrame(rows).sort_values(["candidateId", "featureId"]).reset_index(drop=True)


def add_growth_results(
    episodes: pd.DataFrame, scales: pd.DataFrame
) -> pd.DataFrame:
    scale_map = {
        candidate: group.set_index("featureId")
        .reindex(GROWTH_COLUMNS)["scaledMAD"]
        .to_numpy(dtype=np.float64)
        for candidate, group in scales.groupby("candidateId", sort=False)
    }
    rows = []
    for row in episodes.itertuples(index=False):
        old = np.asarray(json.loads(row.oldGrowthSignature), dtype=np.float64)
        new = np.asarray(json.loads(row.newGrowthSignatures), dtype=np.float64)
        inherited = np.asarray(
            json.loads(row.allInheritedGrowthSignatures), dtype=np.float64
        )
        scale = scale_map[row.candidateId]
        if row.oldGrowthComplete:
            old_new = float(
                np.mean(
                    [
                        np.linalg.norm((value - old) / scale) / np.sqrt(len(scale))
                        for value in new
                    ]
                )
            )
            old_all = float(
                np.mean(
                    [
                        np.linalg.norm((value - old) / scale) / np.sqrt(len(scale))
                        for value in inherited
                    ]
                )
            )
        else:
            old_new = old_all = float("nan")
        new_dispersion = mean_pairwise_distance(new, scale)
        all_dispersion = mean_pairwise_distance(inherited, scale)
        rows.append(
            {
                "stateId": row.stateId,
                "evaluationCohort": row.evaluationCohort,
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "landmark": int(row.landmark),
                "branchIndex": int(row.branchIndex),
                "branchHalf": row.branchHalf,
                "oldGrowthComplete": bool(row.oldGrowthComplete),
                "oldToNewGrowthDistance": old_new,
                "oldToAllInheritedGrowthDistance": old_all,
                "oldGrowthClosenessExcess": old_all - old_new,
                "newTripleGrowthDispersion": new_dispersion,
                "allInheritedGrowthDispersion": all_dispersion,
                "orderedGrowthCoherenceExcess": all_dispersion - new_dispersion,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["candidateId", "matrixIndex", "landmark", "branchIndex"]
    ).reset_index(drop=True)


def metric_results(episodes: pd.DataFrame, growth: pd.DataFrame) -> pd.DataFrame:
    merged = episodes.merge(
        growth,
        on=[
            "stateId",
            "evaluationCohort",
            "candidateId",
            "matrixIndex",
            "landmark",
            "branchIndex",
            "branchHalf",
            "oldGrowthComplete",
        ],
        validate="one_to_one",
    )
    definitions = {
        "CATALYTIC_ACTIVATION_OLD_MINUS_BREAK": merged["activationOldNew"]
        - merged["activationBreakNew"],
        "CATALYTIC_ACTIVATION_OLD_MINUS_PERMUTED": merged["activationOldNew"]
        - merged["activationPermutedNew"],
        "CATALYTIC_ACTIVATION_OLD_MINUS_UNRELATED": merged["activationOldNew"]
        - merged["activationUnrelatedNew"],
        "CATALYTIC_ACTIVATION_ORDERED_COHERENCE_EXCESS": merged[
            "activationNewTripleCoherence"
        ]
        - merged["activationAllInheritedCoherence"],
        "EXPECTED_NET_EXCHANGE_OLD_MINUS_BREAK": merged["netExchangeOldNew"]
        - merged["netExchangeBreakNew"],
        "EXPECTED_NET_EXCHANGE_OLD_MINUS_PERMUTED": merged["netExchangeOldNew"]
        - merged["netExchangePermutedNew"],
        "EXPECTED_NET_EXCHANGE_OLD_MINUS_UNRELATED": merged["netExchangeOldNew"]
        - merged["netExchangeUnrelatedNew"],
        "EXPECTED_NET_EXCHANGE_ORDERED_COHERENCE_EXCESS": merged[
            "netExchangeNewTripleCoherence"
        ]
        - merged["netExchangeAllInheritedCoherence"],
        "GROWTH_OLD_CLOSENESS_EXCESS": merged["oldGrowthClosenessExcess"],
        "GROWTH_ORDERED_COHERENCE_EXCESS": merged[
            "orderedGrowthCoherenceExcess"
        ],
        "COMPOSITION_OLD_MINUS_BREAK": merged["oldNewCompositionH"]
        - merged["breakNewCompositionH"],
    }
    id_columns = [
        "stateId",
        "evaluationCohort",
        "candidateId",
        "matrixIndex",
        "landmark",
        "branchIndex",
        "branchHalf",
    ]
    frames = []
    for metric, values in definitions.items():
        frame = merged[id_columns].copy()
        frame["metricId"] = metric
        frame["value"] = np.asarray(values, dtype=np.float64)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["candidateId", "matrixIndex", "landmark", "branchIndex", "metricId"]
    ).reset_index(drop=True)


def state_metric_results(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "stateId",
        "evaluationCohort",
        "candidateId",
        "matrixIndex",
        "landmark",
        "metricId",
    ]
    rows = []
    for values, group in metrics.groupby(keys, sort=False):
        halves = group.groupby("branchHalf")["value"].mean()
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "eligibleBranches": int(group["value"].notna().sum()),
                "meanValue": float(group["value"].mean()),
                "meanHalfA": float(halves.get("A", np.nan)),
                "meanHalfB": float(halves.get("B", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def aggregate_metrics(
    metrics: pd.DataFrame, states: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = (
        metrics.groupby(
            ["evaluationCohort", "candidateId", "matrixIndex", "metricId"],
            as_index=False,
        )["value"]
        .mean()
        .rename(columns={"value": "matrixMeanValue"})
    )
    bootstrap_rows = []
    group_rows = []
    reliability_rows = []
    group_keys = ["evaluationCohort", "candidateId", "metricId"]
    for keys, group in matrix.groupby(group_keys, sort=False):
        values = group["matrixMeanValue"].dropna().to_numpy(dtype=np.float64)
        replicates = np.empty(BOOTSTRAPS, dtype=np.float64)
        for replicate in range(BOOTSTRAPS):
            rng = np.random.Generator(
                np.random.PCG64DXSM(derived_seed("bootstrap", *keys, replicate))
            )
            replicates[replicate] = float(
                np.mean(values[rng.integers(0, len(values), size=len(values))])
            )
            bootstrap_rows.append(
                {
                    **dict(zip(group_keys, keys, strict=True)),
                    "replicate": replicate,
                    "meanValue": replicates[replicate],
                }
            )
        low, high = interval(replicates)
        group_rows.append(
            {
                **dict(zip(group_keys, keys, strict=True)),
                "matrixCount": len(values),
                "meanValue": float(np.mean(values)),
                "medianValue": float(np.median(values)),
                "lower95": low,
                "upper95": high,
                "positiveDirection": bool(low > 0),
            }
        )
    for keys, group in states.groupby(group_keys, sort=False):
        reliability_rows.append(
            {
                **dict(zip(group_keys, keys, strict=True)),
                "stateCount": len(group),
                "splitHalfSpearman": safe_spearman(
                    group["meanHalfA"].to_numpy(float),
                    group["meanHalfB"].to_numpy(float),
                ),
            }
        )
    return (
        pd.DataFrame(group_rows).sort_values(group_keys).reset_index(drop=True),
        pd.DataFrame(bootstrap_rows).sort_values([*group_keys, "replicate"]).reset_index(drop=True),
        pd.DataFrame(reliability_rows).sort_values(group_keys).reset_index(drop=True),
    )


def scientific_gates(
    groups: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    evaluation = groups[groups["evaluationCohort"].isin(EVALUATION_COHORTS)]
    restoration_metrics = (
        "CATALYTIC_ACTIVATION_OLD_MINUS_BREAK",
        "CATALYTIC_ACTIVATION_OLD_MINUS_PERMUTED",
        "CATALYTIC_ACTIVATION_OLD_MINUS_UNRELATED",
        "EXPECTED_NET_EXCHANGE_OLD_MINUS_BREAK",
        "EXPECTED_NET_EXCHANGE_OLD_MINUS_PERMUTED",
        "EXPECTED_NET_EXCHANGE_OLD_MINUS_UNRELATED",
        "GROWTH_OLD_CLOSENESS_EXCESS",
    )
    coherence_metrics = (
        "CATALYTIC_ACTIVATION_ORDERED_COHERENCE_EXCESS",
        "EXPECTED_NET_EXCHANGE_ORDERED_COHERENCE_EXCESS",
        "GROWTH_ORDERED_COHERENCE_EXCESS",
    )
    rows = []
    for metric in (*restoration_metrics, *coherence_metrics):
        subset = evaluation[evaluation["metricId"].eq(metric)]
        rows.append(
            {
                "gateId": f"ALL_EVALUATION_GROUPS_POSITIVE::{metric}",
                "metricId": metric,
                "requiredGroupCount": 4,
                "observedGroupCount": len(subset),
                "passed": len(subset) == 4 and bool(subset["lower95"].gt(0).all()),
                "criterion": "matrix-bootstrap lower 95% bound above zero in both candidates and both held-out cohorts",
            }
        )
    gate_frame = pd.DataFrame(rows)
    old_pass = bool(
        gate_frame[gate_frame["metricId"].isin(restoration_metrics)]["passed"].all()
    )
    coherence_pass = bool(
        gate_frame[gate_frame["metricId"].isin(coherence_metrics)]["passed"].all()
    )
    domain_passes = int(gate_frame["passed"].sum())
    classifications = [
        (
            "OLD_FUNCTIONAL_REGIME_RESTORATION_DIRECTIONALLY_SUPPORTED"
            if old_pass
            else "OLD_FUNCTIONAL_REGIME_RESTORATION_NOT_SUPPORTED"
        ),
        (
            "NEW_LOCAL_FUNCTIONAL_REGIME_COHERENCE_SUPPORTED"
            if coherence_pass
            else "ORDERED_FUNCTIONAL_COHERENCE_NOT_SUPPORTED"
        ),
    ]
    if 0 < domain_passes < len(gate_frame):
        classifications.append("DOMAIN_SPECIFIC_FUNCTIONAL_EFFECTS")
    classifications.append("NOT_PROMOTABLE_AS_CONFIRMED")
    if old_pass:
        next_theme = "L47_FUNCTIONAL_EQUIVALENCE_TRANSFER_AUDIT"
    elif coherence_pass:
        next_theme = "L47_FUNCTIONAL_REGIME_TRANSITION_PATHWAY_HETEROGENEITY"
    else:
        next_theme = "L47_SHOOTING_MEASUREMENT_AND_TARGET_IDENTIFIABILITY_AUDIT"
    return gate_frame, classifications, next_theme


def candidate_comparison(groups: pd.DataFrame) -> pd.DataFrame:
    evaluation = groups[groups["evaluationCohort"].isin(EVALUATION_COHORTS)]
    pivot = evaluation.pivot_table(
        index=["evaluationCohort", "metricId"],
        columns="candidateId",
        values="meanValue",
    ).reset_index()
    candidates = ["S12F-CANDIDATE-02", "S12F-CANDIDATE-03"]
    if all(candidate in pivot for candidate in candidates):
        pivot["candidateDifferenceC02MinusC03"] = pivot[candidates[0]] - pivot[candidates[1]]
        pivot["candidateDirectionAgreement"] = (
            np.sign(pivot[candidates[0]]) == np.sign(pivot[candidates[1]])
        )
    return pivot


def make_figures(
    episodes: pd.DataFrame,
    groups: pd.DataFrame,
    reliability: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    figure_root = BUILD_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    evaluation = episodes[episodes["evaluationCohort"].isin(EVALUATION_COHORTS)]
    labels = [
        f"{cohort.split('_')[-1][:4]}-C{candidate[-2:]}"
        for cohort in EVALUATION_COHORTS
        for candidate in ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
    ]
    ordered_groups = [
        evaluation[
            evaluation["evaluationCohort"].eq(cohort)
            & evaluation["candidateId"].eq(candidate)
        ]
        for cohort in EVALUATION_COHORTS
        for candidate in ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, [g["oldNewCompositionH"].mean() for g in ordered_groups], 0.36, label="old→new composition H")
    ax.bar(x + 0.18, [g["breakNewCompositionH"].mean() for g in ordered_groups], 0.36, label="break→new composition H")
    ax.axhline(0.9, color="black", ls="--", label="strict H threshold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cosine H")
    ax.set_title("The new hereditary episode does not silently restore old composition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "01_compositional_old_vs_new.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, stem, title in (
        (axes[0], "activation", "Catalytic activation"),
        (axes[1], "netExchange", "Expected net exchange"),
    ):
        actual = [g[f"{stem}OldNew"].mean() for g in ordered_groups]
        broken = [g[f"{stem}BreakNew"].mean() for g in ordered_groups]
        permuted = [g[f"{stem}PermutedNew"].mean() for g in ordered_groups]
        width = 0.25
        axis.bar(x - width, actual, width, label="old anchor")
        axis.bar(x, broken, width, label="break state")
        axis.bar(x + width, permuted, width, label="permuted old")
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_ylabel("Cosine similarity to new episode")
        axis.set_title(title)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "02_functional_restoration_controls.png", dpi=160)
    plt.close(fig)

    selected = groups[
        groups["evaluationCohort"].isin(EVALUATION_COHORTS)
        & groups["metricId"].str.contains("COHERENCE_EXCESS")
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [
        f"{r.evaluationCohort.split('_')[-1][:4]}-C{r.candidateId[-2:]}\n{r.metricId.split('_')[0]}"
        for r in selected.itertuples(index=False)
    ]
    xx = np.arange(len(selected))
    ax.bar(xx, selected["meanValue"], color="#4c78a8")
    ax.vlines(xx, selected["lower95"], selected["upper95"], color="black")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(xx, names, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Consecutive triple minus fixed-count control")
    ax.set_title("Ordered functional coherence of the new hereditary episode")
    fig.tight_layout()
    fig.savefig(figure_root / "03_ordered_functional_coherence.png", dpi=160)
    plt.close(fig)

    restoration = groups[
        groups["evaluationCohort"].isin(EVALUATION_COHORTS)
        & groups["metricId"].str.contains("OLD_MINUS|OLD_CLOSENESS")
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    xx = np.arange(len(restoration))
    ax.bar(xx, restoration["meanValue"], color="#f58518")
    ax.vlines(xx, restoration["lower95"], restoration["upper95"], color="black", lw=0.7)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(xx, restoration["metricId"], rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("Paired restoration effect")
    ax.set_title("Old-regime restoration gates")
    fig.tight_layout()
    fig.savefig(figure_root / "04_old_regime_restoration.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    rel = reliability[reliability["evaluationCohort"].isin(EVALUATION_COHORTS)]
    ax.scatter(np.arange(len(rel)), rel["splitHalfSpearman"], color="#54a24b")
    ax.axhline(0.5, color="grey", ls="--")
    ax.set_xticks(np.arange(len(rel)), rel["metricId"], rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("Split-half state-rank Spearman")
    ax.set_title("State-level reliability of functional effects")
    fig.tight_layout()
    fig.savefig(figure_root / "05_split_half_reliability.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    matrix = gates.set_index("gateId")[["passed"]].astype(int)
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks([0], ["pass"])
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=6)
    ax.set_title("L46 preregistered functional-regime gate matrix")
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(figure_root / "06_decision_matrix.png", dpi=160)
    plt.close(fig)


def report_text(
    groups: pd.DataFrame,
    reliability: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    evaluation = groups[groups["evaluationCohort"].isin(EVALUATION_COHORTS)].copy()
    summary = evaluation[
        ["evaluationCohort", "candidateId", "metricId", "meanValue", "lower95", "upper95"]
    ]
    return f"""# S19-L46 Full Results — Functional Hereditary-Regime Transition Audit

## Top summary

- **Research step:** `{VERSION}`
- **Completion status:** complete; additive exploratory evidence
- **Validation:** PASS — immutable prior, frozen path identity, interval accounting, two independent full replays, matrix bootstraps, regeneration, storage and artifact hashes
- **Outcome classification:** {', '.join(f'`{value}`' for value in classifications)}
- **Lay summary:** The loop asked whether the first post-break run of three inherited fissions restores the old *function* even though it does not restore the old composition, or instead forms a coherent new functional regime. Functional profiles were derived directly from the frozen catalytic matrix and GARD rate equations; no new branch, label, threshold, Phi calculation or intervention was introduced.
- **Recommended next action:** `{next_theme}` under the existing human-authorized sequence; S20 and E02 remain inactive.

## Frozen question and evidence boundary

L43 showed that similarity to the old pre-break composition falls rather than recovers. L44 nevertheless established a modest but reproducible temporal-order excess for a run of three inherited fissions. L45 found no registered PhiID increment beyond direct heredity controls. L46 therefore tests three prospectively locked functional objects on exactly the L41 F12 branch streams: `beta @ relative_composition`, the GARD expected join-minus-loss vector, and a growth/division signature comprising update burden, event burden, active reaction types and daughter retention.

The process target remains the L44 online-certified `NEW_HEREDITARY_EPISODE_RUN3`. It uses no completed trajectory. A positive functional result would remain exploratory and would not establish author code, paper replication, prospective Phi signal, intervention efficacy or causal control.

## Methods

- Exact replay: 35,840 frozen F12 paths, twice, using 8 workers and one numerical-library thread per worker.
- Functional restoration: similarity of the pre-break anchor to the mean certification-window phenotype, compared with the break daughter, a frozen species-permuted old anchor and an unrelated-matrix prefix anchor.
- Ordered coherence: pairwise coherence inside the first consecutive inherited triple versus all inherited post-break states, which preserves inheritance count but removes adjacency selection.
- Growth/division: development-candidate median and 1.4826-MAD scaling only; zero scales were fixed to one before evaluation.
- Statistical unit: catalytic matrix. All registered intervals use 4,096 domain-separated matrix bootstraps. Candidates and cohorts remain separate.

## Primary results

{summary.to_markdown(index=False, floatfmt='.6f')}

## Gate results

{gates.to_markdown(index=False)}

## State-level reliability

{reliability[reliability['evaluationCohort'].isin(EVALUATION_COHORTS)].to_markdown(index=False, floatfmt='.5f')}

## Interpretation

Old-state restoration and new-regime coherence are different claims. Similarity to a matrix-conditioned rate profile can be high for many compositions, so the molecule-permuted, unrelated and break-state controls are mandatory. Likewise, three adjacent inherited fissions are compositionally smooth by definition; only excess coherence over the fixed inherited-state set is treated as temporal evidence. Domain-specific passage cannot rescue a failed complete functional-regime contract.

## Validation and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- Wall time: `{runtime['wallSeconds'] / 3600:.3f}` hours.
- Reused branch streams: `{runtime['reusedBranchStreams']}`; new streams: `0`.
- Workers: `{runtime['workers']}`; GPU hours: `0`.
- Custom code: `{CORE_PATH.relative_to(ROOT)}` and `{RUNNER_PATH.relative_to(ROOT)}`.
- Dependencies: existing pinned E01 simulator plus NumPy, pandas, SciPy, PyArrow and Matplotlib in the workspace runtime.

## Limitations

The catalytic and exchange profiles are deterministic functions of current composition and beta; they are functional proxies, not experimentally measured biochemical phenotypes. The growth signature covers only the reconstructed GARD simulator and the frozen 12-fission horizon. Repeated adaptive loops reduce confirmatory credibility. Every result remains an additive exploratory overlay and changes no S18 or earlier S19 classification.
"""


def input_scope_registry(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "stateId": payload["stateId"],
            "evaluationCohort": payload["evaluationCohort"],
            "candidateId": payload["candidateId"],
            "matrixIndex": int(payload["matrixIndex"]),
            "landmark": int(payload["landmark"]),
            "betaSha256": payload["betaSha256"],
            "currentStateSha256": L28.simulator_array_sha256(
                np.asarray(payload["state"], dtype=np.int64)
            ),
            "frozenF12Branches": len(payload["l46BranchMetadata"]),
        }
        for payload in payloads
    ]
    return pd.DataFrame(rows).sort_values(
        ["candidateId", "matrixIndex", "landmark"]
    ).reset_index(drop=True)


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L46 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L46 local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    payloads = build_payloads()
    scope = input_scope_registry(payloads)
    seeds = analysis_seed_manifest()
    firewall = seed_firewall(seeds)
    benchmark = benchmark_projection()
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or len(scope) != 280
        or firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
    ):
        raise RuntimeError("L46 preoutcome validation failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L46 decision record\n\n"
        "The autonomous human authorization extends bounded S19 loops through L65 and permits up to eight CPUs when useful. L43-L45 separate plastic heredity from old-composition recovery and find no registered PhiID increment. Before opening an L46 functional cohort result, this record freezes one nonduplicative audit: exact replay of only the 35,840 L41 F12 streams, the unchanged L44 genuine-break and run-3 process, two source-equation vector phenotypes (`beta @ x` and expected join-minus-loss), one four-component growth/division signature, old/break/permuted/unrelated controls, a fixed inherited-state order control, development-only robust scaling, candidate/cohort separation and 4,096 catalytic-matrix bootstraps. No label, H threshold, horizon, branch, simulator, feature family, model, Phi scalar or intervention is searched.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "input_scope_registry.parquet", scope)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_parquet(LOOP_ROOT / "source_registry.parquet", source_registry())
    source_snapshot = {
        "schema": "eidosoma.e01.s19_l46.source_snapshot_manifest.v1",
        "simulatorCoreSha256": sha256_file(ROOT / "src/e01_latent_timebase/core.py"),
        "l41FissionClockCoreSha256": sha256_file(
            ROOT / "src/e01_onset_discovery/fission_clock_recurrence.py"
        ),
        "l44ProcessCoreSha256": sha256_file(
            ROOT / "src/e01_onset_discovery/heredity_process_family.py"
        ),
        "l46FunctionalCoreSha256": sha256_file(CORE_PATH),
        "sources": source_registry().to_dict("records"),
    }
    BASE.write_json(LOOP_ROOT / "source_snapshot_manifest.json", source_snapshot)
    locked_inputs = {
        "scopeRegistry": LOOP_ROOT / "input_scope_registry.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "l41BranchTrace": L41_ROOT / "branch_trace_results.parquet",
        "l43BranchGain": L43_ROOT / "branch_gain_results.parquet",
        "l44BranchEpisodes": L44_ROOT / "branch_episode_results.parquet",
        "l45ArtifactManifest": L45_ROOT / "artifact_manifest.json",
    }
    hashes = {name: sha256_file(path) for name, path in locked_inputs.items()}
    lock = {
        "schema": "eidosoma.e01.s19_l46.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "frozenBranchFamily": FAMILY,
        "frozenFissionHorizon": HORIZON,
        "frozenProcess": "NEW_HEREDITARY_EPISODE_RUN3",
        "functionalDomains": list(PROFILE_DOMAINS),
        "growthSignatureColumns": list(GROWTH_COLUMNS),
        "growthScale": "development-candidate median and 1.4826*MAD; zero replaced by 1",
        "matrixBootstraps": BOOTSTRAPS,
        "workers": WORKERS,
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
            "lockedInputHashes": hashes,
        },
    )


def compute_tables(episodes: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[str], str]:
    scales = growth_scale_registry(episodes)
    growth = add_growth_results(episodes, scales)
    metrics = metric_results(episodes, growth)
    states = state_metric_results(metrics)
    groups, bootstrap, reliability = aggregate_metrics(metrics, states)
    gates, classifications, next_theme = scientific_gates(groups)
    tables = {
        "growth_scale_registry.parquet": scales,
        "growth_division_results.parquet": growth,
        "functional_metric_results.parquet": metrics,
        "state_function_results.parquet": states,
        "group_function_results.parquet": groups,
        "bootstrap_results.parquet": bootstrap,
        "split_half_reliability.parquet": reliability,
        "scientific_gate_results.parquet": gates,
        "candidate_comparison.parquet": candidate_comparison(groups),
        "negative_control_results.parquet": groups[
            groups["metricId"].str.contains("PERMUTED|UNRELATED|BREAK")
        ].reset_index(drop=True),
    }
    return tables, classifications, next_theme


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
        "schema": "eidosoma.e01.s19_l46.artifact_manifest.v1",
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
            "beliefBeforeLoop": "L43-L45 imply plastic compositional heredity with modest episode ordering, no old-composition recovery and no registered PhiID increment.",
            "failureOrAmbiguityTargeted": "Whether a new hereditary episode preserves matrix-conditioned catalytic or growth/division function despite molecular identity change.",
            "informationGainRationale": "Exact frozen-path replay separates composition, source-rate function and growth/division behavior without creating a new target or branch.",
            "learned": "L46 functional domains and controls locked before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L43 compositional divergence; L44 ordered run-3 process; L45 PhiID non-increment; reviewer functional-equivalence framing.",
            "proposedNextTest": "Audit old functional restoration and coherent new-regime formation on frozen F12 paths.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Old functional restoration, coherent new functional regime, domain-specific preservation, or no functional organization beyond local inheritance.",
            "selectedHypotheses": "Matrix-conditioned function may be more stable than exact composition.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "One fixed molecular centroid is the only scientifically meaningful organization target.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A functional claim requires paired improvement over break, molecule-permuted, unrelated and fixed-count inherited-state controls in both candidates and cohorts.",
            "failureOrAmbiguityTargeted": "Functional-regime restoration versus local regime switching.",
            "informationGainRationale": "Two independent exact replays and matrix bootstraps make null and domain-specific outcomes informative.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L46 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Functional phenotype of frozen L44 new hereditary episode.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "High unadjusted functional similarity alone demonstrates homeostatic memory.",
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
        + f"\n\n## {LOOP_ID} — functional hereditary-regime transition\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** `{next_theme}`.\n",
    )

    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidate = {
        "branchCount": 3,
        "bundleId": "L46_FUNCTIONAL_HEREDITARY_REGIME",
        "candidateId": "S19-L46-FUNCTIONAL-HEREDITARY-REGIME",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 4,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 1,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 4,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 1,
        "proposedSpecification": "source-rate catalytic activation, expected exchange and growth/division function around the frozen post-break run-3 episode",
        "rankingScore": 28.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": "OLD_FUNCTIONAL_REGIME_RESTORATION_DIRECTIONALLY_SUPPORTED"
        in classifications
        or "NEW_LOCAL_FUNCTIONAL_REGIME_COHERENCE_SUPPORTED" in classifications,
        "selectionReason": "L43_L45_PLASTIC_HEREDITY_AND_REVIEWER_FUNCTIONAL_EQUIVALENCE",
        "sourceGrounding": 5,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidate_path,
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
    additions = []
    for row in source_registry().itertuples(index=False):
        additions.append(
            {
                "commitOrVersion": None,
                "evidenceClass": row.evidenceClass,
                "finding": f"{row.finding}; L46 use: {row.frozenUse}",
                "licenseStatus": "PUBLIC_METADATA_OR_WORKSPACE_EVIDENCE",
                "redistributionStatus": "REFERENCE_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L46_{row.sourceId}",
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


def execute() -> None:
    started = time.perf_counter()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    if (
        git("rev-parse", "HEAD") != lock["head"]
        or git("rev-parse", "origin/eidosoma/groups/42") != lock["remote"]
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("L46 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_inputs = {
        "scopeRegistry": LOOP_ROOT / "input_scope_registry.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "l41BranchTrace": L41_ROOT / "branch_trace_results.parquet",
        "l43BranchGain": L43_ROOT / "branch_gain_results.parquet",
        "l44BranchEpisodes": L44_ROOT / "branch_episode_results.parquet",
        "l45ArtifactManifest": L45_ROOT / "artifact_manifest.json",
    }
    if any(
        sha256_file(path) != lock["lockedInputHashes"][name]
        for name, path in locked_inputs.items()
    ):
        raise RuntimeError("L46 locked input changed")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
    ):
        raise RuntimeError("L46 pre-execution validation failed")
    payloads = build_payloads()
    if frame_hash(input_scope_registry(payloads)) != frame_hash(
        pd.read_parquet(LOOP_ROOT / "input_scope_registry.parquet")
    ):
        raise RuntimeError("L46 input scope regeneration mismatch")
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)

    replay, episodes = execute_paths(payloads)
    tables, classifications, next_theme = compute_tables(episodes)
    make_figures(
        episodes,
        tables["group_function_results.parquet"],
        tables["split_half_reliability.parquet"],
        tables["scientific_gate_results.parquet"],
    )

    replay_again, episodes_again = execute_paths(payloads)
    tables_again, classes_again, next_again = compute_tables(episodes_again)
    exact = {
        "branchReplay": frame_hash(replay) == frame_hash(replay_again),
        "episodeFunctional": frame_hash(episodes) == frame_hash(episodes_again),
        **{
            name: frame_hash(frame) == frame_hash(tables_again[name])
            for name, frame in tables.items()
        },
    }
    regeneration = {
        "schema": "eidosoma.e01.s19_l46.regeneration_validation.v1",
        "status": "PASS"
        if all(exact.values())
        and classifications == classes_again
        and next_theme == next_again
        else "FAIL",
        "tableExact": exact,
        "classificationExact": classifications == classes_again,
        "nextThemeExact": next_theme == next_again,
        "branchReplayRows": len(replay),
        "functionalEpisodeRows": len(episodes),
    }
    if regeneration["status"] != "PASS":
        raise RuntimeError("L46 exact regeneration failure")

    BASE.write_parquet(BUILD_ROOT / "branch_replay_validation.parquet", replay)
    BASE.write_parquet(BUILD_ROOT / "functional_episode_results.parquet", episodes)
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l46.classification.v1",
            "classifications": classifications,
            "nextTheme": next_theme,
            "priorStatusesChanged": False,
            "newBranchStreams": 0,
            "promotableAsConfirmed": False,
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
    runtime = {
        "schema": "eidosoma.e01.s19_l46.runtime.v1",
        "repositoryHead": lock["head"],
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started,
        "estimatedCpuHoursUpper": (time.perf_counter() - started) * WORKERS / 3600,
        "states": len(payloads),
        "reusedBranchStreams": len(replay),
        "fullReplayPasses": 2,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "completedAtUtc": utc_now(),
    }
    if runtime["estimatedCpuHoursUpper"] > 100 or runtime["wallSeconds"] > 72 * 3600:
        raise RuntimeError("L46 runtime ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    retained_bytes = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    )
    storage = {
        "schema": "eidosoma.e01.s19_l46.storage_validation.v1",
        "status": "PASS" if retained_bytes <= 25 * 1024**3 else "FAIL",
        "retainedBytes": retained_bytes,
        "retainedGiBCeiling": 25,
        "temporaryGiBCeiling": 75,
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        tables["group_function_results.parquet"],
        tables["split_half_reliability.parquet"],
        tables["scientific_gate_results.parquet"],
        classifications,
        runtime,
        next_theme,
    )
    BASE.atomic_text(BUILD_ROOT / "S19_L46_FULL_RESULTS.md", report)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L46 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Next:** `{next_theme}`.\n",
    )
    if storage["status"] != "PASS":
        raise RuntimeError("L46 storage ceiling exceeded")
    for path in (BUILD_ROOT / "figures").glob("*.png"):
        if not path.stat().st_size:
            raise RuntimeError(f"empty L46 figure: {path}")

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
        raise RuntimeError("L46 artifact manifest regeneration failed")

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
            "nextAuthorizedLoop": "S19-L47",
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
    args = parser.parse_args()
    if args.prepare_lock:
        prepare_lock()
    else:
        execute()


if __name__ == "__main__":
    main()
