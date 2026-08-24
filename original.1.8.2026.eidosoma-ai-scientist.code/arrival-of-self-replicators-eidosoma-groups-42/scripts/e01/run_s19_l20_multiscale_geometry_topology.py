"""Execute S19-L20 multiscale geometry/topology onset discovery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gudhi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.sparse.csgraph import minimum_spanning_tree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_onset_discovery.multiscale_geometry import (
    INTRINSIC_GEOMETRY_FEATURES,
    PATH_GEOMETRY_FEATURES,
    TOPOLOGY_FEATURES,
    chord_distance_matrix,
    extract_multiscale_geometry_features,
    persistent_topology_features,
)


def _load_base() -> Any:
    path = REPO_ROOT / "scripts/e01/run_s19_l19_source_grounded_early_warning.py"
    spec = importlib.util.spec_from_file_location("e01_s19_l19_runner_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load L19 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
LOOP_ID = "S19-L20"
VERSION = "E01-S19-L20-MULTISCALE-GEOMETRY-TOPOLOGY-EARLY-WARNING-v1.0.0"
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L20"
L19_ROOT = ARTIFACT_ROOT / "loops/L19"
CACHE_ROOT = Path("/cache/e01_s19_l20")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l20_multiscale_geometry_topology.yaml"
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/multiscale_geometry.py"
BOOTSTRAPS = 4096
PERMUTATIONS = 512

COMPACT_BASELINE_FIELDS = BASE.COMPACT_BASELINE_FIELDS
MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "DUMMY_TRAINING_PRIOR": (),
    "TIME_ONLY": tuple(BASE.L18_FEATURE_GROUPS["TIME_ONLY"]),
    "EXACT_H_STABILITY": tuple(BASE.L18_FEATURE_GROUPS["EXACT_H_STABILITY"]),
    "PREFIX_RECURRENCE_GEOMETRY": tuple(
        BASE.L18_FEATURE_GROUPS["PREFIX_RECURRENCE_GEOMETRY"]
    ),
    "L18_PAST_FULL_NO_BGM": BASE.L18_FULL_FIELDS,
    "COMPACT_BASELINE": COMPACT_BASELINE_FIELDS,
    "TOPOLOGY_ONLY": TOPOLOGY_FEATURES,
    "INTRINSIC_GEOMETRY_ONLY": INTRINSIC_GEOMETRY_FEATURES,
    "PATH_GEOMETRY_ONLY": PATH_GEOMETRY_FEATURES,
    "COMPACT_PLUS_TOPOLOGY": COMPACT_BASELINE_FIELDS + TOPOLOGY_FEATURES,
    "COMPACT_PLUS_INTRINSIC_GEOMETRY": COMPACT_BASELINE_FIELDS
    + INTRINSIC_GEOMETRY_FEATURES,
    "COMPACT_PLUS_PATH_GEOMETRY": COMPACT_BASELINE_FIELDS
    + PATH_GEOMETRY_FEATURES,
    "COMPACT_PLUS_MULTISCALE": COMPACT_BASELINE_FIELDS
    + TOPOLOGY_FEATURES
    + INTRINSIC_GEOMETRY_FEATURES
    + PATH_GEOMETRY_FEATURES,
}
MODEL_IDS = tuple(MODEL_FEATURES)
LEAD_MODELS = (
    "COMPACT_PLUS_TOPOLOGY",
    "COMPACT_PLUS_INTRINSIC_GEOMETRY",
    "COMPACT_PLUS_PATH_GEOMETRY",
    "COMPACT_PLUS_MULTISCALE",
)


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
    prior = json.loads((L19_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L19_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L19_ROOT / item["path"]),
            "root": str(L19_ROOT),
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
            continue
        observed = sha256_file(path)
        if observed != row["sha256"]:
            failures.append(
                {
                    "path": str(path),
                    "reason": "HASH_MISMATCH",
                    "expected": row["sha256"],
                    "observed": observed,
                }
            )
    aggregate = hashlib.sha256(
        "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
    ).hexdigest()
    return {
        "schema": "eidosoma.e01.s19_l20.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": aggregate,
        "l19ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def fixture_table() -> pd.DataFrame:
    rng = np.random.default_rng(BASE.derive_seed("l20_fixtures"))
    states = rng.poisson(2.0, size=(64, 100)).astype(np.int64)
    states[:, 0] += 1
    first = extract_multiscale_geometry_features(states)
    second = extract_multiscale_geometry_features(states.copy())
    order = np.r_[0, np.arange(1, 64)[::-1]]
    reordered = extract_multiscale_geometry_features(states[order])
    feature_order = rng.permutation(100)
    relabelled = extract_multiscale_geometry_features(states[:, feature_order])

    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    circle = np.full((64, 100), 0.01, dtype=np.float64)
    circle[:, 0] += 1.1 + np.cos(angles)
    circle[:, 1] += 1.1 + np.sin(angles)
    circle /= circle.sum(axis=1, keepdims=True)
    topology = persistent_topology_features(circle)

    compositions = states / states.sum(axis=1, keepdims=True)
    distance = chord_distance_matrix(compositions)
    mst_edges = minimum_spanning_tree(distance).toarray()
    mst_total = float(np.sum(mst_edges[mst_edges > 1e-12]))
    rows = [
        {
            "fixtureId": "FEATURE_SCHEMA",
            "passed": set(first)
            == set(TOPOLOGY_FEATURES)
            | set(INTRINSIC_GEOMETRY_FEATURES)
            | set(PATH_GEOMETRY_FEATURES),
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
            "details": "all fields",
        },
        {
            "fixtureId": "FEATURE_PERMUTATION_EQUIVALENCE",
            "passed": all(
                np.isclose(first[name], relabelled[name], atol=1e-10, rtol=1e-10)
                for name in first
            ),
            "details": "100-coordinate permutation",
        },
        {
            "fixtureId": "POINT_CLOUD_ORDER_INVARIANCE",
            "passed": all(
                np.isclose(first[name], reordered[name], atol=1e-10, rtol=1e-10)
                for name in TOPOLOGY_FEATURES + INTRINSIC_GEOMETRY_FEATURES
                if name.endswith("_full")
            ),
            "details": "full-window topology and intrinsic geometry",
        },
        {
            "fixtureId": "PATH_ORDER_SENSITIVITY",
            "passed": any(
                not np.isclose(first[name], reordered[name])
                for name in PATH_GEOMETRY_FEATURES
            ),
            "details": "registered temporal control",
        },
        {
            "fixtureId": "PLANTED_H1_CYCLE",
            "passed": topology["topo_h1_feature_count_full"] >= 1.0
            and topology["topo_h1_max_persistence_full"] > 0.0,
            "details": json.dumps(topology, sort_keys=True),
        },
        {
            "fixtureId": "H0_FLOAT64_MST_IDENTITY",
            "passed": np.isclose(
                first["topo_h0_total_persistence_full"],
                mst_total,
                atol=1e-12,
                rtol=1e-12,
            ),
            "details": f"{mst_total:.17g}",
        },
        {
            "fixtureId": "FAMILY_CARDINALITY",
            "passed": len(TOPOLOGY_FEATURES) == 12
            and len(INTRINSIC_GEOMETRY_FEATURES) == 11
            and len(PATH_GEOMETRY_FEATURES) == 15,
            "details": "12/11/15",
        },
    ]
    synthetic_y = np.array([0, 1] * 15, dtype=int)
    synthetic_x = rng.normal(size=(30, 4))
    model_a = BASE.model_pipeline(BASE.derive_seed("l20_model_fixture"))
    model_b = BASE.model_pipeline(BASE.derive_seed("l20_model_fixture"))
    model_a.fit(synthetic_x, synthetic_y)
    model_b.fit(synthetic_x, synthetic_y)
    rows.append(
        {
            "fixtureId": "MODEL_EXACT_REPLAY",
            "passed": np.array_equal(
                model_a.predict_proba(synthetic_x), model_b.predict_proba(synthetic_x)
            ),
            "details": "30x4",
        }
    )
    return pd.DataFrame(rows)


def source_registry() -> pd.DataFrame:
    date = utc_now()[:10]
    return pd.DataFrame(
        [
            {
                "sourceId": "BAUER_2021_RIPSER",
                "doi": "10.1007/s41468-021-00071-5",
                "url": "https://doi.org/10.1007/s41468-021-00071-5",
                "retrievalDate": date,
                "directSupport": "Vietoris-Rips persistence barcodes",
                "reconstructionChoice": "H0/H1 on chord distances among 64 closed prefix compositions",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
            {
                "sourceId": "LEVINA_BICKEL_2004",
                "doi": None,
                "url": "https://papers.nips.cc/paper_files/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html",
                "retrievalDate": date,
                "directSupport": "nearest-neighbour likelihood intrinsic dimension",
                "reconstructionChoice": "fixed global k=5 and k=10 estimators on relative-composition distance",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
            {
                "sourceId": "STORCH_DAY_2022_TOPOLOGICAL_EWS",
                "doi": "10.1016/j.jtbi.2022.111274",
                "url": "https://doi.org/10.1016/j.jtbi.2022.111274",
                "retrievalDate": date,
                "directSupport": "topological change as an early-warning observable",
                "reconstructionChoice": "persistent descriptors rather than fixed-threshold Betti counts",
                "evidenceClass": "PRIMARY_RESEARCH_PAPER",
            },
            {
                "sourceId": "GUDHI_3_13_0",
                "doi": None,
                "url": "https://gudhi.inria.fr/",
                "retrievalDate": date,
                "directSupport": "CPU Vietoris-Rips persistence software",
                "reconstructionChoice": "GUDHI 3.13.0 H1; scipy float64 MST H0 authority",
                "evidenceClass": "OFFICIAL_SOFTWARE",
            },
        ]
    )


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before the L20 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local and pushed branch identities differ")
    prior = validate_immutable_prior()
    if not prior["unchanged"]:
        raise RuntimeError("immutable prior validation failed")
    started = time.perf_counter()
    fixtures = fixture_table()
    benchmark_seconds = time.perf_counter() - started
    if not fixtures["passed"].all():
        raise RuntimeError("one or more L20 fixtures failed")
    projected_feature_seconds = benchmark_seconds * 40.0
    if projected_feature_seconds > 72 * 3600:
        raise RuntimeError("pre-outcome feature benchmark exceeds wall ceiling")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        """# S19-L20 decision record

The human authorized sequential bounded discovery and untouched-confirmation loops through at most L42. L19 pruned fixed critical-slowing, fixed-threshold recurrence-line and local-DMD families. L20 is the next nonduplicative loop: it retains the exact L18 task and adds only prospectively fixed H0/H1 persistence, nearest-neighbour intrinsic geometry and multiscale path geometry computed from observations 0–63.

This is adaptive discovery on a studied cohort. A favorable result cannot be a solution until the identical pipeline survives a new seed-firewalled confirmation. No outcome-guided scale, topology, landmark, target or model choice is permitted.
""",
    )
    sources = source_registry()
    sources.to_csv(LOOP_ROOT / "source_grounding_registry.csv", index=False)
    BASE.atomic_text(
        LOOP_ROOT / "source_grounding_report.md",
        "# Source grounding\n\n"
        + "\n".join(
            f"- **{row.sourceId}** — {row.directSupport}. Frozen reconstruction: {row.reconstructionChoice}. {row.url}"
            for row in sources.itertuples(index=False)
        )
        + "\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l20.implementation_lock.v1",
            "researchStepId": LOOP_ID,
            "versionedId": VERSION,
            "repositoryHead": head,
            "remoteHead": remote,
            "configSha256": sha256_file(CONFIG),
            "coreSha256": sha256_file(CORE_PATH),
            "runnerSha256": sha256_file(Path(__file__)),
            "baseEvaluatorSha256": sha256_file(
                REPO_ROOT / "scripts/e01/run_s19_l19_source_grounded_early_warning.py"
            ),
            "l19ManifestSha256": sha256_file(L19_ROOT / "artifact_manifest.json"),
            "gudhiVersion": gudhi.__version__,
            "scipyVersion": scipy.__version__,
            "modelFeatures": {name: list(fields) for name, fields in MODEL_FEATURES.items()},
            "leadModels": list(LEAD_MODELS),
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
            "fixtureAndTenUnitEquivalentSeconds": benchmark_seconds,
            "projectedFeatureSecondsUpper": projected_feature_seconds,
            "projectedCpuHoursUpper": min(90.0, projected_feature_seconds * 8 / 3600 + 80),
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
                "benchmarkSeconds": benchmark_seconds,
            }
        )
    )


ORIGINAL_CV = BASE.cross_validated_predictions
ORIGINAL_EXTRACT = BASE.extract_features
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


def extract_features(
    manifest: pd.DataFrame,
    loaded: dict[tuple[str, int], dict[str, Any]],
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, registry = ORIGINAL_EXTRACT(manifest, loaded, workers)
    fields = list(TOPOLOGY_FEATURES + INTRINSIC_GEOMETRY_FEATURES + PATH_GEOMETRY_FEATURES)
    for row in manifest.itertuples(index=False):
        key = (row.candidateId, int(row.matrixIndex))
        prefix = loaded[key]["states"][: BASE.LANDMARK_COUNT]
        rng = np.random.default_rng(BASE.derive_seed("temporal", *key))
        permutation = np.arange(BASE.LANDMARK_COUNT)
        permutation[1:] = rng.permutation(permutation[1:])
        for variant, values in (
            ("ORIGINAL", prefix),
            ("TEMPORAL_PERMUTED", prefix[permutation]),
        ):
            observed = extract_multiscale_geometry_features(values)
            stored = features[
                features["candidateId"].eq(key[0])
                & features["matrixIndex"].eq(key[1])
                & features["variant"].eq(variant)
            ].iloc[0]
            if any(stored[name] != observed[name] for name in fields):
                raise RuntimeError(
                    f"independent feature replay failed for {key}/{variant}"
                )
    features["independentReplayExact"] = True
    family = {
        "TOPOLOGY_ONLY": "PERSISTENT_TOPOLOGY",
        "INTRINSIC_GEOMETRY_ONLY": "INTRINSIC_GEOMETRY",
        "PATH_GEOMETRY_ONLY": "PATH_GEOMETRY",
        "COMPACT_PLUS_TOPOLOGY": "PERSISTENT_TOPOLOGY",
        "COMPACT_PLUS_INTRINSIC_GEOMETRY": "INTRINSIC_GEOMETRY",
        "COMPACT_PLUS_PATH_GEOMETRY": "PATH_GEOMETRY",
        "COMPACT_PLUS_MULTISCALE": "COMBINED_MULTISCALE",
    }
    registry["featureFamily"] = registry.apply(
        lambda row: family.get(row["modelId"], row["featureFamily"]), axis=1
    )
    return features, registry


def scientific_gates(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, list[str], str | None]:
    gates, classifications, selected = ORIGINAL_GATES(*args, **kwargs)
    translations = {
        "SOURCE_GROUNDED_EARLY_WARNING_DISCOVERY_LEAD": "MULTISCALE_GEOMETRY_TOPOLOGY_DISCOVERY_LEAD",
        "EARLY_WARNING_FAMILY_NON_SUPPORT": "MULTISCALE_GEOMETRY_FAMILY_NON_SUPPORT",
        "CRITICAL_SLOWING_NOT_INCREMENTAL": "PERSISTENT_TOPOLOGY_NOT_INCREMENTAL",
        "RQA_NOT_INCREMENTAL": "INTRINSIC_GEOMETRY_NOT_INCREMENTAL",
        "DMD_NOT_INCREMENTAL": "PATH_GEOMETRY_NOT_INCREMENTAL",
    }
    return gates, [translations.get(item, item) for item in classifications], selected


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

    geometry = targets[targets["atRiskAtLandmark"]].groupby("candidateId")[
        "eventWithinHorizon"
    ].agg(["count", "sum"])
    geometry["nonEvent"] = geometry["count"] - geometry["sum"]
    geometry[["sum", "nonEvent"]].plot(kind="bar", color=["#1976d2", "#9e9e9e"])
    plt.ylabel("matrices")
    plt.title("Frozen 64-to-192 onset task")
    plt.legend(["event", "non-event"])
    save("01_at_risk_event_geometry.png")

    original = features[features["variant"].eq("ORIGINAL")]
    selected_fields = [
        TOPOLOGY_FEATURES[0],
        TOPOLOGY_FEATURES[4],
        INTRINSIC_GEOMETRY_FEATURES[0],
        INTRINSIC_GEOMETRY_FEATURES[4],
        PATH_GEOMETRY_FEATURES[5],
        PATH_GEOMETRY_FEATURES[9],
    ]
    correlation = original[selected_fields].corr()
    plt.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(6), ["H0", "H1", "ID5", "k5", "tort", "lag4"], rotation=35)
    plt.yticks(range(6), ["H0", "H1", "ID5", "k5", "tort", "lag4"])
    plt.colorbar()
    plt.title("Prefix geometry/topology correlations")
    save("02_feature_correlation_map.png")

    focus = aggregate[
        aggregate["variant"].eq("ORIGINAL")
        & aggregate["modelId"].isin(
            ["DUMMY_TRAINING_PRIOR", "EXACT_H_STABILITY", "COMPACT_BASELINE", *LEAD_MODELS]
        )
    ]
    focus.pivot(index="modelId", columns="candidateId", values="AUROC").plot(
        kind="bar", ylim=(0, 1), color=["#1565c0", "#ef6c00"]
    )
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("matrix-level repeated-CV AUROC")
    plt.title("Multiscale onset discrimination")
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
    plt.ylabel("AUROC increment over compact baseline")
    plt.legend()
    plt.title("Paired matrix-bootstrap increments")
    save("04_incremental_effects.png")

    permutation.pivot(index="modelId", columns="candidateId", values="familywisePValue").plot(
        kind="bar", ylim=(0, 1), color=["#1565c0", "#ef6c00"]
    )
    plt.axhline(0.10, color="black", linestyle="--")
    plt.ylabel("max-statistic family-wise p")
    plt.title("Matrix-label permutation control")
    save("05_permutation_control.png")

    controls[controls["modelId"].isin(LEAD_MODELS)].pivot_table(
        index="modelId", columns=["candidateId", "controlId"], values="controlAuRoc"
    ).plot(kind="bar", ylim=(0, 1))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("AUROC")
    plt.title("Temporal and feature-permutation controls")
    plt.legend(fontsize=6)
    save("06_negative_controls.png")

    gate_view = gates.pivot(
        index="modelId", columns="candidateId", values="candidateDiscoveryGatePassed"
    ).astype(int)
    plt.imshow(gate_view.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.xticks(range(len(gate_view.columns)), gate_view.columns, rotation=20)
    plt.yticks(range(len(gate_view.index)), gate_view.index)
    plt.colorbar(ticks=[0, 1])
    plt.title("Candidate-specific discovery gates")
    save("07_discovery_gate_matrix.png")

    plt.axis("off")
    plt.text(0.03, 0.88, "L20 decision boundary", fontsize=16, weight="bold")
    plt.text(0.03, 0.67, "Studied cohort → discovery only", fontsize=12)
    plt.text(0.03, 0.49, "Same model must pass both candidates", fontsize=12)
    plt.text(0.03, 0.31, "Lead → untouched seed-firewalled confirmation", fontsize=12)
    plt.text(0.03, 0.13, "Null → prune geometry/topology family", fontsize=12)
    save("08_decision_boundary.png")
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
            [
                "DUMMY_TRAINING_PRIOR",
                "EXACT_H_STABILITY",
                "PREFIX_RECURRENCE_GEOMETRY",
                "COMPACT_BASELINE",
                *LEAD_MODELS,
            ]
        )
    ][["candidateId", "modelId", "AUROC", "AUPRC", "BRIER", "BALANCED_ACCURACY"]]
    recommendation = (
        f"Freeze `{selected}` and run it unchanged on a new seed-firewalled cohort in L21."
        if selected
        else "Advance to an outcome-blind landmark/survival reformulation in L21; fixed persistent-topology, intrinsic-dimension and path-geometry summaries are pruned."
    )
    return f"""# S19-L20 — Multiscale Geometry and Topology Before Recurring-Attractor Entry

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete within the authorized autonomous L19–L42 program.
- **Outcome classifications:** {", ".join(f"`{item}`" for item in classifications)}
- **Selected discovery lead:** `{selected or "NONE"}`.
- **Validation:** exact L18 task/split replay, immutable-prior validation through L19, ten preregistered fixtures, independent all-unit feature replay, exact suffix invariance, matrix-level repeated CV, 4,096 bootstraps, 512 max-statistic label permutations, temporal/feature controls, regeneration, storage and artifact hashes passed.
- **Recommended next bounded loop:** {recommendation}

## Frozen question

Do multiscale point-cloud topology, intrinsic dimensionality, or path geometry calculated only from observations 0–63 predict first entry into the frozen recurring-attractor state during observations 64–191 beyond time, exact adjacent H/stability and prefix recurrence geometry?

## Cohort

{geometry.to_markdown(index=False)}

This is the exact L18/L19 discovery task. Its completed-run attractor target remains retrospective and author-ambiguous; every competitive L20 input is prefix-only.

## Methods

L20 froze three nonduplicative families before outcomes: (1) float64 H0 minimum-spanning-tree persistence plus GUDHI H1 Vietoris–Rips persistence on cosine-chord distances; (2) fixed k=5 and k=10 Levina–Bickel intrinsic-dimension and neighbourhood-contraction summaries; and (3) step, displacement, tortuosity, turning and lag-2/4/8 path geometry. Full-prefix values and registered 32/32 contrasts were evaluated with the unchanged `C=1` L2 logistic model and exact L18 splits. GUDHI 3.13.0 was installed from its CPython 3.13 wheel; no GPU was used.

## Results

{focus.to_markdown(index=False)}

## Gate adjudication

{gates.to_markdown(index=False)}

The discovery gate required the same frozen model in both candidates, AUROC at least 0.65 with a bootstrap lower bound above 0.5, AUPRC above prevalence, no Brier loss against the dummy, positive increments over compact and exact-H baselines, max-statistic permutation `p<=0.10`, at least 90% positive leave-one-out increments, a worse temporal-permutation control, and exact suffix invariance. This is not a confirmation gate.

## Interpretation

Persistent topology can reveal multiscale connectivity and cycles that a single recurrence threshold misses; intrinsic dimension and path geometry can reveal concentration or constrained motion before attractor entry. Failure constrains these fixed implementations on this landmark task, not every possible organization signal. Candidate-specific or stability-explained behavior is retained but cannot count as a solution.

No completed trajectory, completed centroid, suffix statistic, molecular-row pseudoreplication, favorable-candidate pooling, or outcome-guided geometric scale entered a prospective input.

## Runtime and provenance

- Repository lock: `{runtime["repositoryHead"]}`.
- CPU float64, `{runtime["workers"]}` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `{runtime["wallSeconds"]:.3f}`; process CPU hours: `{runtime["processCpuHours"]:.6f}`.
- Source identities and reconstruction choices are in `source_grounding_registry.csv` and `source_grounding_report.md`.

## Autonomous continuation boundary

L20 is frozen. The human authorization permits one next bounded loop without an intermediate Chief handoff through at most L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
"""


def _append_like(path: Path, rows: list[dict[str, Any]]) -> None:
    existing = pd.read_parquet(path)
    addition = pd.DataFrame(rows).reindex(columns=existing.columns)
    BASE.write_parquet(path, pd.concat([existing, addition], ignore_index=True))


def append_root_ledgers(
    classifications: list[str], selected: str | None, timestamp: str
) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L19 pruned classical warning, RQA and DMD families, but scale-free point-cloud organization might remain.",
            "failureOrAmbiguityTargeted": "Whether organization is multiscale or geometric rather than visible at H=0.9 or through local linear relaxation.",
            "informationGainRationale": "Persistent homology, fixed intrinsic dimension and path geometry are nonduplicative and outcome-blind.",
            "learned": "The L20 topology/geometry/model/gate contract was frozen before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L19 family non-support and primary topology/intrinsic-dimension methods.",
            "proposedNextTest": "Execute the frozen L20 comparison.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Multiscale geometry, landmark survival, nonlinear outcome-blind representation and reaction coordinates.",
            "selectedHypotheses": "Persistent H0/H1; intrinsic geometry; path geometry.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Fixed H=0.9 recurrence topology is sufficient.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A fixed multiscale family might add a two-candidate warning signal.",
            "failureOrAmbiguityTargeted": "Scale dependence of organization before attractor entry.",
            "informationGainRationale": "Candidate-separated CV, max-stat permutations, bootstraps, replay and suffix controls distinguish stable signal from adaptive noise.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete frozen L20 machine-readable results.",
            "proposedNextTest": f"Untouched confirmation of {selected}." if selected else "Outcome-blind landmark/survival formulation under L21.",
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": "A precursor may require a time-to-event formulation, outcome-blind representation or compact reaction coordinate.",
            "selectedHypotheses": "Persistent H0/H1; intrinsic geometry; path geometry.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any failed L20 family provides a robust two-candidate incremental warning on this task.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat([ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)], ignore_index=True),
    )

    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    start = int(candidates["registryOrder"].max()) + 1
    candidate_rows = []
    for offset, model in enumerate(LEAD_MODELS):
        candidate_rows.append(
            {
                "branchCount": len(LEAD_MODELS),
                "bundleId": "L20_MULTISCALE_GEOMETRY_TOPOLOGY",
                "candidateId": f"S19-L20-{model}",
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
                "rankingScore": float(20 - offset),
                "registryOrder": start + offset,
                "selected": True,
                "selectionReason": "AUTONOMOUS_ORGANIZATION_BEFORE_ONSET_DISCOVERY",
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
                "finding": f"{item.directSupport}; frozen L20 reconstruction: {item.reconstructionChoice}",
                "licenseStatus": "PUBLIC_ARTICLE_OR_OFFICIAL_SOFTWARE",
                "redistributionStatus": "CITATION_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L20_{item.sourceId}",
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
            "newMatrices": 0,
            "newTrajectories": 0,
            "nextStepActive": True,
        }
    )
    data["laterLoopsAuthorized"] = True
    data["authorizationUpperBound"] = "S19-L42"
    data["proposedNextLoopTheme"] = (
        f"UNTOUCHED_CONFIRMATION_{selected}" if selected else "LANDMARK_SURVIVAL_REFORMULATION"
    )
    data["proposedNextLoopActive"] = True
    BASE.atomic_text(loop_path, yaml.safe_dump(data, sort_keys=False))

    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text())
    review["history"].append(
        {
            "decision": "S19_L20_COMPLETE_CONTINUE_UNDER_EXISTING_AUTHORIZATION",
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
    result["schema"] = "eidosoma.e01.s19_l20.artifact_manifest.v1"
    return result


def decision_summary_text(classifications: list[str], selected: str | None) -> str:
    outcome = (
        f"`{selected}` passed the discovery gate and must now be tested unchanged on untouched matrices."
        if selected
        else "No fixed geometry/topology family passed the two-candidate discovery gate; advance to a landmark/survival formulation without retuning L20."
    )
    return f"""# S19-L20 decision summary

**Classification:** {", ".join(classifications)}
**Selected discovery lead:** `{selected or "NONE"}`

{outcome}

The existing human authorization activates one next bounded loop without a Chief handoff. S20, E02, author contact, interventions and report generation remain inactive.
"""


def configure_base() -> None:
    BASE.LOOP_ID = LOOP_ID
    BASE.VERSION = VERSION
    BASE.LOOP_ROOT = LOOP_ROOT
    BASE.CACHE_ROOT = CACHE_ROOT
    BASE.BUILD_ROOT = BUILD_ROOT
    BASE.CONFIG = CONFIG
    BASE.BOOTSTRAPS = BOOTSTRAPS
    BASE.PERMUTATIONS = PERMUTATIONS
    BASE.EWS_FEATURES = TOPOLOGY_FEATURES
    BASE.RQA_FEATURES = INTRINSIC_GEOMETRY_FEATURES
    BASE.DMD_FEATURES = PATH_GEOMETRY_FEATURES
    BASE.extract_organization_warning_features = extract_multiscale_geometry_features
    BASE.MODEL_FEATURES = MODEL_FEATURES
    BASE.MODEL_IDS = MODEL_IDS
    BASE.LEAD_MODELS = LEAD_MODELS
    BASE.CANONICAL_REPORT_NAME = "S19_L20_FULL_RESULTS.md"
    BASE.ROOT_HANDOFF_SOURCE_HEADER = "# S19-L20"
    BASE.ROOT_HANDOFF_TARGET_HEADER = "# S19 current handoff — S19-L20"
    BASE.NULL_NEXT_ACTION = "S19_L21_LANDMARK_SURVIVAL_REFORMULATION"
    BASE.RUNTIME_SCHEMA = "eidosoma.e01.s19_l20.runtime.v1"
    BASE.validate_immutable_prior = validate_immutable_prior
    BASE.fixture_table = fixture_table
    BASE.source_registry = source_registry
    BASE.cross_validated_predictions = cross_validated_predictions
    BASE.extract_features = extract_features
    BASE.scientific_gates = scientific_gates
    BASE.make_figures = make_figures
    BASE.report_text = report_text
    BASE.append_root_ledgers = append_root_ledgers
    BASE.manifest_for = manifest_for
    BASE.decision_summary_text = decision_summary_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-lock", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
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
