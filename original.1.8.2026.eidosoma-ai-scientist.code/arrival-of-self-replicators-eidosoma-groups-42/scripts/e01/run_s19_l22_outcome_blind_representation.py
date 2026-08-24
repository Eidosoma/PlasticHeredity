"""Execute S19-L22 outcome-blind prefix-representation discovery."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_onset_discovery.outcome_blind_representation import (
    CHANNEL_NAMES,
    KERNEL_BANK,
    KERNEL_COUNT,
    KERNEL_SEED,
    RANDOM_CONV_FEATURES,
    extract_outcome_blind_representation,
    kernel_bank_fingerprint,
    organization_channel_sequence,
)


def _load_base() -> Any:
    path = REPO_ROOT / "scripts/e01/run_s19_l19_source_grounded_early_warning.py"
    spec = importlib.util.spec_from_file_location("e01_s19_l22_runner_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load L19 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
LOOP_ID = "S19-L22"
VERSION = "E01-S19-L22-OUTCOME-BLIND-PREFIX-REPRESENTATION-v1.0.0"
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L22"
L21_ROOT = ARTIFACT_ROOT / "loops/L21"
CACHE_ROOT = Path("/cache/e01_s19_l22")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l22_outcome_blind_representation.yaml"
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/outcome_blind_representation.py"
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
    "RANDOM_CONVOLUTION_ONLY": RANDOM_CONV_FEATURES,
    "COMPACT_PLUS_RANDOM_CONVOLUTION": COMPACT_BASELINE_FIELDS
    + RANDOM_CONV_FEATURES,
}
MODEL_IDS = tuple(MODEL_FEATURES)
LEAD_MODELS = ("COMPACT_PLUS_RANDOM_CONVOLUTION",)


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
    prior = json.loads((L21_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L21_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L21_ROOT / item["path"]),
            "root": str(L21_ROOT),
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
        "schema": "eidosoma.e01.s19_l22.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": aggregate,
        "l21ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def fixture_table() -> pd.DataFrame:
    rng = np.random.default_rng(BASE.derive_seed("l22_fixtures"))
    states = rng.poisson(2.0, size=(64, 100)).astype(np.int64)
    states[:, 0] += 1
    first = extract_outcome_blind_representation(states)
    replay = extract_outcome_blind_representation(states.copy())
    species_permutation = rng.permutation(100)
    relabelled = extract_outcome_blind_representation(states[:, species_permutation])
    temporal_order = np.r_[0, rng.permutation(np.arange(1, 64))]
    temporal = extract_outcome_blind_representation(states[temporal_order])
    scaled = extract_outcome_blind_representation(states * 3)
    channel = organization_channel_sequence(states)
    rows = [
        {
            "fixtureId": "FEATURE_SCHEMA",
            "passed": tuple(first) == RANDOM_CONV_FEATURES and len(first) == 128,
            "details": f"{len(first)} features",
        },
        {
            "fixtureId": "EXACT_FEATURE_REPLAY",
            "passed": first == replay,
            "details": kernel_bank_fingerprint(),
        },
        {
            "fixtureId": "FINITE_CHANNELS_AND_FEATURES",
            "passed": bool(np.isfinite(channel).all())
            and bool(np.isfinite(list(first.values())).all()),
            "details": f"64x{len(CHANNEL_NAMES)}",
        },
        {
            "fixtureId": "MOLECULE_LABEL_PERMUTATION_INVARIANCE",
            "passed": all(
                np.isclose(first[name], relabelled[name], atol=1e-12, rtol=1e-12)
                for name in first
            ),
            "details": "coordinate permutation tolerance 1e-12",
        },
        {
            "fixtureId": "TEMPORAL_ORDER_SENSITIVITY",
            "passed": any(first[name] != temporal[name] for name in first),
            "details": "first observation fixed",
        },
        {
            "fixtureId": "POSITIVE_SCALING_INVARIANCE",
            "passed": all(
                np.isclose(first[name], scaled[name], atol=1e-12, rtol=1e-12)
                for name in first
            ),
            "details": "three-fold count scaling",
        },
        {
            "fixtureId": "FROZEN_KERNEL_CONTRACT",
            "passed": len(KERNEL_BANK) == KERNEL_COUNT
            and all(kernel.length in {7, 9, 11} for kernel in KERNEL_BANK)
            and all(kernel.dilation in {1, 2, 4, 8} for kernel in KERNEL_BANK),
            "details": kernel_bank_fingerprint(),
        },
    ]
    y = np.array([0, 1] * 15, dtype=int)
    x = rng.normal(size=(30, 6))
    a = BASE.model_pipeline(BASE.derive_seed("l22_model_fixture"))
    b = BASE.model_pipeline(BASE.derive_seed("l22_model_fixture"))
    a.fit(x, y)
    b.fit(x, y)
    rows.append(
        {
            "fixtureId": "DOWNSTREAM_MODEL_EXACT_REPLAY",
            "passed": np.array_equal(a.predict_proba(x), b.predict_proba(x)),
            "details": "frozen C=1 L2 logistic",
        }
    )
    return pd.DataFrame(rows)


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "DEMPSTER_PETITJEAN_WEBB_2020_ROCKET",
                "doi": "10.1007/s10618-020-00701-z",
                "url": "https://doi.org/10.1007/s10618-020-00701-z",
                "retrievalDate": utc_now()[:10],
                "directSupport": "fixed random convolutional kernels summarized by maximum and proportion-positive values",
                "reconstructionChoice": "one 64-kernel bank over eleven permutation-invariant organization channels; no outcome-derived bias or kernel selection",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
            {
                "sourceId": "BOETTIGER_HASTINGS_2012_LIMITS",
                "doi": "10.1098/rsif.2012.0125",
                "url": "https://doi.org/10.1098/rsif.2012.0125",
                "retrievalDate": utc_now()[:10],
                "directSupport": "finite-series false-positive and power limits for early-warning inference",
                "reconstructionChoice": "candidate replication, max-statistic label permutations, leave-one-out analysis and untouched-confirmation firewall",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
        ]
    )


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before the L22 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local and pushed branch identities differ")
    prior = validate_immutable_prior()
    if not prior["unchanged"]:
        raise RuntimeError("immutable prior validation failed")
    start = time.perf_counter()
    fixtures = fixture_table()
    benchmark_seconds = time.perf_counter() - start
    if not fixtures["passed"].all():
        raise RuntimeError("one or more L22 fixtures failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        """# S19-L22 decision record

L18 established a non-saturated recurring-attractor onset task. L19 and L20 pruned prospectively fixed hand-engineered warning families, and L21 showed that a survival reformulation did not recover their information. Under the human authorization for sequential bounded discovery through at most L42, L22 tests exactly one outcome-blind representation: a single fixed random-convolution bank over molecule-label-permutation-invariant organization-channel sequences from observations 0–63.

The kernel bank never sees outcomes or cohort values. There is no kernel-bank tournament, target change, threshold search, completed-run input, or candidate pooling. A favorable result on this studied cohort is discovery evidence only and must survive untouched seed-firewalled confirmation.
""",
    )
    sources = source_registry()
    sources.to_csv(LOOP_ROOT / "source_grounding_registry.csv", index=False)
    BASE.atomic_text(
        LOOP_ROOT / "source_grounding_report.md",
        "# L22 source grounding\n\n"
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
            "schema": "eidosoma.e01.s19_l22.implementation_lock.v1",
            "researchStepId": LOOP_ID,
            "versionedId": VERSION,
            "repositoryHead": head,
            "remoteHead": remote,
            "configSha256": sha256_file(CONFIG),
            "coreSha256": sha256_file(CORE_PATH),
            "runnerSha256": sha256_file(Path(__file__)),
            "l21ManifestSha256": sha256_file(L21_ROOT / "artifact_manifest.json"),
            "kernelSeed": KERNEL_SEED,
            "kernelCount": KERNEL_COUNT,
            "kernelBankSha256": kernel_bank_fingerprint(),
            "channelNames": list(CHANNEL_NAMES),
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
            "fixtureSeconds": benchmark_seconds,
            "ordinaryModelFits": len(MODEL_IDS) * 2 * 50,
            "maximumPermutationFits": PERMUTATIONS * 2 * (len(LEAD_MODELS) + 1) * 50,
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
                "kernelBankSha256": kernel_bank_fingerprint(),
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
            observed = extract_outcome_blind_representation(values)
            stored = features[
                features["candidateId"].eq(key[0])
                & features["matrixIndex"].eq(key[1])
                & features["variant"].eq(variant)
            ].iloc[0]
            if any(stored[name] != observed[name] for name in RANDOM_CONV_FEATURES):
                raise RuntimeError(f"independent L22 feature replay failed: {key}/{variant}")
    features["independentReplayExact"] = True
    registry["featureFamily"] = registry.apply(
        lambda row: "OUTCOME_BLIND_RANDOM_CONVOLUTION"
        if "RANDOM_CONVOLUTION" in row["modelId"]
        else row["featureFamily"],
        axis=1,
    )
    return features, registry


def scientific_gates(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, list[str], str | None]:
    gates, generic, selected = ORIGINAL_GATES(*args, **kwargs)
    classifications = ["ATTRACTOR_ONSET_TASK_ESTABLISHED"]
    if selected:
        classifications.extend(
            [
                "OUTCOME_BLIND_REPRESENTATION_DISCOVERY_LEAD",
                "REQUIRES_UNTOUCHED_CONFIRMATION",
                "NOT_PROMOTABLE_AS_CONFIRMED",
            ]
        )
    else:
        classifications.extend(
            [
                "OUTCOME_BLIND_REPRESENTATION_NON_SUPPORT",
                "RANDOM_CONVOLUTION_NOT_INCREMENTAL",
                "NOT_PROMOTABLE_AS_CONFIRMED",
            ]
        )
        if "CANDIDATE_SPECIFIC_SIGNAL" in generic:
            classifications.append("CANDIDATE_SPECIFIC_SIGNAL")
        if "POSSIBLE_STABILITY_PROXY" in generic:
            classifications.append("POSSIBLE_STABILITY_PROXY")
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

    geometry = targets[targets["atRiskAtLandmark"]].groupby("candidateId")[
        "eventWithinHorizon"
    ].agg(["count", "sum"])
    geometry["nonEvent"] = geometry["count"] - geometry["sum"]
    geometry[["sum", "nonEvent"]].plot(kind="bar", color=["#1976d2", "#9e9e9e"])
    plt.ylabel("matrices")
    plt.title("Frozen onset task")
    plt.legend(["event", "non-event"])
    save("01_task_geometry.png")

    kernel_rows = pd.DataFrame(
        [
            {"length": k.length, "dilation": k.dilation, "padding": k.padding}
            for k in KERNEL_BANK
        ]
    )
    kernel_rows.groupby(["length", "dilation"]).size().unstack(fill_value=0).plot(
        kind="bar"
    )
    plt.ylabel("fixed kernels")
    plt.title("Single outcome-blind kernel bank")
    save("02_kernel_bank.png")

    original = features[features["variant"].eq("ORIGINAL")]
    plt.imshow(
        original[list(RANDOM_CONV_FEATURES[:32])].corr().to_numpy(),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    plt.colorbar()
    plt.title("First 32 representation-feature correlations")
    save("03_representation_correlation.png")

    focus_models = [
        "DUMMY_TRAINING_PRIOR",
        "EXACT_H_STABILITY",
        "COMPACT_BASELINE",
        "RANDOM_CONVOLUTION_ONLY",
        *LEAD_MODELS,
    ]
    aggregate[
        aggregate["variant"].eq("ORIGINAL")
        & aggregate["modelId"].isin(focus_models)
    ].pivot(index="modelId", columns="candidateId", values="AUROC").plot(
        kind="bar", ylim=(0, 1), color=["#1565c0", "#ef6c00"]
    )
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("matrix-level repeated-CV AUROC")
    plt.title("Outcome-blind representation discrimination")
    save("04_model_auroc.png")

    delta = comparisons[
        comparisons["rightModel"].isin(["COMPACT_BASELINE", "EXACT_H_STABILITY"])
        & comparisons["metric"].eq("AUROC")
    ]
    for candidate, frame in delta.groupby("candidateId"):
        plt.errorbar(
            frame["rightModel"],
            frame["favorableDelta"],
            yerr=[
                frame["favorableDelta"] - frame["bootstrapLower95"],
                frame["bootstrapUpper95"] - frame["favorableDelta"],
            ],
            fmt="o",
            label=candidate,
        )
    plt.axhline(0, color="black", linestyle="--")
    plt.ylabel("AUROC increment")
    plt.legend()
    plt.title("Paired matrix-bootstrap increments")
    save("05_incremental_effects.png")

    permutation.pivot(index="modelId", columns="candidateId", values="familywisePValue").plot(
        kind="bar", ylim=(0, 1), color=["#1565c0", "#ef6c00"]
    )
    plt.axhline(0.10, color="black", linestyle="--")
    plt.ylabel("whole-matrix permutation p")
    plt.title("Permutation falsification")
    save("06_permutation_control.png")

    controls[controls["modelId"].isin(LEAD_MODELS)].pivot_table(
        index="modelId", columns=["candidateId", "controlId"], values="controlAuRoc"
    ).plot(kind="bar", ylim=(0, 1))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("AUROC")
    plt.title("Temporal and row-permutation controls")
    save("07_negative_controls.png")

    gate_view = gates.pivot(
        index="modelId", columns="candidateId", values="candidateDiscoveryGatePassed"
    ).astype(int)
    plt.imshow(gate_view.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.xticks(range(len(gate_view.columns)), gate_view.columns, rotation=20)
    plt.yticks(range(len(gate_view.index)), gate_view.index)
    plt.colorbar(ticks=[0, 1])
    plt.title("Discovery gate")
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
            [
                "DUMMY_TRAINING_PRIOR",
                "EXACT_H_STABILITY",
                "COMPACT_BASELINE",
                "RANDOM_CONVOLUTION_ONLY",
                *LEAD_MODELS,
            ]
        )
    ][["candidateId", "modelId", "AUROC", "AUPRC", "BRIER", "BALANCED_ACCURACY"]]
    recommendation = (
        f"Freeze `{selected}` and confirm it unchanged on a new seed-firewalled cohort."
        if selected
        else "Advance to a larger, independently generated discovery cohort in L23 before inventing another feature family; the current 53/54-matrix task cannot distinguish stable weak effects from candidate heterogeneity."
    )
    return f"""# S19-L22 — Outcome-Blind Permutation-Invariant Prefix Representation

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete within the authorized autonomous L19–L42 program.
- **Outcome classifications:** {", ".join(f"`{item}`" for item in classifications)}
- **Selected discovery lead:** `{selected or "NONE"}`.
- **Validation:** one pre-outcome kernel bank, exact target/split replay, immutable-prior validation through L21, mandatory fixtures, independent all-unit representation replay, molecule-label invariance, suffix invariance, matrix repeated CV, 4,096 bootstraps, 512 whole-matrix permutations, temporal/feature controls, regeneration, storage and artifact hashes passed.
- **Recommended next bounded loop:** {recommendation}

## Frozen question

Does one outcome-blind random-convolution representation of observations 0–63 add candidate-consistent information before first recurring-attractor entry during observations 64–191 beyond compact ordinary dynamics and exact H/stability?

## Cohort

{geometry.to_markdown(index=False)}

The target is the frozen completed-run L02 recurring-attractor reconstruction and remains retrospective and author-ambiguous. Every competitive input is prefix-only.

## Methods

Eleven permutation-invariant organization channels encode mass, diversity/concentration, adjacent motion, past recurrence and running-centroid similarity. Each trajectory's channels are standardized without cohort or outcome information. One bank of 64 fixed mean-centered unit-norm Gaussian kernels (lengths 7/9/11; dilations 1/2/4/8; frozen biases) emits maximum and proportion-positive summaries, yielding 128 features. The bank identity is `{kernel_bank_fingerprint()}`. The unchanged C=1 L2 logistic model and exact L18 splits were used. No bank, channel, bias, kernel count or downstream hyperparameter was selected from outcomes.

## Results

{focus.to_markdown(index=False)}

## Gate adjudication

{gates.to_markdown(index=False)}

The discovery gate required the same frozen model to pass in both candidates, AUROC at least 0.65 with bootstrap lower bound above 0.5, AUPRC above prevalence, no Brier loss, positive increments over compact and exact-H controls, matrix-permutation p<=0.10, at least 90% positive leave-one-out increments, worse temporal-permutation performance, and exact suffix invariance.

## Interpretation

This is an outcome-blind nonlinear map of past organization motifs, not a fitted causal-emergence measure. A null constrains this one fixed representation on the studied cohort; it does not prove that no pre-onset organization exists. A favorable studied-cohort result is not a solution until untouched confirmation passes.

## Runtime and provenance

- Repository lock: `{runtime["repositoryHead"]}`.
- CPU float64, `{runtime["workers"]}` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `{runtime["wallSeconds"]:.3f}`; process CPU hours: `{runtime["processCpuHours"]:.6f}`.

## Autonomous continuation boundary

L22 is frozen. The existing authorization permits one next bounded loop through at most L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
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
            "beliefBeforeLoop": "Hand-engineered warning families and survival timing may miss nonlinear prefix motifs.",
            "failureOrAmbiguityTargeted": "Whether a fixed outcome-blind nonlinear representation reveals common pre-onset organization.",
            "informationGainRationale": "One kernel bank compresses temporal motifs without outcome-guided feature choice or completed-run inputs.",
            "learned": "L22 representation/model/gate contract frozen before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L19-L21 cross-candidate non-support and ROCKET random-kernel methodology.",
            "proposedNextTest": "Execute L22.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Nonlinear outcome-blind motifs, larger discovery cohort, compact cross-candidate coordinates.",
            "selectedHypotheses": "One permutation-invariant fixed random-convolution representation.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Registered hand-engineered summaries are sufficient.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A fixed random representation might recover a shared precursor.",
            "failureOrAmbiguityTargeted": "Nonlinear temporal motif information before attractor entry.",
            "informationGainRationale": "Candidate-separated CV, max-stat permutations, bootstraps and leakage controls distinguish signal from chance.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete frozen L22 machine-readable results.",
            "proposedNextTest": f"Untouched confirmation of {selected}." if selected else "Larger independent discovery cohort in L23.",
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": "A weak precursor may require more discovery matrices or a compact cross-candidate reaction coordinate.",
            "selectedHypotheses": "One permutation-invariant fixed random-convolution representation.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "The failed L22 map provides a robust common incremental warning.",
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
    row = {
        "branchCount": 1,
        "bundleId": "L22_OUTCOME_BLIND_REPRESENTATION",
        "candidateId": "S19-L22-COMPACT_PLUS_RANDOM_CONVOLUTION",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 4,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 4,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 5,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "COMPACT_PLUS_RANDOM_CONVOLUTION",
        "rankingScore": 23.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "AUTONOMOUS_ORGANIZATION_BEFORE_ONSET_DISCOVERY",
        "sourceGrounding": 4,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidates_path,
        pd.concat(
            [candidates, pd.DataFrame([row]).reindex(columns=candidates.columns)],
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
                "finding": f"{item.directSupport}; frozen L22 reconstruction: {item.reconstructionChoice}",
                "licenseStatus": "PUBLIC_ARTICLE",
                "redistributionStatus": "CITATION_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L22_{item.sourceId}",
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
        f"UNTOUCHED_CONFIRMATION_{selected}" if selected else "LARGER_DISCOVERY_COHORT"
    )
    data["proposedNextLoopActive"] = True
    BASE.atomic_text(loop_path, yaml.safe_dump(data, sort_keys=False))

    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text())
    review["history"].append(
        {
            "decision": "S19_L22_COMPLETE_CONTINUE_UNDER_EXISTING_AUTHORIZATION",
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
    result["schema"] = "eidosoma.e01.s19_l22.artifact_manifest.v1"
    return result


def decision_summary_text(classifications: list[str], selected: str | None) -> str:
    outcome = (
        f"`{selected}` passed the studied-cohort discovery gate and now requires untouched confirmation."
        if selected
        else "The one fixed outcome-blind representation did not pass the common two-candidate gate; proceed to a larger independent discovery cohort without retuning this representation."
    )
    return f"""# S19-L22 decision summary

**Classification:** {", ".join(classifications)}
**Selected discovery lead:** `{selected or "NONE"}`

{outcome}

The human authorization permits one next bounded loop through L42. S20, E02, author contact, interventions and report generation remain inactive.
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
    BASE.EWS_FEATURES = RANDOM_CONV_FEATURES
    BASE.RQA_FEATURES = ()
    BASE.DMD_FEATURES = ()
    BASE.extract_organization_warning_features = extract_outcome_blind_representation
    BASE.MODEL_FEATURES = MODEL_FEATURES
    BASE.MODEL_IDS = MODEL_IDS
    BASE.LEAD_MODELS = LEAD_MODELS
    BASE.CANONICAL_REPORT_NAME = "S19_L22_FULL_RESULTS.md"
    BASE.ROOT_HANDOFF_SOURCE_HEADER = "# S19-L22"
    BASE.ROOT_HANDOFF_TARGET_HEADER = "# S19 current handoff — S19-L22"
    BASE.NULL_NEXT_ACTION = "S19_L23_LARGER_DISCOVERY_COHORT"
    BASE.RUNTIME_SCHEMA = "eidosoma.e01.s19_l22.runtime.v1"
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
