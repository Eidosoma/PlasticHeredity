"""Prospective beta-completeness correction for mechanistic attribution.

The workflow is intentionally versioned and three-stage:

* ``prepare`` fits only retained scaled-development outcomes and seals a bundle;
* ``diagnose`` applies that sealed bundle to the already-seen MECHCONF cohort;
* ``confirm`` verifies the seal before generating untouched MECHCONF2 matrices.

No command modifies the v1 mechanistic registration or result directories.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .experiment import (
    StateCase,
    _digest_batches,
    _json_ready,
    _process_prevalence,
    _runtime_manifest,
    _save_branch_table,
    _stack_targets,
    build_cohort,
    extract_features,
    run_branches,
)
from .mechanistic import (
    MECHCONF_MASTER_SEED,
    REPOSITORY_ROOT,
    _atomic_destination,
    _canonical_digest,
    _candidate_mask,
    _prediction_readback_audit,
    _result_hashes,
    _scaled5_experiment,
    _verify_development_targets,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .mechanistic_v2_features import (
    FEATURE_NAMES,
    MechanisticV2RawFeatures,
    extract_mechanistic_v2_features,
    provenance_contract,
)
from .mechanistic_v2_metrics import (
    compute_mechanistic_v2_metrics,
    independently_recompute_primary_gains,
)
from .mechanistic_v2_models import (
    CV_FOLDS,
    RIDGE_LAMBDAS,
    CandidateRegistryV2,
    fit_candidate_registry_v2,
    load_registries_v2,
    predict_candidate_registry_v2,
    save_registries_v2,
)
from .memory import MEMORY_CONFIRM_MASTER_SEED

MECHCONF2_MASTER_SEED = (
    "4b80c4056081287f4ad0c4359f9229cba2cb803124278b6961ca8edbaf86a086"
)
REGISTRATION_FORMAT = "plastic-heredity-beta-completeness-registration-v2"
SOURCE_FILES = (
    "plastic_heredity/config.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/features.py",
    "plastic_heredity/mechanistic.py",
    "plastic_heredity/mechanistic_features.py",
    "plastic_heredity/mechanistic_metrics.py",
    "plastic_heredity/mechanistic_v2.py",
    "plastic_heredity/mechanistic_v2_features.py",
    "plastic_heredity/mechanistic_v2_metrics.py",
    "plastic_heredity/mechanistic_v2_models.py",
    "plastic_heredity/metrics.py",
    "plastic_heredity/processes.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/simulator.py",
    "requirements-lock.txt",
)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _confirmation_experiment(seed: str = MECHCONF2_MASTER_SEED) -> ExperimentConfig:
    return ExperimentConfig(
        development=CohortConfig(matrices=200, branches_per_state=32),
        confirmation=CohortConfig(matrices=200, branches_per_state=64),
        master_seed=seed,
        bootstrap_repetitions=4_096,
        permutation_repetitions=4_096,
        regenerate_confirmation=True,
    )


def _bundle_checksum_digest(path: Path) -> str:
    verify_checksums(path)
    return sha256_file(path / "SHA256SUMS")


def _fit_registries(
    cases: list[StateCase],
    raw: MechanisticV2RawFeatures,
    targets: np.ndarray,
) -> dict[str, CandidateRegistryV2]:
    registries: dict[str, CandidateRegistryV2] = {}
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        matrix_ids = np.asarray(
            [case.matrix_id for case, keep in zip(cases, selected) if keep],
            dtype=np.int64,
        )
        registries[candidate] = fit_candidate_registry_v2(
            candidate,
            raw.selected(selected),
            targets[selected],
            matrix_ids,
        )
    return registries


def _predictions(
    registries: dict[str, CandidateRegistryV2],
    cases: list[StateCase],
    raw: MechanisticV2RawFeatures,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        values = predict_candidate_registry_v2(
            registries[candidate], raw.selected(selected)
        )
        expected = int(selected.sum())
        for name, prediction in values.items():
            if (
                prediction.shape != (expected,)
                or not np.isfinite(prediction).all()
                or np.any((prediction < 0.0) | (prediction > 1.0))
            ):
                raise ValueError(f"invalid v2 prediction {candidate}/{name}")
        result[candidate] = values
    return result


def _portable_prediction_audit(
    before: dict[str, CandidateRegistryV2],
    after: dict[str, CandidateRegistryV2],
    cases: list[StateCase],
    raw: MechanisticV2RawFeatures,
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        candidate_raw = raw.selected(selected)
        left = predict_candidate_registry_v2(before[candidate], candidate_raw)
        right = predict_candidate_registry_v2(after[candidate], candidate_raw)
        errors = {
            name: float(np.max(np.abs(left[name] - right[name]))) for name in left
        }
        audit[candidate] = {
            "maximum_absolute_errors": errors,
            "all_within_1e-12": all(value <= 1e-12 for value in errors.values()),
        }
    return audit


def _model_summary(registries: dict[str, CandidateRegistryV2]) -> dict[str, Any]:
    return {
        candidate: {
            "raw_features": {
                block: len(FEATURE_NAMES[block]) for block in FEATURE_NAMES
            },
            "retained_features": {
                block: registry.transforms[block].output_features
                for block in registry.transforms
            },
            "selected_lambdas": registry.selected_lambdas,
            "cv_scores": registry.cv_scores,
            "uses_pca": False,
        }
        for candidate, registry in registries.items()
    }


def prepare_registration(
    source_results: Path,
    diagnostic_source: Path,
    registration: Path,
) -> None:
    source_results = source_results.resolve()
    diagnostic_source = diagnostic_source.resolve()
    print("[v2 prepare 1/7] Validating retained development and diagnostic bundles", flush=True)
    experiment = _scaled5_experiment(source_results)
    diagnostic_checksum = _bundle_checksum_digest(diagnostic_source)

    print("[v2 prepare 2/7] Reconstructing development trajectories and targets", flush=True)
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, "VALI", experiment.development)
        legacy = extract_features(cases, experiment)
    with np.load(source_results / "analysis_arrays.npz") as retained:
        targets = retained["development_targets"].copy()
        exact_arrays = {
            "state_graph": bool(
                np.array_equal(legacy.state_graph, retained["development_state_graph"])
            ),
            "history": bool(
                np.array_equal(legacy.history, retained["development_history"])
            ),
            "beta": bool(np.array_equal(legacy.beta, retained["development_beta"])),
        }
    if not all(exact_arrays.values()):
        raise ValueError(f"v2 development reconstruction diverged: {exact_arrays}")
    target_rows = _verify_development_targets(
        source_results / "development_branches.csv.gz", cases, targets
    )

    print("[v2 prepare 3/7] Building provenance-selected no-PCA blocks", flush=True)
    with threadpool_limits(limits=1):
        raw = extract_mechanistic_v2_features(cases, experiment)

    print("[v2 prepare 4/7] Selecting penalties by whole-matrix development CV", flush=True)
    with threadpool_limits(limits=1):
        registries = _fit_registries(cases, raw, targets)

    with _atomic_destination(registration) as output:
        print("[v2 prepare 5/7] Writing portable models and complete feature contract", flush=True)
        save_registries_v2(
            output / "models.npz", output / "model_contract.json", registries
        )
        reloaded = load_registries_v2(
            output / "models.npz", output / "model_contract.json"
        )
        prediction_audit = _portable_prediction_audit(
            registries, reloaded, cases, raw
        )
        if not all(item["all_within_1e-12"] for item in prediction_audit.values()):
            raise AssertionError("v2 portable model archive changed predictions")
        provenance = provenance_contract()
        (output / "feature_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            output / "reconstructed_development.npz",
            state_ids=np.asarray([case.state_id for case in cases]),
            candidates=np.asarray([case.candidate for case in cases]),
            matrix_ids=np.asarray([case.matrix_id for case in cases], dtype=np.int64),
            landmarks=np.asarray([case.landmark for case in cases], dtype=np.int64),
            compositions=np.vstack([case.snapshot.composition for case in cases]),
            h10=raw.h10,
            state_block=raw.state,
            beta_block=raw.beta,
            interaction_block=raw.interaction,
            targets=targets,
        )
        summary = _model_summary(registries)
        audit = {
            "legacy_arrays_exact": exact_arrays,
            "target_rows_validated": target_rows,
            "states": len(cases),
            "matrices": experiment.development.matrices,
            "feature_provenance_complete": all(
                len(provenance[block]) == len(FEATURE_NAMES[block])
                for block in FEATURE_NAMES
            ),
            "portable_prediction_audit": prediction_audit,
            "portable_predictions_exact": True,
            "models": summary,
        }
        (output / "development_audit.json").write_text(
            json.dumps(_json_ready(audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# Beta-completeness development audit",
            "",
            "The retained fivefold development outcomes were used without resimulation. All legacy coordinates and 64,000 target rows were reconstructed before fitting.",
            "",
            "| Check | Result |",
            "|---|---:|",
            f"| Legacy state/history/beta arrays exact | {all(exact_arrays.values())} |",
            f"| Retained target rows validated | {target_rows} |",
            f"| Provenance records cover every raw feature | {audit['feature_provenance_complete']} |",
            "| Added-block PCA components | 0 |",
            "| Whole-matrix CV folds | 5 |",
            "| Portable predictions within 1e-12 | True |",
            "",
            "## Registered dimensions and penalties",
            "",
            "| Candidate | H10 | State | Beta | Interaction | Lambda state/beta/interaction |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for candidate, item in summary.items():
            retained = item["retained_features"]
            selected = item["selected_lambdas"]
            lines.append(
                f"| {candidate} | {retained['h10']} | {retained['state']} | "
                f"{retained['beta']} | {retained['interaction']} | "
                f"{selected['state']:g} / {selected['beta']:g} / {selected['interaction']:g} |"
            )
        (output / "DEVELOPMENT_AUDIT.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        print("[v2 prepare 6/7] Sealing prospective protocol and disjoint seed", flush=True)
        confirmation = _confirmation_experiment()
        seed_domains = {
            "scaled5": experiment.master_seed,
            "MECHCONF": MECHCONF_MASTER_SEED,
            "MEMCONF": MEMORY_CONFIRM_MASTER_SEED,
            "MECHCONF2": MECHCONF2_MASTER_SEED,
        }
        payload: dict[str, Any] = {
            "format": REGISTRATION_FORMAT,
            "status": "sealed_before_posthoc_diagnostic_and_confirmation",
            "scope": "prospective beta-completeness correction",
            "development_source": {
                "path": str(source_results),
                "input_hashes": _result_hashes(source_results),
                "experiment": experiment.to_dict(),
                "outcomes_resimulated": False,
            },
            "posthoc_diagnostic_source": {
                "path": str(diagnostic_source),
                "sha256sums_digest": diagnostic_checksum,
                "may_not_change_registration": True,
            },
            "source_hashes": _source_hashes(),
            "feature_contract": {
                "provenance_file": "feature_provenance.json",
                "provenance_digest": sha256_file(output / "feature_provenance.json"),
                "raw_counts": {block: len(names) for block, names in FEATURE_NAMES.items()},
                "selection": "dependency flags, followed by development-only constant and exact affine-duplicate removal",
                "beta_panel": "legacy beta-only invariant coordinates plus fixed threshold-free distribution, strength, reciprocity, asymmetry, singular-spectrum, entropy, and concentration descriptors",
                "interaction_residualization_base": ["h10", "state", "beta"],
                "pca_components": 0,
            },
            "model_contract": {
                "structure": "sequential frozen-offset ridge",
                "sequence": ["h10", "state", "beta", "interaction"],
                "ridge_lambda_grid": RIDGE_LAMBDAS,
                "cv_folds": CV_FOLDS,
                "cv_split": "matrix_id modulo 5",
                "cv_tie_break": "largest lambda within 1e-12 of minimum loss",
                "baseline_h10_penalty": 0.0,
                "objective_scale": "mean negative log likelihood per Bernoulli trial",
                "uses_pca": False,
                "selected_lambdas": {
                    candidate: registry.selected_lambdas
                    for candidate, registry in registries.items()
                },
            },
            "statistical_contract": {
                "primary_contrasts": {
                    "state": ["h10", "h10_state"],
                    "network": ["h10_state", "h10_state_beta"],
                    "interaction": [
                        "h10_state_beta",
                        "h10_state_beta_interaction",
                    ],
                },
                "family": "3 contrasts x 2 candidates x 2 branch halves",
                "bootstrap_repetitions": 4_096,
                "randomization_repetitions": 4_096,
                "multiplicity": "Holm family-wise adjustment over 12 tests",
                "gate": "gain > 0, matrix-bootstrap lower 95% > 0, Holm p < 0.05 in both candidates and both halves",
                "null_language": "no incremental signal detected with the frozen comprehensive panel",
            },
            "confirmation_contract": {
                "cohort_name": "MECHCONF2",
                "experiment": confirmation.to_dict(),
                "seed_domains": seed_domains,
                "all_seed_domains_unique": len(set(seed_domains.values())) == len(seed_domains),
            },
        }
        if not payload["confirmation_contract"]["all_seed_domains_unique"]:
            raise AssertionError("MECHCONF2 seed domain collides with an earlier campaign")
        payload["registration_id"] = _canonical_digest(payload)
        (output / "registration.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("[v2 prepare 7/7] Writing immutable checksums", flush=True)
        write_checksums(output)
    print(f"V2 registration sealed at {registration.resolve()}", flush=True)


def verify_registration(registration: Path) -> dict[str, Any]:
    registration = registration.resolve()
    verify_checksums(registration)
    payload = json.loads((registration / "registration.json").read_text(encoding="utf-8"))
    if payload.get("format") != REGISTRATION_FORMAT:
        raise ValueError("unsupported beta-completeness registration")
    registered_id = payload.pop("registration_id")
    if _canonical_digest(payload) != registered_id:
        raise ValueError("beta-completeness registration identifier mismatch")
    payload["registration_id"] = registered_id
    current_sources = _source_hashes()
    if payload["source_hashes"] != current_sources:
        changed = [
            name
            for name, digest in payload["source_hashes"].items()
            if current_sources.get(name) != digest
        ]
        raise ValueError(f"registered v2 scientific source changed: {changed}")
    source = Path(payload["development_source"]["path"])
    if payload["development_source"]["input_hashes"] != _result_hashes(source):
        raise ValueError("retained v2 development inputs changed")
    diagnostic = Path(payload["posthoc_diagnostic_source"]["path"])
    if (
        _bundle_checksum_digest(diagnostic)
        != payload["posthoc_diagnostic_source"]["sha256sums_digest"]
    ):
        raise ValueError("registered MECHCONF diagnostic source changed")
    if sha256_file(registration / "feature_provenance.json") != payload["feature_contract"][
        "provenance_digest"
    ]:
        raise ValueError("feature provenance digest mismatch")
    if not payload["confirmation_contract"]["all_seed_domains_unique"]:
        raise ValueError("confirmation seed domains are not disjoint")
    current_confirmation = json.loads(
        json.dumps(_confirmation_experiment().to_dict())
    )
    if payload["confirmation_contract"]["experiment"] != current_confirmation:
        raise ValueError("MECHCONF2 implementation diverged from registration")
    return payload


def _state_table(
    cases: list[StateCase],
    labels: np.ndarray,
    predictions: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    offsets = {candidate: 0 for candidate in CANDIDATES}
    split = labels.shape[1] // 2
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        local = offsets[case.candidate]
        offsets[case.candidate] += 1
        row: dict[str, Any] = {
            "state_id": case.state_id,
            "cohort": case.cohort,
            "candidate": case.candidate,
            "matrix_id": case.matrix_id,
            "landmark": case.landmark,
            "mass": int(case.snapshot.composition.sum()),
            "previous_growth_steps": case.snapshot.previous_growth_steps,
            "cumulative_growth_steps": case.snapshot.cumulative_growth_steps,
            "q_all": float(labels[index].mean()),
            "q_half_A": float(labels[index, :split].mean()),
            "q_half_B": float(labels[index, split:].mean()),
        }
        for name, values in predictions[case.candidate].items():
            row[f"prediction_{name}"] = float(values[local])
        rows.append(row)
    return pd.DataFrame(rows)


def _read_predictions(
    path: Path, model_names: tuple[str, ...]
) -> dict[str, dict[str, np.ndarray]]:
    table = pd.read_csv(path, dtype={"candidate": str})
    table["candidate"] = table["candidate"].str.zfill(2)
    return {
        candidate: {
            name: table.loc[
                table["candidate"] == candidate, f"prediction_{name}"
            ].to_numpy(dtype=np.float64)
            for name in model_names
        }
        for candidate in CANDIDATES
    }


def _write_result_report(
    output: Path,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    registries: dict[str, CandidateRegistryV2],
    diagnostic: bool,
) -> None:
    support = metrics["support"]
    title = (
        "# Post-hoc beta-completeness diagnostic"
        if diagnostic
        else "# Prospective beta-completeness confirmation"
    )
    boundary = (
        "This applies a previously sealed correction to an already-seen cohort and cannot support a new prospective claim."
        if diagnostic
        else "The registration was sealed before any MECHCONF2 matrix was generated."
    )
    statements = []
    statements.append(
        "Current composition added information beyond all-clock history."
        if support["state"]
        else "The registered current-composition contrast did not pass all gates."
    )
    statements.append(
        "Static beta structure added information beyond history and current composition."
        if support["network"]
        else "No incremental static-beta signal was detected with the frozen comprehensive threshold-free panel."
    )
    statements.append(
        "The beta-conditioned current-state block added information beyond the comprehensive additive baseline."
        if support["interaction"]
        else "The beta-conditioned current-state block did not pass beyond the comprehensive additive baseline."
    )
    lines = [
        title,
        "",
        "## Outcome",
        "",
        *statements,
        "",
        "## Primary registered contrasts",
        "",
        "| Contrast | Candidate | Half | Log-loss gain | 95% CI | Holm p | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["primary_tests"]:
        interval = row["log_loss_gain_ci95"]
        lines.append(
            f"| {row['contrast']} | {row['candidate']} | {row['direction']} | "
            f"{row['log_loss_gain']:.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] | "
            f"{row['randomization_p_holm']:.6f} | {row['passes_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen representation",
            "",
            "| Candidate | Retained state | Retained beta | Retained interaction | Lambda state/beta/interaction |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for candidate, registry in registries.items():
        count = {
            block: registry.transforms[block].output_features
            for block in ("state", "beta", "interaction")
        }
        selected = registry.selected_lambdas
        lines.append(
            f"| {candidate} | {count['state']} | {count['beta']} | {count['interaction']} | "
            f"{selected['state']:g} / {selected['beta']:g} / {selected['interaction']:g} |"
        )
    lines.extend(
        [
            "",
            "No added block uses PCA. The beta panel is threshold-free and includes the complete normalized singular spectrum. A null beta result is representation-specific and is not proof that beta is generally uninformative.",
            "",
            "## Audit boundary",
            "",
            boundary,
            "",
            f"Registration: `{manifest['registration_id']}`. Exact future replay: **{manifest.get('confirmation_replay_exact', 'not applicable')}**. Independent gain recomputation within 1e-14: **{manifest['metric_recomputation_exact']}**.",
            "",
            "These are predictive, simulator-specific contrasts. They do not establish a causal mechanism or biological chemistry.",
            "",
        ]
    )
    filename = "DIAGNOSTIC_RESULTS.md" if diagnostic else "BETA_COMPLETE_RESULTS.md"
    (output / filename).write_text("\n".join(lines), encoding="utf-8")


def run_diagnostic(
    registration: Path, output_directory: Path
) -> None:
    registration = registration.resolve()
    print("[v2 diagnose 1/5] Verifying sealed v2 registration", flush=True)
    payload = verify_registration(registration)
    source = Path(payload["posthoc_diagnostic_source"]["path"])
    old_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    experiment = _confirmation_experiment(MECHCONF_MASTER_SEED)
    registered_old_experiment = json.loads(json.dumps(experiment.to_dict()))
    if old_manifest["experiment"] != registered_old_experiment:
        raise ValueError("old MECHCONF contract does not match diagnostic declaration")

    print("[v2 diagnose 2/5] Reconstructing and validating old MECHCONF states", flush=True)
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, "MECHCONF", experiment.confirmation)
        legacy = extract_features(cases, experiment)
    with np.load(source / "analysis_arrays.npz") as retained:
        labels = retained["confirmation_targets"].copy()
        exact = {
            "state_graph": bool(
                np.array_equal(legacy.state_graph, retained["confirmation_state_graph"])
            ),
            "history": bool(
                np.array_equal(legacy.history, retained["confirmation_history"])
            ),
            "beta": bool(np.array_equal(legacy.beta, retained["confirmation_beta"])),
            "compositions": bool(
                np.array_equal(
                    np.vstack([case.snapshot.composition for case in cases]),
                    retained["confirmation_compositions"],
                )
            ),
        }
    if not all(exact.values()):
        raise ValueError(f"old MECHCONF diagnostic reconstruction diverged: {exact}")

    print("[v2 diagnose 3/5] Applying frozen comprehensive models", flush=True)
    with threadpool_limits(limits=1):
        raw = extract_mechanistic_v2_features(cases, experiment)
    registries = load_registries_v2(
        registration / "models.npz", registration / "model_contract.json"
    )
    predictions = _predictions(registries, cases, raw)
    metrics = compute_mechanistic_v2_metrics(
        cases, labels, predictions, experiment, "MECHCONF.posthoc_v2"
    )
    audit = independently_recompute_primary_gains(cases, labels, predictions, metrics)
    if not audit["all_within_1e-14"]:
        raise AssertionError("diagnostic metric recomputation failed")

    print("[v2 diagnose 4/5] Writing explicitly post-hoc result bundle", flush=True)
    with _atomic_destination(output_directory) as output:
        state_table = _state_table(cases, labels, predictions)
        state_table.to_csv(output / "diagnostic_states.csv", index=False)
        (output / "metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(metrics["primary_tests"]).to_csv(
            output / "primary_tests.csv", index=False
        )
        pd.DataFrame(metrics["descriptive_tests"]).to_csv(
            output / "descriptive_tests.csv", index=False
        )
        (output / "reconstruction_audit.json").write_text(
            json.dumps({"legacy_arrays_exact": exact}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "metric_recomputation_audit.json").write_text(
            json.dumps(_json_ready(audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "clean_room": True,
            "status": "posthoc_diagnostic_only",
            "cohort": "MECHCONF",
            "registration_id": payload["registration_id"],
            "registration_checksums_verified": True,
            "source_hashes_verified": True,
            "source_bundle_checksums_verified": True,
            "states": len(cases),
            "confirmation_futures_reused": int(labels.size),
            "outcomes_resimulated": False,
            "support_descriptive_only": metrics["support"],
            "metric_recomputation_exact": True,
            "runtime": _runtime_manifest(),
        }
        (output / "manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_result_report(output, metrics, manifest, registries, diagnostic=True)
        print("[v2 diagnose 5/5] Sealing diagnostic checksums", flush=True)
        write_checksums(output)
    print(f"V2 diagnostic written to {output_directory.resolve()}", flush=True)


def run_confirmation(
    registration: Path,
    output_directory: Path,
    workers: int | None = None,
) -> None:
    registration = registration.resolve()
    print("[v2 confirm 1/8] Verifying sealed registration and source hashes", flush=True)
    payload = verify_registration(registration)
    experiment = _confirmation_experiment()
    workers = workers or max(1, min(os.cpu_count() or 1, 12))
    registries = load_registries_v2(
        registration / "models.npz", registration / "model_contract.json"
    )

    print("[v2 confirm 2/8] Generating untouched MECHCONF2 trajectories", flush=True)
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, "MECHCONF2", experiment.confirmation)
        raw = extract_mechanistic_v2_features(cases, experiment)
    predictions = _predictions(registries, cases, raw)

    print("[v2 confirm 3/8] Shooting 128,000 untouched futures", flush=True)
    batches = run_branches(
        cases, experiment, experiment.confirmation.branches_per_state, workers
    )
    first_digest = _digest_batches(batches)
    labels = _stack_targets(batches)

    print("[v2 confirm 4/8] Exactly replaying all 128,000 futures", flush=True)
    replayed = run_branches(
        cases, experiment, experiment.confirmation.branches_per_state, workers
    )
    second_digest = _digest_batches(replayed)
    if first_digest != second_digest:
        raise AssertionError("MECHCONF2 exact replay failed")

    print("[v2 confirm 5/8] Computing the sealed 12-test family", flush=True)
    metrics = compute_mechanistic_v2_metrics(
        cases, labels, predictions, experiment, "MECHCONF2"
    )
    process_summary = _process_prevalence(cases, batches, experiment)
    extinct = int(
        sum(
            batch.completed_horizon.size - batch.completed_horizon.sum()
            for batch in batches
        )
    )

    with _atomic_destination(output_directory) as output:
        print("[v2 confirm 6/8] Writing complete confirmation artifacts", flush=True)
        state_table = _state_table(cases, labels, predictions)
        state_table.to_csv(output / "confirmation_states.csv", index=False)
        _save_branch_table(output / "confirmation_branches.csv.gz", cases, batches)
        pd.DataFrame(process_summary).to_csv(
            output / "process_summary.csv", index=False
        )
        (output / "metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(metrics["primary_tests"]).to_csv(
            output / "primary_tests.csv", index=False
        )
        pd.DataFrame(metrics["descriptive_tests"]).to_csv(
            output / "descriptive_tests.csv", index=False
        )
        np.savez_compressed(
            output / "analysis_arrays.npz",
            confirmation_h10=raw.h10,
            confirmation_state_block=raw.state,
            confirmation_beta_block=raw.beta,
            confirmation_interaction_block=raw.interaction,
            confirmation_compositions=np.vstack(
                [case.snapshot.composition for case in cases]
            ),
            confirmation_targets=labels,
        )
        prediction_audit = _prediction_readback_audit(
            output / "confirmation_states.csv", predictions
        )
        if not all(item["all_within_1e-12"] for item in prediction_audit.values()):
            raise AssertionError("saved v2 predictions failed readback")
        model_names = tuple(next(iter(predictions.values())).keys())
        saved_predictions = _read_predictions(
            output / "confirmation_states.csv", model_names
        )
        with np.load(output / "analysis_arrays.npz") as archive:
            saved_labels = archive["confirmation_targets"].copy()
        metric_audit = independently_recompute_primary_gains(
            cases, saved_labels, saved_predictions, metrics
        )
        if not metric_audit["all_within_1e-14"]:
            raise AssertionError("saved v2 metric recomputation failed")
        (output / "prediction_audit.json").write_text(
            json.dumps(_json_ready(prediction_audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "metric_recomputation_audit.json").write_text(
            json.dumps(_json_ready(metric_audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "clean_room": True,
            "scope": "prospective beta-completeness correction",
            "cohort": "MECHCONF2",
            "registration_id": payload["registration_id"],
            "registration_path": str(registration),
            "registration_checksums_verified": True,
            "source_hashes_verified": True,
            "experiment": experiment.to_dict(),
            "states": len(cases),
            "confirmation_futures": int(labels.size),
            "extinct_futures": extinct,
            "confirmation_digest_first": first_digest,
            "confirmation_digest_second": second_digest,
            "confirmation_replay_exact": True,
            "prediction_readback_exact": True,
            "metric_recomputation_exact": True,
            "support": metrics["support"],
            "models": _model_summary(registries),
            "runtime": _runtime_manifest(),
        }
        (output / "manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_result_report(output, metrics, manifest, registries, diagnostic=False)
        print("[v2 confirm 7/8] Verifying result report and readback audits", flush=True)
        if not (output / "BETA_COMPLETE_RESULTS.md").is_file():
            raise AssertionError("v2 result report missing")
        print("[v2 confirm 8/8] Sealing result checksums", flush=True)
        write_checksums(output)
    print(f"MECHCONF2 artifacts written to {output_directory.resolve()}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prospective beta-completeness correction"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="fit and seal development-only v2 models")
    prepare.add_argument("--source", type=Path, default=Path("results/scaled5"))
    prepare.add_argument(
        "--diagnostic-source",
        type=Path,
        default=Path("results/mechanistic_confirmation"),
    )
    prepare.add_argument(
        "--registration",
        type=Path,
        default=Path("results/beta_complete_registration"),
    )
    diagnose = commands.add_parser("diagnose", help="score old MECHCONF post hoc")
    diagnose.add_argument(
        "--registration",
        type=Path,
        default=Path("results/beta_complete_registration"),
    )
    diagnose.add_argument(
        "--output", type=Path, default=Path("results/beta_complete_diagnostic")
    )
    confirm = commands.add_parser("confirm", help="run untouched MECHCONF2")
    confirm.add_argument(
        "--registration",
        type=Path,
        default=Path("results/beta_complete_registration"),
    )
    confirm.add_argument(
        "--output", type=Path, default=Path("results/beta_complete_confirmation")
    )
    confirm.add_argument("--workers", type=int, default=None)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "prepare":
        prepare_registration(
            arguments.source, arguments.diagnostic_source, arguments.registration
        )
    elif arguments.command == "diagnose":
        run_diagnostic(arguments.registration, arguments.output)
    else:
        run_confirmation(arguments.registration, arguments.output, arguments.workers)


if __name__ == "__main__":
    main()
