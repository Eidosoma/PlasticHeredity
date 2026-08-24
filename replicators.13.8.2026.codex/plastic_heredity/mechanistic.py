"""Two-stage prospective mechanistic-ablation workflow.

`prepare` reconstructs the retained development states, fits every registered
model, and seals an immutable bundle. `confirm` verifies that bundle before it
creates any new matrices, then evaluates the frozen suite on MECHCONF.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .experiment import (
    BranchBatch,
    StateCase,
    _candidate_indices,
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
from .mechanistic_features import (
    FEATURE_NAMES,
    H8_FEATURE_NAMES,
    H10_FEATURE_NAMES,
    INTERACTION_FEATURE_NAMES,
    STATE_ONLY_FEATURE_NAMES,
    MechanisticRawFeatures,
    extract_mechanistic_features,
)
from .mechanistic_metrics import compute_mechanistic_metrics
from .mechanistic_models import (
    CandidateRegistry,
    fit_candidate_registry,
    load_registries,
    predict_candidate_registry,
    save_registries,
)
from .models import predict_frozen_archive

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MECHCONF_MASTER_SEED = (
    "9329144f93f357f7c53f235edbbbfbc2aacd49a88e162499eebe77749fa950fd"
)
REGISTRATION_FORMAT = "plastic-heredity-mechanistic-registration-v1"
SOURCE_FILES = (
    "plastic_heredity/config.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/features.py",
    "plastic_heredity/mechanistic.py",
    "plastic_heredity/mechanistic_features.py",
    "plastic_heredity/mechanistic_metrics.py",
    "plastic_heredity/mechanistic_models.py",
    "plastic_heredity/mechanistic_plotting.py",
    "plastic_heredity/metrics.py",
    "plastic_heredity/models.py",
    "plastic_heredity/processes.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/simulator.py",
    "requirements-lock.txt",
)
SOURCE_RESULT_FILES = (
    "analysis_arrays.npz",
    "development_branches.csv.gz",
    "frozen_models.npz",
    "manifest.json",
    "model_contract.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_checksums(directory: Path) -> None:
    destination = directory / "SHA256SUMS"
    paths = sorted(
        path for path in directory.rglob("*") if path.is_file() and path != destination
    )
    destination.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(directory)}\n" for path in paths
        ),
        encoding="ascii",
    )


def verify_checksums(directory: Path) -> dict[str, bool]:
    checksum_path = directory / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"missing registration checksums: {checksum_path}")
    results: dict[str, bool] = {}
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = directory / relative
        results[relative] = path.is_file() and sha256_file(path) == expected
    if not results or not all(results.values()):
        failures = [name for name, valid in results.items() if not valid]
        raise ValueError(f"checksum verification failed: {failures}")
    return results


@contextmanager
def _atomic_destination(destination: Path) -> Iterator[Path]:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.preparing-", dir=destination.parent
        )
    )
    try:
        yield temporary
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _result_hashes(source_results: Path) -> dict[str, str]:
    return {
        name: sha256_file(source_results / name) for name in SOURCE_RESULT_FILES
    }


def _scaled5_experiment(source_results: Path) -> ExperimentConfig:
    manifest = json.loads((source_results / "manifest.json").read_text(encoding="utf-8"))
    experiment = ExperimentConfig.scaled5()
    expected = json.loads(json.dumps(experiment.to_dict()))
    if manifest["experiment"] != expected:
        raise ValueError("retained source does not match the current scaled5 contract")
    return experiment


def _verify_development_targets(
    path: Path, cases: list[StateCase], targets: np.ndarray
) -> int:
    expected_rows = targets.size
    observed = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for case_index, case in enumerate(cases):
            for branch in range(targets.shape[1]):
                try:
                    row = next(reader)
                except StopIteration as error:
                    raise ValueError("development branch table ended early") from error
                if (
                    row["state_id"] != case.state_id
                    or int(row["branch"]) != branch
                    or int(row["joint_break_run3"]) != int(targets[case_index, branch])
                ):
                    raise ValueError(
                        f"development target mismatch at {case.state_id}, branch {branch}"
                    )
                observed += 1
        try:
            extra = next(reader)
        except StopIteration:
            extra = None
        if extra is not None:
            raise ValueError("development branch table has unexpected extra rows")
    if observed != expected_rows:
        raise AssertionError("development target count mismatch")
    return observed


def _candidate_mask(cases: list[StateCase], candidate: str) -> np.ndarray:
    return np.asarray([case.candidate == candidate for case in cases], dtype=bool)


def _fit_registries(
    cases: list[StateCase], raw: MechanisticRawFeatures, targets: np.ndarray
) -> dict[str, CandidateRegistry]:
    registries: dict[str, CandidateRegistry] = {}
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        registries[candidate] = fit_candidate_registry(
            candidate,
            raw.selected(selected),
            targets[selected],
            pca_components=12,
            c=0.1,
        )
    return registries


def _development_prediction_audit(
    before: dict[str, CandidateRegistry],
    after: dict[str, CandidateRegistry],
    cases: list[StateCase],
    raw: MechanisticRawFeatures,
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        candidate_raw = raw.selected(selected)
        left = predict_candidate_registry(before[candidate], candidate_raw)
        right = predict_candidate_registry(after[candidate], candidate_raw)
        errors = {
            model: float(np.max(np.abs(left[model] - right[model]))) for model in left
        }
        audit[candidate] = {
            "states": int(selected.sum()),
            "maximum_absolute_errors": errors,
            "all_within_1e-12": all(value <= 1e-12 for value in errors.values()),
        }
    return audit


def _write_development_audit(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Mechanistic ablation development audit",
        "",
        "No development futures were resimulated. The retained 5× targets were used only after exact trajectory-derived feature reconstruction and branch-table validation.",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Reconstructed state/graph array exact | {audit['legacy_arrays']['state_graph']} |",
        f"| Reconstructed direct-history array exact | {audit['legacy_arrays']['history']} |",
        f"| Reconstructed beta-only array exact | {audit['legacy_arrays']['beta']} |",
        f"| Retained target rows validated | {audit['target_rows_validated']} |",
        f"| Direct trailing-run duplicate exact | {audit['direct_history_duplicate_exact']} |",
        f"| Portable registered predictions within 1e-12 | {audit['portable_predictions_exact']} |",
        "",
        "Raw compositions were not retained by the earlier campaign, so they cannot be compared byte-for-byte with a prior file. They are reconstructed from the unchanged seed contract, and all 399 retained state/history/beta coordinates match exactly.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_registration(source_results: Path, registration: Path) -> None:
    source_results = source_results.resolve()
    print("[prepare 1/6] Validating retained scaled5 inputs", flush=True)
    experiment = _scaled5_experiment(source_results)
    input_hashes = _result_hashes(source_results)

    print("[prepare 2/6] Reconstructing development trajectories and clocks", flush=True)
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
        raise ValueError(f"development reconstruction diverged: {exact_arrays}")
    target_rows = _verify_development_targets(
        source_results / "development_branches.csv.gz", cases, targets
    )

    print("[prepare 3/6] Constructing disjoint feature blocks", flush=True)
    raw = extract_mechanistic_features(cases, experiment)
    duplicate_exact = bool(np.array_equal(legacy.history[:, 4], legacy.history[:, 6]))
    if not duplicate_exact:
        raise AssertionError("registered direct-history duplication was not recovered")

    print("[prepare 4/6] Fitting candidate-separated registered models", flush=True)
    registries = _fit_registries(cases, raw, targets)

    with _atomic_destination(registration) as output:
        save_registries(
            output / "mechanistic_models.npz",
            output / "model_contract.json",
            registries,
        )
        reloaded = load_registries(
            output / "mechanistic_models.npz", output / "model_contract.json"
        )
        prediction_audit = _development_prediction_audit(
            registries, reloaded, cases, raw
        )
        portable_exact = all(
            item["all_within_1e-12"] for item in prediction_audit.values()
        )
        if not portable_exact:
            raise AssertionError("portable mechanistic archive changed predictions")

        print("[prepare 5/6] Writing reconstruction and audit artifacts", flush=True)
        shutil.copyfile(
            source_results / "frozen_models.npz",
            output / "legacy_frozen_models.npz",
        )
        shutil.copyfile(
            source_results / "model_contract.json",
            output / "legacy_model_contract.json",
        )
        np.savez_compressed(
            output / "reconstructed_development.npz",
            state_ids=np.asarray([case.state_id for case in cases]),
            candidates=np.asarray([case.candidate for case in cases]),
            matrix_ids=np.asarray([case.matrix_id for case in cases], dtype=np.int64),
            landmarks=np.asarray([case.landmark for case in cases], dtype=np.int64),
            compositions=np.vstack([case.snapshot.composition for case in cases]),
            previous_growth_steps=np.asarray(
                [case.snapshot.previous_growth_steps for case in cases], dtype=np.int64
            ),
            cumulative_growth_steps=np.asarray(
                [case.snapshot.cumulative_growth_steps for case in cases], dtype=np.int64
            ),
            h10=raw.h10,
        )
        development_audit = {
            "legacy_arrays": exact_arrays,
            "target_rows_validated": target_rows,
            "direct_history_duplicate_exact": duplicate_exact,
            "raw_compositions_previously_retained": False,
            "portable_prediction_audit": prediction_audit,
            "portable_predictions_exact": portable_exact,
            "states": len(cases),
            "matrices": experiment.development.matrices,
        }
        (output / "development_audit.json").write_text(
            json.dumps(_json_ready(development_audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_development_audit(output / "DEVELOPMENT_AUDIT.md", development_audit)

        confirmation_experiment = ExperimentConfig(
            development=CohortConfig(matrices=200, branches_per_state=32),
            confirmation=CohortConfig(matrices=200, branches_per_state=64),
            master_seed=MECHCONF_MASTER_SEED,
            bootstrap_repetitions=4_096,
            permutation_repetitions=4_096,
            regenerate_confirmation=True,
        )
        registration_payload: dict[str, Any] = {
            "format": REGISTRATION_FORMAT,
            "status": "sealed_before_confirmation",
            "scope": "prospective mechanistic attribution of plastic-heredity prediction",
            "development_source": {
                "path": str(source_results),
                "input_hashes": input_hashes,
                "experiment": experiment.to_dict(),
                "outcomes_resimulated": False,
            },
            "source_hashes": _source_hashes(),
            "feature_contract": {
                "h8": H8_FEATURE_NAMES,
                "h10": H10_FEATURE_NAMES,
                "state": STATE_ONLY_FEATURE_NAMES,
                "beta": FEATURE_NAMES["beta"],
                "interaction": INTERACTION_FEATURE_NAMES,
                "duplicate": FEATURE_NAMES["duplicate"],
                "interaction_residualization_base": ["h10", "state", "beta"],
            },
            "model_contract": {
                "pca_components_per_added_block": 12,
                "logistic_c": 0.1,
                "common_h10_unpenalized": True,
                "legacy_duplicate_control_penalizes_all_non_intercept_coefficients": True,
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
                "bootstrap_repetitions": 4_096,
                "randomization_repetitions": 4_096,
                "family": "3 contrasts x 2 candidates x 2 preassigned branch halves",
                "multiplicity": "Holm family-wise adjustment over 12 tests",
                "gate": (
                    "gain > 0, matrix-bootstrap lower 95% > 0, Holm p < 0.05 "
                    "for both candidates and both directions"
                ),
            },
            "confirmation_contract": {
                "cohort_name": "MECHCONF",
                "experiment": confirmation_experiment.to_dict(),
                "seed_domain_is_disjoint": (
                    MECHCONF_MASTER_SEED != experiment.master_seed
                ),
            },
        }
        registration_payload["registration_id"] = _canonical_digest(
            registration_payload
        )
        (output / "registration.json").write_text(
            json.dumps(_json_ready(registration_payload), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print("[prepare 6/6] Sealing registration checksums", flush=True)
        write_checksums(output)
    print(f"Sealed registration written to {registration.resolve()}", flush=True)


def verify_registration(registration: Path) -> dict[str, Any]:
    registration = registration.resolve()
    verify_checksums(registration)
    payload = json.loads((registration / "registration.json").read_text(encoding="utf-8"))
    if payload.get("format") != REGISTRATION_FORMAT:
        raise ValueError("unsupported registration format")
    registered_id = payload.pop("registration_id")
    if _canonical_digest(payload) != registered_id:
        raise ValueError("registration identifier mismatch")
    payload["registration_id"] = registered_id
    current_sources = _source_hashes()
    if payload["source_hashes"] != current_sources:
        changed = [
            name
            for name, digest in payload["source_hashes"].items()
            if current_sources.get(name) != digest
        ]
        raise ValueError(f"registered scientific source changed: {changed}")
    source_results = Path(payload["development_source"]["path"])
    if payload["development_source"]["input_hashes"] != _result_hashes(source_results):
        raise ValueError("retained development inputs changed after registration")
    if not payload["confirmation_contract"]["seed_domain_is_disjoint"]:
        raise ValueError("confirmation seed domain is not disjoint")
    return payload


def _confirmation_experiment(payload: dict[str, Any]) -> ExperimentConfig:
    registered = payload["confirmation_contract"]["experiment"]
    experiment = ExperimentConfig(
        development=CohortConfig(matrices=200, branches_per_state=32),
        confirmation=CohortConfig(matrices=200, branches_per_state=64),
        master_seed=MECHCONF_MASTER_SEED,
        bootstrap_repetitions=4_096,
        permutation_repetitions=4_096,
        regenerate_confirmation=True,
    )
    if json.loads(json.dumps(experiment.to_dict())) != registered:
        raise ValueError("confirmation implementation diverged from registration")
    return experiment


def _all_predictions(
    registration: Path,
    registries: dict[str, CandidateRegistry],
    cases: list[StateCase],
    raw: MechanisticRawFeatures,
    legacy: Any,
) -> dict[str, dict[str, np.ndarray]]:
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        values = predict_candidate_registry(
            registries[candidate], raw.selected(selected)
        )
        legacy_values = predict_frozen_archive(
            registration / "legacy_frozen_models.npz",
            candidate,
            legacy.state_graph[selected],
            legacy.history[selected],
            legacy.beta[selected],
        )
        values.update(
            {
                "legacy_prior": legacy_values["prior"],
                "legacy_h9": legacy_values["history"],
                "legacy_beta": legacy_values["beta"],
                "legacy_full": legacy_values["full"],
            }
        )
        for name, prediction in values.items():
            if (
                prediction.shape != (int(selected.sum()),)
                or not np.isfinite(prediction).all()
                or np.any((prediction < 0.0) | (prediction > 1.0))
            ):
                raise ValueError(f"invalid {candidate}/{name} predictions")
        predictions[candidate] = values
    return predictions


def _confirmation_state_table(
    cases: list[StateCase],
    batches: list[BranchBatch],
    predictions: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    labels = _stack_targets(batches)
    offsets = {candidate: 0 for candidate in CANDIDATES}
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        local = offsets[case.candidate]
        offsets[case.candidate] += 1
        split = labels.shape[1] // 2
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


def _prediction_readback_audit(
    path: Path,
    predictions: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    table = pd.read_csv(path, dtype={"candidate": str})
    table["candidate"] = table["candidate"].str.zfill(2)
    audit: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = table["candidate"] == candidate
        errors = {
            name: float(
                np.max(
                    np.abs(
                        values
                        - table.loc[selected, f"prediction_{name}"].to_numpy()
                    )
                )
            )
            for name, values in predictions[candidate].items()
        }
        audit[candidate] = {
            "maximum_absolute_errors": errors,
            "all_within_1e-12": all(value <= 1e-12 for value in errors.values()),
        }
    return audit


def _write_confirmation_report(
    output: Path, metrics: dict[str, Any], manifest: dict[str, Any]
) -> None:
    support = metrics["support"]
    supported = [name for name, value in support.items() if value]
    if support["interaction"]:
        outcome = (
            "The mass-free network-conditioned state interaction added prospective "
            "information beyond the additive history + state + network model."
        )
    elif support["network"]:
        outcome = (
            "Static catalytic-network information added prospectively beyond history "
            "and current composition, but the residual interaction did not pass."
        )
    elif support["state"]:
        outcome = (
            "Current composition/state added prospectively beyond all-clock history, "
            "but the stronger network attribution gates did not pass."
        )
    else:
        outcome = (
            "None of the three preregistered mechanistic attribution contrasts passed "
            "in both candidates and both branch-half directions."
        )
    lines = [
        "# Prospective mechanistic-ablation confirmation",
        "",
        "## Outcome",
        "",
        outcome,
        "",
        f"Supported registered contrasts: **{', '.join(supported) if supported else 'none'}**.",
        "",
        "## Primary gates",
        "",
        "| Contrast | Candidate | Half | Log-loss gain | 95% CI | Holm p | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["primary_tests"]:
        interval = row["log_loss_gain_ci95"]
        lines.append(
            f"| {row['contrast']} | {row['candidate']} | {row['direction']} | "
            f"{row['log_loss_gain']:.5f} | [{interval[0]:.5f}, {interval[1]:.5f}] | "
            f"{row['randomization_p_holm']:.5f} | {row['passes_gate']} |"
        )
    lines.extend(
        (
            "",
            "## Reliability and duplicate controls",
            "",
            "| Candidate | Split-half rho | Centered rho | Corrected duplicate gain A/B | Same-penalty duplicate gain A/B |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for candidate in CANDIDATES:
        corrected = [
            row["log_loss_gain"]
            for row in metrics["descriptive_tests"]
            if row["candidate"] == candidate
            and row["contrast"] == "corrected_duplicate"
        ]
        ridge = [
            row["log_loss_gain"]
            for row in metrics["descriptive_tests"]
            if row["candidate"] == candidate and row["contrast"] == "ridge_duplicate"
        ]
        item = metrics["candidates"][candidate]
        lines.append(
            f"| {candidate} | {item['branch_half_reliability']:.4f} | "
            f"{item['centered_branch_half_reliability']:.4f} | "
            f"{corrected[0]:.5f} / {corrected[1]:.5f} | "
            f"{ridge[0]:.5f} / {ridge[1]:.5f} |"
        )
    lines.extend(
        (
            "",
            "## Audit boundary",
            "",
            f"Registration `{manifest['registration_id']}` was sealed before MECHCONF generation. All {manifest['confirmation_futures']} confirmation futures were regenerated exactly: **{manifest['confirmation_replay_exact']}**.",
            "",
            "This supports only the narrow contrasts that passed. It remains a clean-room test of explicit candidate contracts, not an execution of the unavailable original-paper code.",
            "",
        )
    )
    (output / "MECHANISTIC_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_confirmation(
    registration: Path, output_directory: Path, workers: int | None = None
) -> None:
    registration = registration.resolve()
    print("[confirm 1/8] Verifying sealed registration and source hashes", flush=True)
    payload = verify_registration(registration)
    experiment = _confirmation_experiment(payload)
    workers = workers or max(1, min(os.cpu_count() or 1, 12))
    registries = load_registries(
        registration / "mechanistic_models.npz",
        registration / "model_contract.json",
    )

    print("[confirm 2/8] Generating untouched MECHCONF trajectories", flush=True)
    cases = build_cohort(experiment, "MECHCONF", experiment.confirmation)
    legacy = extract_features(cases, experiment)
    raw = extract_mechanistic_features(cases, experiment)
    predictions = _all_predictions(registration, registries, cases, raw, legacy)

    print("[confirm 3/8] Shooting 128,000 untouched futures", flush=True)
    batches = run_branches(
        cases, experiment, experiment.confirmation.branches_per_state, workers
    )
    first_digest = _digest_batches(batches)

    print("[confirm 4/8] Exactly regenerating all confirmation futures", flush=True)
    regenerated = run_branches(
        cases, experiment, experiment.confirmation.branches_per_state, workers
    )
    second_digest = _digest_batches(regenerated)
    if first_digest != second_digest:
        raise AssertionError("MECHCONF exact regeneration failed")

    print("[confirm 5/8] Computing paired matrix tests and Holm correction", flush=True)
    metrics = compute_mechanistic_metrics(cases, batches, predictions, experiment)
    process_summary = _process_prevalence(cases, batches, experiment)
    state_table = _confirmation_state_table(cases, batches, predictions)
    targets = _stack_targets(batches)
    extinct = int(sum(batch.completed_horizon.size - batch.completed_horizon.sum() for batch in batches))

    with _atomic_destination(output_directory) as output:
        print("[confirm 6/8] Writing complete audit artifacts", flush=True)
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
        pd.DataFrame(process_summary).to_csv(
            output / "process_summary.csv", index=False
        )
        state_table.to_csv(output / "confirmation_states.csv", index=False)
        _save_branch_table(output / "confirmation_branches.csv.gz", cases, batches)
        np.savez_compressed(
            output / "analysis_arrays.npz",
            confirmation_state_graph=legacy.state_graph,
            confirmation_history=legacy.history,
            confirmation_beta=legacy.beta,
            confirmation_h10=raw.h10,
            confirmation_state_block=raw.state,
            confirmation_beta_block=raw.beta,
            confirmation_interaction_block=raw.interaction,
            confirmation_duplicate_block=raw.duplicate,
            confirmation_compositions=np.vstack(
                [case.snapshot.composition for case in cases]
            ),
            confirmation_targets=targets,
        )
        prediction_audit = _prediction_readback_audit(
            output / "confirmation_states.csv", predictions
        )
        if not all(item["all_within_1e-12"] for item in prediction_audit.values()):
            raise AssertionError("saved predictions failed readback audit")
        (output / "prediction_audit.json").write_text(
            json.dumps(_json_ready(prediction_audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "clean_room": True,
            "scope": "prospective mechanistic attribution only",
            "registration_id": payload["registration_id"],
            "registration_path": str(registration),
            "registration_checksums_verified": True,
            "source_hashes_verified": True,
            "experiment": experiment.to_dict(),
            "cohort": "MECHCONF",
            "states": len(cases),
            "confirmation_futures": int(targets.size),
            "extinct_futures": extinct,
            "confirmation_digest_first": first_digest,
            "confirmation_digest_second": second_digest,
            "confirmation_replay_exact": True,
            "runtime": _runtime_manifest(),
            "support": metrics["support"],
            "prediction_readback_exact": True,
        }
        (output / "manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_confirmation_report(output, metrics, manifest)

        print("[confirm 7/8] Rendering mechanistic-only figures", flush=True)
        from .mechanistic_plotting import create_mechanistic_figures

        create_mechanistic_figures(metrics, state_table, output)
        print("[confirm 8/8] Sealing result checksums", flush=True)
        write_checksums(output)
    print(f"MECHCONF artifacts written to {output_directory.resolve()}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prospective mechanistic ablation of plastic-heredity prediction"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="fit and seal development-only models")
    prepare.add_argument("--source", type=Path, default=Path("results/scaled5"))
    prepare.add_argument(
        "--registration", type=Path, default=Path("results/mechanistic_registration")
    )
    confirm = commands.add_parser("confirm", help="run untouched MECHCONF evaluation")
    confirm.add_argument(
        "--registration", type=Path, default=Path("results/mechanistic_registration")
    )
    confirm.add_argument(
        "--output", type=Path, default=Path("results/mechanistic_confirmation")
    )
    confirm.add_argument("--workers", type=int, default=None)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "prepare":
        prepare_registration(arguments.source, arguments.registration)
    else:
        run_confirmation(arguments.registration, arguments.output, arguments.workers)


if __name__ == "__main__":
    main()
