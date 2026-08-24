"""Prospectively registered tests of post-break inheritance dependence.

The workflow is intentionally separate from the confirmed L54 risk predictor:

``diagnose``
    Regenerate the existing scaled confirmation futures and quantify the
    reviewer-identified IID fitting-support mismatch.
``prepare``
    Seal the sequence, model, cross-fit, inference, and new-cohort contracts.
``confirm``
    Verify the seal, generate untouched 32-fission futures, and evaluate the
    registered IID -> Markov -> duration-aware semi-Markov hierarchy.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .experiment import (
    PROCESS_COLUMNS,
    StateCase,
    _digest_batches,
    _json_ready,
    _runtime_manifest,
    build_cohort,
)
from .mechanistic import (
    MECHCONF_MASTER_SEED,
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .memory_metrics import (
    CandidateCrossfitScores,
    SequenceRecord,
    calibration_rows,
    compute_memory_metrics,
    crossfit_memory_models,
    score_archive_arrays,
    sequence_count_rows,
)
from .processes import ProcessOutcome, evaluate_process
from .seeds import derive_seed
from .simulator import simulate_future_absorbing

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MEMORY_CONFIRM_MASTER_SEED = (
    "25cd8e32e8e2176b36c31ecf2486827a523a9d49ab18eb6e39cd74237f153868"
)
MEMORY_REGISTRATION_FORMAT = "plastic-heredity-memory-registration-v1"
MEMORY_CONFIRMATION_HORIZON = 32
MEMORY_MATRICES = 200
MEMORY_BRANCHES = 64
MEMORY_LANDMARKS = (20, 35, 50, 65, 80)
MEMORY_REPETITIONS = 4_096

SCIENTIFIC_SOURCE_FILES = (
    "plastic_heredity/config.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/mechanistic.py",
    "plastic_heredity/memory.py",
    "plastic_heredity/memory_metrics.py",
    "plastic_heredity/memory_models.py",
    "plastic_heredity/memory_plotting.py",
    "plastic_heredity/metrics.py",
    "plastic_heredity/processes.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/simulator.py",
    "pyproject.toml",
    "requirements-lock.txt",
)
DIAGNOSTIC_SOURCE_FILES = (
    "manifest.json",
    "confirmation_branches.csv.gz",
    "analysis_arrays.npz",
    "SHA256SUMS",
)


@dataclass
class MemoryBatch:
    target: NDArray[np.int8]
    process: NDArray[np.float64]
    completed_horizon: NDArray[np.int8]
    observed_fissions: NDArray[np.int16]
    first_break_index: NDArray[np.int16]
    post_break_sequences: tuple[tuple[bool, ...], ...]


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SCIENTIFIC_SOURCE_FILES}


def _selected_hashes(directory: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names}


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _memory_confirmation_experiment() -> ExperimentConfig:
    cohort = CohortConfig(
        matrices=MEMORY_MATRICES,
        branches_per_state=MEMORY_BRANCHES,
        landmarks=MEMORY_LANDMARKS,
    )
    return ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=MEMORY_CONFIRMATION_HORIZON,
        bootstrap_repetitions=MEMORY_REPETITIONS,
        permutation_repetitions=MEMORY_REPETITIONS,
        regenerate_confirmation=True,
        master_seed=MEMORY_CONFIRM_MASTER_SEED,
    )


def post_break_sequence(inheritance: Iterable[bool]) -> tuple[int, tuple[bool, ...]]:
    """Return zero-based first-break index and symbols strictly after it."""

    values = tuple(bool(value) for value in inheritance)
    try:
        first_break = values.index(False)
    except ValueError:
        return -1, ()
    return first_break, values[first_break + 1 :]


def _memory_branch_worker(
    args: tuple[StateCase, ExperimentConfig, int]
) -> MemoryBatch:
    case, experiment, branches = args
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
    except Exception:  # pragma: no cover - optional runtime guard
        limiter = None
    try:
        target = np.empty(branches, dtype=np.int8)
        process = np.empty((branches, len(PROCESS_COLUMNS)), dtype=np.float64)
        completed = np.empty(branches, dtype=np.int8)
        observed = np.empty(branches, dtype=np.int16)
        first_break = np.empty(branches, dtype=np.int16)
        sequences: list[tuple[bool, ...]] = []
        contract = CANDIDATES[case.candidate]
        for branch in range(branches):
            rng = np.random.default_rng(
                derive_seed(
                    experiment.master_seed,
                    f"{case.cohort}.future",
                    case.candidate,
                    case.matrix_id,
                    case.landmark,
                    branch,
                )
            )
            records, completed_horizon = simulate_future_absorbing(
                case.snapshot,
                case.beta,
                experiment.gard,
                contract,
                experiment.horizon,
                rng,
            )
            outcome: ProcessOutcome = evaluate_process(
                records, experiment.gard.inheritance_threshold
            )
            inherited = tuple(
                record.h > experiment.gard.inheritance_threshold for record in records
            )
            break_index, sequence = post_break_sequence(inherited)
            values = outcome.to_dict()
            target[branch] = int(outcome.joint_break_run3)
            process[branch] = [float(values[name]) for name in PROCESS_COLUMNS]
            completed[branch] = int(completed_horizon)
            observed[branch] = len(records)
            first_break[branch] = break_index
            sequences.append(sequence)
        return MemoryBatch(
            target=target,
            process=process,
            completed_horizon=completed,
            observed_fissions=observed,
            first_break_index=first_break,
            post_break_sequences=tuple(sequences),
        )
    finally:
        if limiter is not None:
            limiter.restore_original_limits()


def run_memory_branches(
    cases: list[StateCase],
    experiment: ExperimentConfig,
    branches: int,
    workers: int,
) -> list[MemoryBatch]:
    arguments = [(case, experiment, branches) for case in cases]
    if workers <= 1:
        return [_memory_branch_worker(argument) for argument in arguments]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_memory_branch_worker, arguments, chunksize=1))


def _digest_memory_batches(batches: list[MemoryBatch]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(np.ascontiguousarray(batch.target).tobytes())
        digest.update(np.ascontiguousarray(batch.completed_horizon).tobytes())
        digest.update(np.ascontiguousarray(batch.observed_fissions).tobytes())
        digest.update(np.ascontiguousarray(batch.first_break_index).tobytes())
        canonical = np.nan_to_num(batch.process, nan=-999.0)
        digest.update(np.ascontiguousarray(canonical).tobytes())
        for sequence in batch.post_break_sequences:
            digest.update(len(sequence).to_bytes(2, "little", signed=False))
            digest.update(np.asarray(sequence, dtype=np.int8).tobytes())
    return digest.hexdigest()


def _sequence_records(
    cases: list[StateCase], batches: list[MemoryBatch]
) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for state_index, (case, batch) in enumerate(zip(cases, batches)):
        for branch, sequence in enumerate(batch.post_break_sequences):
            records.append(
                SequenceRecord(
                    state_index=state_index,
                    state_id=case.state_id,
                    candidate=case.candidate,
                    matrix_id=case.matrix_id,
                    landmark=case.landmark,
                    branch=branch,
                    symbols=sequence,
                    completed_horizon=bool(batch.completed_horizon[branch]),
                    observed_fissions=int(batch.observed_fissions[branch]),
                    first_break_index=int(batch.first_break_index[branch]),
                )
            )
    return records


def _write_sequence_table(path: Path, records: list[SequenceRecord]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(
                    (
                        "state_index",
                        "state_id",
                        "candidate",
                        "matrix_id",
                        "landmark",
                        "branch",
                        "matrix_fold",
                        "completed_horizon",
                        "observed_fissions",
                        "first_break_index",
                        "post_break_length",
                        "post_break_bits",
                    )
                )
                for record in records:
                    writer.writerow(
                        (
                            record.state_index,
                            record.state_id,
                            record.candidate,
                            record.matrix_id,
                            record.landmark,
                            record.branch,
                            record.fold,
                            int(record.completed_horizon),
                            record.observed_fissions,
                            record.first_break_index,
                            len(record.symbols),
                            "".join("1" if value else "0" for value in record.symbols),
                        )
                    )


def _verify_sequence_table(path: Path, records: list[SequenceRecord]) -> int:
    observed = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for expected in records:
            try:
                row = next(reader)
            except StopIteration as error:
                raise ValueError("saved sequence table ended early") from error
            bits = tuple(value == "1" for value in row["post_break_bits"])
            if (
                int(row["state_index"]) != expected.state_index
                or row["state_id"] != expected.state_id
                or row["candidate"] != expected.candidate
                or int(row["matrix_id"]) != expected.matrix_id
                or int(row["landmark"]) != expected.landmark
                or int(row["branch"]) != expected.branch
                or int(row["matrix_fold"]) != expected.fold
                or bool(int(row["completed_horizon"]))
                != expected.completed_horizon
                or int(row["observed_fissions"]) != expected.observed_fissions
                or int(row["first_break_index"]) != expected.first_break_index
                or int(row["post_break_length"]) != len(expected.symbols)
                or bits != expected.symbols
            ):
                raise ValueError(f"sequence table mismatch at row {observed}")
            observed += 1
        if next(reader, None) is not None:
            raise ValueError("saved sequence table has extra rows")
    return observed


def _save_score_archive(
    path: Path, scores: dict[str, CandidateCrossfitScores]
) -> dict[str, Any]:
    arrays = score_archive_arrays(scores)
    np.savez_compressed(path, **arrays)
    audit: dict[str, Any] = {"arrays": len(arrays), "all_exact": True, "rows": {}}
    with np.load(path) as retained:
        if set(retained.files) != set(arrays):
            raise ValueError("saved cross-fit archive keys changed")
        for name, expected in arrays.items():
            exact = bool(np.array_equal(retained[name], expected))
            audit["all_exact"] = bool(audit["all_exact"] and exact)
            audit["rows"][name] = int(expected.size)
            if not exact:
                raise ValueError(f"saved cross-fit array changed: {name}")
    return audit


def _flatten_primary_rows(metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in metrics["primary_tests"]:
        directions = list(item["directions"].items())
        rows.append(
            {
                "contrast": item["contrast"],
                "baseline": item["baseline"],
                "enhanced": item["enhanced"],
                "candidate": item["candidate"],
                "gain_bits_per_transition": item["gain_bits_per_transition"],
                "ci95_lower": item["gain_ci95"][0],
                "ci95_upper": item["gain_ci95"][1],
                "equal_state_macro_gain_bits": item["equal_state_macro_gain_bits"],
                "randomization_p_raw": item["randomization_p_raw"],
                "randomization_p_holm": item["randomization_p_holm"],
                "direction_1": directions[0][0],
                "direction_1_gain": directions[0][1]["gain_bits_per_transition"],
                "direction_2": directions[1][0],
                "direction_2_gain": directions[1][1]["gain_bits_per_transition"],
                "both_directions_positive": item["both_directions_positive"],
                "passes_gate": item["passes_gate"],
                "transitions": item["transitions"],
            }
        )
    return pd.DataFrame(rows)


def _verify_legacy_branch_rows(
    path: Path, cases: list[StateCase], batches: list[MemoryBatch]
) -> int:
    observed = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for case, batch in zip(cases, batches):
            for branch in range(batch.target.size):
                row = next(reader, None)
                if row is None:
                    raise ValueError("legacy branch table ended early")
                if (
                    row["state_id"] != case.state_id
                    or int(row["branch"]) != branch
                    or int(row["joint_break_run3"]) != int(batch.target[branch])
                    or int(row["completed_horizon"])
                    != int(batch.completed_horizon[branch])
                ):
                    raise ValueError(
                        f"legacy branch mismatch at {case.state_id}/{branch}"
                    )
                for column, name in enumerate(PROCESS_COLUMNS):
                    raw = row[name]
                    actual = batch.process[branch, column]
                    if raw == "":
                        if not np.isnan(actual):
                            raise ValueError(f"legacy {name} NaN mismatch")
                    elif float(raw) != float(actual):
                        raise ValueError(f"legacy {name} mismatch")
                observed += 1
        if next(reader, None) is not None:
            raise ValueError("legacy branch table has extra rows")
    return observed


def _write_diagnostic_report(
    output: Path, metrics: dict[str, Any], manifest: dict[str, Any]
) -> None:
    lines = [
        "# Retrospective IID-support diagnostic",
        "",
        "## Outcome",
        "",
        "This regenerates the existing clean-room 12-fission futures and measures the reviewer-identified support mismatch. It is diagnostic only and does not confirm a memory claim.",
        "",
        "| Candidate | Legacy mismatched Markov gain | Corrected Markov gain | Legacy minus corrected IID loss |",
        "|---|---:|---:|---:|",
    ]
    for candidate, item in sorted(metrics["candidates"].items()):
        diagnostic = item["support_mismatch_diagnostic"]
        lines.append(
            f"| {candidate} | "
            f"{diagnostic['legacy_markov_gain']['gain_bits_per_transition']:.6f} | "
            f"{diagnostic['corrected_markov_gain']['gain_bits_per_transition']:.6f} | "
            f"{diagnostic['legacy_minus_corrected_iid_loss']['gain_bits_per_transition']:.6f} |"
        )
    lines.extend(
        (
            "",
            "Positive values in the last column mean that the mismatched IID fit makes the apparent Markov advantage larger; negative values mean it makes it smaller.",
            "",
            "## Audit boundary",
            "",
            f"All {manifest['legacy_rows_validated']} retained branch rows and the original batch digest matched exactly: **{manifest['legacy_reconstruction_exact']}**.",
            "",
            "The unavailable original L44 sequences are not present here, so this cannot repair or reproduce the preprint's numerical 0.015–0.022-bit result.",
            "",
        )
    )
    (output / "MEMORY_DIAGNOSTIC.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_registration_report(output: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Sealed prospective memory protocol",
        "",
        f"Registration ID: `{payload['registration_id']}`",
        "",
        "The protocol was sealed after the retrospective support-mismatch diagnostic and before generation of any `MEMCONF` matrix or future.",
        "",
        "- Primary hierarchy: support-matched IID → first-order Markov → duration-aware semi-Markov.",
        "- Cross-fitting: whole matrices, even-to-odd and odd-to-even.",
        "- Primary estimand: transition-weighted held-out bits per transition.",
        "- Multiplicity: Holm correction over two contrasts × two simulator candidates.",
        "- Confirmation: 200 new matrices, five landmarks, 64 futures per state, 32 fissions per future, exact replay.",
        "- Claim boundary: statistical predictive dependence only; not biological memory, error correction, or causal storage.",
        "",
    ]
    (output / "MEMORY_REGISTRATION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_confirmation_report(
    output: Path,
    metrics: dict[str, Any],
    counts: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    markov = metrics["support"]["markov_vs_iid"]
    duration = metrics["support"]["semimarkov_vs_markov"]
    if markov and duration:
        outcome = (
            "Both first-order and registered duration-aware dependence passed the "
            "prospective gates in both simulator candidates."
        )
    elif markov:
        outcome = (
            "First-order dependence passed, but the registered duration-aware "
            "extension did not pass in both simulator candidates."
        )
    else:
        outcome = (
            "The corrected first-order Markov comparison did not pass the "
            "prospective gates in both simulator candidates."
        )
    lines = [
        "# Prospective inheritance-dependence confirmation",
        "",
        "## Outcome",
        "",
        outcome,
        "",
        "| Contrast | Candidate | Pooled gain (bits/transition) | 95% CI | Holm p | Both directions positive | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics["primary_tests"]:
        lines.append(
            f"| {item['contrast']} | {item['candidate']} | "
            f"{item['gain_bits_per_transition']:.6f} | "
            f"[{item['gain_ci95'][0]:.6f}, {item['gain_ci95'][1]:.6f}] | "
            f"{item['randomization_p_holm']:.6f} | "
            f"{item['both_directions_positive']} | {item['passes_gate']} |"
        )
    lines.extend(
        (
            "",
            "## Sequence support",
            "",
            "| Candidate | Futures | No break | Empty suffix | Singleton suffix | Usable suffixes | Scored transitions |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in counts:
        if row["matrix_fold"] != "all":
            continue
        lines.append(
            f"| {row['candidate']} | {row['futures']} | {row['no_break']} | "
            f"{row['empty_post_break_suffix']} | "
            f"{row['singleton_post_break_suffix']} | {row['usable_suffixes']} | "
            f"{row['scored_transitions']} |"
        )
    lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            "A passing result means that recent binary inheritance state, and if applicable registered run duration, improves out-of-matrix next-symbol prediction over the nested baseline. It does not by itself establish biological memory, molecular information storage, error correction, or a causal mechanism; latent state and matrix heterogeneity may contribute to the predictive dependence.",
            "",
            "This memory analysis is separate from, and does not alter, the confirmed 12-fission L54 break-and-renewal risk predictor.",
            "",
            "## Audit boundary",
            "",
            f"Registration `{manifest['registration_id']}` was sealed before `MEMCONF`. All {manifest['confirmation_futures']} futures were regenerated exactly: **{manifest['confirmation_replay_exact']}**.",
            "",
        )
    )
    (output / "MEMORY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_diagnostic(
    source_results: Path, output_directory: Path, workers: int | None = None
) -> None:
    source_results = source_results.resolve()
    workers = workers or max(1, min(os.cpu_count() or 1, 12))
    print("[diagnose 1/6] Verifying retained scaled5 artifacts", flush=True)
    verify_checksums(source_results)
    manifest = json.loads((source_results / "manifest.json").read_text(encoding="utf-8"))
    experiment = ExperimentConfig.scaled5()
    if manifest["experiment"] != json.loads(json.dumps(experiment.to_dict())):
        raise ValueError("diagnostic source does not match scaled5 experiment")

    print("[diagnose 2/6] Reconstructing retained confirmation states", flush=True)
    cases = build_cohort(experiment, "CONF", experiment.confirmation)
    print("[diagnose 3/6] Regenerating retained 12-fission sequences", flush=True)
    batches = run_memory_branches(
        cases, experiment, experiment.confirmation.branches_per_state, workers
    )
    legacy_digest = _digest_batches(batches)
    digest_exact = legacy_digest == manifest["confirmation_digest_first"]
    if not digest_exact:
        raise AssertionError("retained future regeneration digest changed")
    legacy_rows = _verify_legacy_branch_rows(
        source_results / "confirmation_branches.csv.gz", cases, batches
    )

    print("[diagnose 4/6] Cross-fitting corrected and mismatched IID baselines", flush=True)
    records = _sequence_records(cases, batches)
    scores, fit_rows = crossfit_memory_models(records, include_legacy_iid=True)
    metrics = compute_memory_metrics(
        scores,
        repetitions=experiment.bootstrap_repetitions,
        master_seed=experiment.master_seed,
        confirmatory=False,
    )
    counts = sequence_count_rows(records)

    with _atomic_destination(output_directory) as output:
        print("[diagnose 5/6] Writing diagnostic audit artifacts", flush=True)
        (output / "metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _flatten_primary_rows(metrics).to_csv(output / "primary_tests.csv", index=False)
        pd.DataFrame(fit_rows).to_csv(output / "model_fits.csv", index=False)
        pd.DataFrame(calibration_rows(scores)).to_csv(
            output / "heldout_calibration.csv", index=False
        )
        pd.DataFrame(counts).to_csv(output / "sequence_counts.csv", index=False)
        _write_sequence_table(output / "sequences.csv.gz", records)
        sequence_rows = _verify_sequence_table(output / "sequences.csv.gz", records)
        score_audit = _save_score_archive(output / "crossfit_losses.npz", scores)
        diagnostic_manifest = {
            "clean_room": True,
            "scope": "retrospective IID fitting-support sensitivity diagnostic only",
            "confirmatory": False,
            "source_results": str(source_results),
            "source_result_hashes": _selected_hashes(
                source_results, DIAGNOSTIC_SOURCE_FILES
            ),
            "source_hashes": _source_hashes(),
            "experiment": experiment.to_dict(),
            "cohort": "CONF",
            "states": len(cases),
            "futures": len(records),
            "legacy_rows_validated": legacy_rows,
            "legacy_digest_expected": manifest["confirmation_digest_first"],
            "legacy_digest_regenerated": legacy_digest,
            "legacy_reconstruction_exact": True,
            "sequence_rows_readback_exact": sequence_rows == len(records),
            "crossfit_archive_readback": score_audit,
            "runtime": _runtime_manifest(),
        }
        (output / "manifest.json").write_text(
            json.dumps(_json_ready(diagnostic_manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_diagnostic_report(output, metrics, diagnostic_manifest)
        print("[diagnose 6/6] Sealing diagnostic checksums", flush=True)
        write_checksums(output)
    print(f"Memory diagnostic written to {output_directory.resolve()}", flush=True)


def prepare_registration(diagnostic: Path, registration: Path) -> None:
    diagnostic = diagnostic.resolve()
    print("[prepare 1/3] Verifying retrospective diagnostic", flush=True)
    verify_checksums(diagnostic)
    diagnostic_manifest = json.loads(
        (diagnostic / "manifest.json").read_text(encoding="utf-8")
    )
    if not diagnostic_manifest.get("legacy_reconstruction_exact"):
        raise ValueError("diagnostic did not exactly reconstruct retained futures")
    current_sources = _source_hashes()
    if diagnostic_manifest.get("source_hashes") != current_sources:
        raise ValueError("scientific source changed after diagnostic")

    experiment = _memory_confirmation_experiment()
    diagnostic_hashes = _artifact_hashes(diagnostic)
    payload: dict[str, Any] = {
        "format": MEMORY_REGISTRATION_FORMAT,
        "status": "sealed_before_confirmation",
        "scope": "prospective statistical inheritance-dependence hierarchy",
        "diagnostic": {
            "path": str(diagnostic),
            "artifact_hashes": diagnostic_hashes,
            "confirmatory": False,
            "legacy_reconstruction_exact": True,
        },
        "source_hashes": current_sources,
        "sequence_contract": {
            "inheritance": "strict parent-selected-daughter cosine H > 0.9",
            "conditioning": "first break in each future",
            "sequence": "all observed symbols strictly after the first break",
            "scored_support": "sequence destinations at indices 1 onward",
            "excluded_from_fit_and_score": (
                "no-break, empty-suffix, and singleton-suffix futures"
            ),
            "extinction": "absorbing; only observed pre-extinction symbols retained",
        },
        "model_contract": {
            "iid": "one probability fitted only on scored destinations",
            "markov": "destination probability conditional on previous symbol",
            "semimarkov": (
                "destination probability conditional on previous symbol and "
                "past-only run-duration bin"
            ),
            "duration_bins": ["1", "2", "3", "4", "5+"],
            "smoothing": "independent Beta(1,1) posterior means in every cell",
            "common_transition_support": True,
        },
        "crossfit_contract": {
            "unit": "catalytic matrix",
            "fold_0": "even matrix IDs",
            "fold_1": "odd matrix IDs",
            "directions": ["even_to_odd", "odd_to_even"],
            "candidate_separated": True,
        },
        "statistical_contract": {
            "primary_contrasts": {
                "markov_vs_iid": ["iid", "markov"],
                "semimarkov_vs_markov": ["markov", "semimarkov"],
            },
            "primary_estimand": "transition-weighted held-out bits per transition",
            "robustness_estimand": "equal-state macro-average gain",
            "bootstrap": "4096 paired resamples of whole catalytic matrices",
            "randomization": "4096 paired whole-matrix sign randomizations",
            "multiplicity": "Holm adjustment over 2 contrasts x 2 candidates",
            "gate": (
                "positive pooled gain, positive lower 95% matrix-bootstrap bound, "
                "Holm p < 0.05, and positive point gain in both cross-fit "
                "directions, in both candidates"
            ),
        },
        "confirmation_contract": {
            "cohort_name": "MEMCONF",
            "experiment": experiment.to_dict(),
            "matrices": MEMORY_MATRICES,
            "landmarks": list(MEMORY_LANDMARKS),
            "branches_per_state": MEMORY_BRANCHES,
            "horizon": MEMORY_CONFIRMATION_HORIZON,
            "exact_replay": True,
            "seed_disjoint_from_scaled5": (
                MEMORY_CONFIRM_MASTER_SEED != ExperimentConfig.scaled5().master_seed
            ),
            "seed_disjoint_from_mechconf": (
                MEMORY_CONFIRM_MASTER_SEED != MECHCONF_MASTER_SEED
            ),
        },
        "claim_boundary": (
            "statistical out-of-matrix predictive dependence, not biological "
            "memory, molecular storage, error correction, or causal mechanism"
        ),
    }
    payload["registration_id"] = _canonical_digest(payload)
    with _atomic_destination(registration) as output:
        print("[prepare 2/3] Writing frozen protocol", flush=True)
        (output / "registration.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_registration_report(output, payload)
        print("[prepare 3/3] Sealing registration checksums", flush=True)
        write_checksums(output)
    print(f"Memory registration written to {registration.resolve()}", flush=True)


def verify_registration(registration: Path) -> dict[str, Any]:
    registration = registration.resolve()
    verify_checksums(registration)
    payload = json.loads((registration / "registration.json").read_text(encoding="utf-8"))
    if payload.get("format") != MEMORY_REGISTRATION_FORMAT:
        raise ValueError("unsupported memory registration format")
    registered_id = payload.pop("registration_id")
    if _canonical_digest(payload) != registered_id:
        raise ValueError("memory registration identifier mismatch")
    payload["registration_id"] = registered_id
    if payload["source_hashes"] != _source_hashes():
        changed = [
            name
            for name, digest in payload["source_hashes"].items()
            if _source_hashes().get(name) != digest
        ]
        raise ValueError(f"registered memory source changed: {changed}")
    diagnostic = Path(payload["diagnostic"]["path"])
    if payload["diagnostic"]["artifact_hashes"] != _artifact_hashes(diagnostic):
        raise ValueError("memory diagnostic artifacts changed after registration")
    contract = payload["confirmation_contract"]
    experiment = _memory_confirmation_experiment()
    if contract["experiment"] != json.loads(json.dumps(experiment.to_dict())):
        raise ValueError("memory confirmation implementation diverged from registration")
    if not contract["seed_disjoint_from_scaled5"] or not contract[
        "seed_disjoint_from_mechconf"
    ]:
        raise ValueError("memory confirmation seed is not disjoint")
    return payload


def run_confirmation(
    registration: Path, output_directory: Path, workers: int | None = None
) -> None:
    registration = registration.resolve()
    workers = workers or max(1, min(os.cpu_count() or 1, 12))
    print("[confirm 1/8] Verifying sealed memory protocol", flush=True)
    payload = verify_registration(registration)
    experiment = _memory_confirmation_experiment()

    print("[confirm 2/8] Generating untouched MEMCONF trajectories", flush=True)
    cases = build_cohort(experiment, "MEMCONF", experiment.confirmation)
    print("[confirm 3/8] Shooting 128,000 untouched 32-fission futures", flush=True)
    batches = run_memory_branches(
        cases, experiment, experiment.confirmation.branches_per_state, workers
    )
    first_digest = _digest_memory_batches(batches)

    print("[confirm 4/8] Exactly regenerating all untouched futures", flush=True)
    regenerated = run_memory_branches(
        cases, experiment, experiment.confirmation.branches_per_state, workers
    )
    second_digest = _digest_memory_batches(regenerated)
    if first_digest != second_digest:
        raise AssertionError("MEMCONF exact sequence regeneration failed")
    del regenerated

    print("[confirm 5/8] Cross-fitting and testing the registered hierarchy", flush=True)
    records = _sequence_records(cases, batches)
    scores, fit_rows = crossfit_memory_models(records)
    metrics = compute_memory_metrics(
        scores,
        repetitions=MEMORY_REPETITIONS,
        master_seed=MEMORY_CONFIRM_MASTER_SEED,
        confirmatory=True,
    )
    counts = sequence_count_rows(records)

    with _atomic_destination(output_directory) as output:
        print("[confirm 6/8] Writing complete sequence and loss audit", flush=True)
        (output / "metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _flatten_primary_rows(metrics).to_csv(output / "primary_tests.csv", index=False)
        pd.DataFrame(fit_rows).to_csv(output / "model_fits.csv", index=False)
        pd.DataFrame(calibration_rows(scores)).to_csv(
            output / "heldout_calibration.csv", index=False
        )
        pd.DataFrame(counts).to_csv(output / "sequence_counts.csv", index=False)
        _write_sequence_table(output / "sequences.csv.gz", records)
        sequence_rows = _verify_sequence_table(output / "sequences.csv.gz", records)
        score_audit = _save_score_archive(output / "crossfit_losses.npz", scores)
        manifest = {
            "clean_room": True,
            "scope": "prospective statistical inheritance-dependence hierarchy",
            "registration_id": payload["registration_id"],
            "registration_path": str(registration),
            "registration_checksums_verified": True,
            "source_hashes_verified": True,
            "experiment": experiment.to_dict(),
            "cohort": "MEMCONF",
            "states": len(cases),
            "confirmation_futures": len(records),
            "confirmation_digest_first": first_digest,
            "confirmation_digest_second": second_digest,
            "confirmation_replay_exact": True,
            "sequence_rows_readback_exact": sequence_rows == len(records),
            "crossfit_archive_readback": score_audit,
            "support": metrics["support"],
            "runtime": _runtime_manifest(),
        }
        (output / "manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_confirmation_report(output, metrics, counts, manifest)

        print("[confirm 7/8] Rendering memory-only figures", flush=True)
        from .memory_plotting import create_memory_figures

        create_memory_figures(metrics, fit_rows, output)
        print("[confirm 8/8] Sealing confirmation checksums", flush=True)
        write_checksums(output)
    print(f"Memory confirmation written to {output_directory.resolve()}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corrected prospective tests of inheritance dependence"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    diagnose = commands.add_parser(
        "diagnose", help="quantify the IID support mismatch on retained futures"
    )
    diagnose.add_argument("--source", type=Path, default=Path("results/scaled5"))
    diagnose.add_argument(
        "--output", type=Path, default=Path("results/memory_diagnostic")
    )
    diagnose.add_argument("--workers", type=int, default=None)
    prepare = commands.add_parser(
        "prepare", help="seal the protocol before generating untouched matrices"
    )
    prepare.add_argument(
        "--diagnostic", type=Path, default=Path("results/memory_diagnostic")
    )
    prepare.add_argument(
        "--registration", type=Path, default=Path("results/memory_registration")
    )
    confirm = commands.add_parser(
        "confirm", help="run the sealed hierarchy on untouched matrices"
    )
    confirm.add_argument(
        "--registration", type=Path, default=Path("results/memory_registration")
    )
    confirm.add_argument(
        "--output", type=Path, default=Path("results/memory_confirmation")
    )
    confirm.add_argument("--workers", type=int, default=None)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "diagnose":
        run_diagnostic(arguments.source, arguments.output, arguments.workers)
    elif arguments.command == "prepare":
        prepare_registration(arguments.diagnostic, arguments.registration)
    else:
        run_confirmation(arguments.registration, arguments.output, arguments.workers)


if __name__ == "__main__":
    main()
