"""PX5 information-dynamic remeasurement of the sealed GN1 null campaign."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, GardConfig
from .generative_nulls import (
    INTERVENTION_ARMS,
    MECHANISMS,
    NullBatch,
    NullCase,
    _batch_digest as gn_batch_digest,
    _future_seed as gn_future_seed,
    _random_edit as gn_random_edit,
    _simulate_case_future,
    _state_digest as gn_state_digest,
    build_cases,
)
from .intervention_core import MolecularEdit, apply_molecular_edit
from .intervention_outgoing_rule import select_outgoing_rule_edits
from .mechanistic import verify_checksums, write_checksums
from .phir_ch5 import _append_ledger
from .phir_extension_common import (
    BOOTSTRAP_DRAWS,
    MASTER_REGISTRATION,
    MAX_WORKERS,
    MINIMUM_FREE_DISK_BYTES,
    RANDOMIZATION_DRAWS,
    RESULT_ROOT,
    ROOT,
    apply_holm,
    atomic_json,
    atomic_pickle,
    canonical_digest,
    canonical_json,
    paired_summary,
    runtime_versions,
    safe_score_pairs,
    sha256_file,
)
from .processes import evaluate_process
from .simulator import FissionRecord


DOCUMENT = "CODEX_CH5_PHIR_EXTENSION_PREREGISTRATION.md"
GN_RESULT = ROOT / "results" / "generative_null_decomposition"
GN_WORK = ROOT / "results" / ".generative_null_work" / "generate"
DEFAULT_VALIDATION = RESULT_ROOT / "px5_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "px5_registration"
DEFAULT_SMOKE = RESULT_ROOT / "px5_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "px5_generative_null_remeasurement"
DEFAULT_WORK = RESULT_ROOT / ".px5_work"
DEFAULT_LOG = RESULT_ROOT / "px5_generative_null_remeasurement.log"

LABEL = "CODEX_CH5_PHIR_EXTENSION_PX5_V1"
REGISTRATION_FORMAT = "codex-ch5-phir-extension-px5-registration-v1"
RESULT_FORMAT = "codex-ch5-phir-extension-px5-result-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-extension-px5-checkpoint-v1"
SERVICE_NAME = "codex-phir-extension-px5-20260820"

MATRICES = 24
LANDMARKS = (20, 35, 50, 65, 80)
UNTREATED_BRANCHES = 16
UNTREATED_HALVES = {"A": (0, 8), "B": (8, 16)}
INTERVENTION_BRANCHES = 8
HORIZON = 12
CPU_SECONDS = 12.0 * 3600.0
REPRESENTATIONS = ("material", "functional_flux")
INFORMATION_METRICS = ("full_revised", "public_revised")
GN_CASES_PER_MATRIX = len(CANDIDATES) * len(MECHANISMS) * len(LANDMARKS)

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_extension_px5.py",
    "plastic_heredity/phir_extension_common.py",
    "tests/test_phir_extension_px5.py",
    "plastic_heredity/generative_nulls.py",
    "plastic_heredity/config.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/processes.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_outgoing_rule.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_rescue_instruments.py",
    "plastic_heredity/seeds.py",
)


@dataclass(frozen=True)
class PX5Spec:
    label: str
    matrices: int
    landmarks: tuple[int, ...]
    untreated_branches: int
    intervention_branches: int
    horizon: int
    cpu_seconds: float


@dataclass(frozen=True)
class PX5Batch:
    matrix_id: int
    score_rows: tuple[dict[str, Any], ...]
    audit_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str


def scientific_spec() -> PX5Spec:
    return PX5Spec(
        "scientific",
        MATRICES,
        LANDMARKS,
        UNTREATED_BRANCHES,
        INTERVENTION_BRANCHES,
        HORIZON,
        CPU_SECONDS,
    )


def smoke_spec() -> PX5Spec:
    return PX5Spec("smoke", 1, (20,), 16, 8, 12, 300.0)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def _batch_digest(batch: PX5Batch) -> str:
    value = asdict(batch)
    value["cpu_seconds"] = 0.0
    value["scientific_digest"] = ""
    return canonical_digest(value)


def _case_archive_path(index: int) -> Path:
    return GN_WORK / f"state_{index:04d}.pkl"


class _GNCompatUnpickler(pickle.Unpickler):
    """Read sealed GN1 checkpoints created while that module ran as __main__."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "NullBatch":
            return NullBatch
        return super().find_class(module, name)


def _load_gn_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return _GNCompatUnpickler(handle).load()


def _load_archived_case(index: int, case: NullCase) -> NullBatch:
    batch = _load_gn_pickle(_case_archive_path(index))
    if (
        not isinstance(batch, NullBatch)
        or batch.state_id != case.state_id
        or batch.state_digest != gn_state_digest(case)
    ):
        raise ValueError(f"PX5 archived GN1 case mismatch: {index}")
    return batch


def _records_to_pairs(
    launch: NDArray, records: Sequence[FissionRecord]
) -> tuple[list[NDArray], list[NDArray]]:
    previous = np.asarray(launch, dtype=np.int64)
    past: list[NDArray] = []
    future: list[NDArray] = []
    for record in records:
        past.append(previous.copy())
        future.append(record.daughter.copy())
        previous = record.daughter
    return past, future


def _summary(records: Sequence[FissionRecord], completed: bool) -> tuple[int, int, int, int]:
    outcome = evaluate_process(list(records))
    return (
        int(outcome.joint_break_run3),
        int(outcome.break_event),
        int(sum(record.h > 0.9 for record in records)),
        int(completed and len(records) == HORIZON),
    )


def _edits(case: NullCase) -> tuple[MolecularEdit | None, ...]:
    rules = select_outgoing_rule_edits(case.snapshot.composition, case.source_beta)
    return (None, rules["RULE_UP"], rules["RULE_DOWN"], gn_random_edit(case))


def _score_group(
    case: NullCase,
    context: str,
    arm: str,
    half: str,
    past: Sequence[NDArray],
    future: Sequence[NDArray],
    outcomes: Sequence[tuple[int, int, int, int]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "state_id": case.state_id,
        "matrix_id": case.matrix_id,
        "candidate": case.candidate,
        "mechanism": case.mechanism,
        "landmark": case.landmark,
        "context": context,
        "arm": arm,
        "half": half,
        "branches": len(outcomes),
        "transition_pairs": len(past),
        "joint_break_run3": float(np.mean([item[0] for item in outcomes])),
        "break_probability": float(np.mean([item[1] for item in outcomes])),
        "inherited_fraction": float(
            np.mean([item[2] / HORIZON for item in outcomes])
        ),
        "survival_probability": float(np.mean([item[3] for item in outcomes])),
    }
    for representation in REPRESENTATIONS:
        score = safe_score_pairs(
            np.asarray(past),
            np.asarray(future),
            case.active_beta,
            representation,
            GardConfig(),
        )
        row.update(score.fields(representation))
    return row


def _run_matrix(
    args: tuple[int, PX5Spec, tuple[tuple[int, NullCase], ...]]
) -> PX5Batch:
    matrix_id, spec, indexed_cases = args
    started = time.process_time()
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for index, case in indexed_cases:
            archived = _load_archived_case(index, case)
            edits = _edits(case)
            if edits != archived.edits:
                raise AssertionError("PX5 selected edits differ from sealed GN1")
            untreated_records: dict[int, tuple[list[FissionRecord], bool]] = {}
            untreated_summaries: dict[int, tuple[int, int, int, int]] = {}
            mismatch = 0
            for branch in range(spec.untreated_branches):
                records, completed = _simulate_case_future(
                    case,
                    GardConfig(),
                    spec.horizon,
                    np.random.default_rng(gn_future_seed(case, branch)),
                )
                summary = _summary(records, completed)
                untreated_records[branch] = (records, completed)
                untreated_summaries[branch] = summary
                expected = (
                    int(archived.f12_joint[branch]),
                    int(archived.f12_break[branch]),
                    int(archived.f12_inherited_count[branch]),
                    int(archived.f12_survival[branch]),
                )
                mismatch += int(summary != expected)
            for half, (start, stop) in UNTREATED_HALVES.items():
                if stop > spec.untreated_branches:
                    continue
                past: list[NDArray] = []
                future: list[NDArray] = []
                outcomes: list[tuple[int, int, int, int]] = []
                for branch in range(start, stop):
                    records, _completed = untreated_records[branch]
                    local_past, local_future = _records_to_pairs(
                        case.snapshot.composition, records
                    )
                    past.extend(local_past)
                    future.extend(local_future)
                    outcomes.append(untreated_summaries[branch])
                rows.append(
                    _score_group(
                        case, "UNTREATED", "NOOP", half, past, future, outcomes
                    )
                )
            for arm_index, arm in enumerate(INTERVENTION_ARMS):
                past = []
                future = []
                outcomes = []
                edit = edits[arm_index]
                launch = (
                    case.snapshot.composition
                    if edit is None
                    else apply_molecular_edit(case.snapshot.composition, edit)
                )
                for branch in range(spec.intervention_branches):
                    if arm == "NOOP":
                        records, completed = untreated_records[branch]
                        summary = untreated_summaries[branch]
                    else:
                        records, completed = _simulate_case_future(
                            case,
                            GardConfig(),
                            spec.horizon,
                            np.random.default_rng(gn_future_seed(case, branch)),
                            edit,
                        )
                        summary = _summary(records, completed)
                    expected = (
                        int(archived.intervention_joint[arm_index, branch]),
                        int(archived.intervention_break[arm_index, branch]),
                        int(archived.intervention_inherited_count[arm_index, branch]),
                        int(archived.intervention_survival[arm_index, branch]),
                    )
                    mismatch += int(summary != expected)
                    local_past, local_future = _records_to_pairs(launch, records)
                    past.extend(local_past)
                    future.extend(local_future)
                    outcomes.append(summary)
                rows.append(
                    _score_group(
                        case,
                        "INTERVENTION",
                        arm,
                        "ALL",
                        past,
                        future,
                        outcomes,
                    )
                )
            audits.append(
                {
                    "state_id": case.state_id,
                    "matrix_id": matrix_id,
                    "case_index": index,
                    "state_digest_exact": int(
                        archived.state_digest == gn_state_digest(case)
                    ),
                    "selected_edits_exact": int(edits == archived.edits),
                    "outcome_mismatches": mismatch,
                    "archived_batch_digest": gn_batch_digest(archived),
                }
            )
    provisional = PX5Batch(
        matrix_id,
        tuple(rows),
        tuple(audits),
        float(time.process_time() - started),
        "",
    )
    return PX5Batch(
        provisional.matrix_id,
        provisional.score_rows,
        provisional.audit_rows,
        provisional.cpu_seconds,
        _batch_digest(provisional),
    )


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "phase": "PX5",
        "question": "which information-dynamic features remain under homogeneous, coupling-deranged, and fission-only generative nulls?",
        "source_campaign": "sealed GN1 generative-null decomposition",
        "spec": asdict(scientific_spec()),
        "selection": "first 24 matrix identifiers from the sealed 96-matrix seed order",
        "mechanisms": list(MECHANISMS),
        "representations": list(REPRESENTATIONS),
        "information_metrics": list(INFORMATION_METRICS),
        "remeasurement": {
            "untreated_branches": list(range(UNTREATED_BRANCHES)),
            "untreated_halves": UNTREATED_HALVES,
            "intervention_branches": list(range(INTERVENTION_BRANCHES)),
            "intervention_arms": list(INTERVENTION_ARMS),
            "horizon": HORIZON,
            "explicit_transition_pairs": True,
            "cross_branch_transitions": False,
            "every_replayed_outcome_checked_against_sealed_GN1": True,
        },
        "primary_comparisons": {
            "levels": "NATURAL_GARD minus each generative null for untreated full-block and public revised readings",
            "steerability": "SOURCE_RULE_DOWN minus SOURCE_RULE_UP information and heredity response in each mechanism",
            "reliability": "within-matrix Spearman between independent untreated branch-half readings across landmarks",
            "coupling": "within-matrix Spearman between information and JOINT_BREAK_RUN3 probability",
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap": BOOTSTRAP_DRAWS,
            "randomization": RANDOMIZATION_DRAWS,
            "candidate_pooling": False,
        },
        "classification": {
            "catalytic_enrichment": "natural-minus-null full-block or public information has positive 95% lower bounds in both candidates and both halves",
            "catalytic_steerability": "natural SOURCE_RULE_DOWN-minus-SOURCE_RULE_UP information and heredity effects have positive 95% lower bounds in both candidates",
            "geometric_residual": "finite nonzero null information is reported as a floor, not catalytic heredity",
            "no_omnibus_gate": True,
        },
        "no_new_matrix_or_branch_selection": True,
        "run_regardless_of_prior_phase_gates": True,
        "no_48_matrix_campaign": True,
        "claim_boundary": [
            "a geometric information floor is not catalytic heredity",
            "Phi-r is not consciousness, agency, or life",
            "null remeasurement cannot overwrite earlier intervention results",
            "strict-eight is excluded",
        ],
    }
    value["protocol_id"] = canonical_digest(value)
    return value


def _archive_prefix_digest() -> str:
    values: list[str] = []
    for index in range(MATRICES * GN_CASES_PER_MATRIX):
        path = _case_archive_path(index)
        if not path.exists():
            raise FileNotFoundError(f"missing sealed GN1 checkpoint: {path}")
        batch = _load_gn_pickle(path)
        if not isinstance(batch, NullBatch):
            raise ValueError(f"unexpected GN1 checkpoint type: {path}")
        values.append(gn_batch_digest(batch))
    return canonical_digest(values)


def validation_checks() -> dict[str, bool]:
    return {
        "master_registration_exists": MASTER_REGISTRATION.exists(),
        "gn_result_exists": (GN_RESULT / "manifest.json").exists(),
        "gn_first_checkpoint_exists": _case_archive_path(0).exists(),
        "gn_last_selected_checkpoint_exists": _case_archive_path(
            MATRICES * GN_CASES_PER_MATRIX - 1
        ).exists(),
        "matrix_subset_fixed": scientific_spec().matrices == 24,
        "all_four_mechanisms": len(MECHANISMS) == 4,
        "landmarks_fixed": scientific_spec().landmarks == LANDMARKS,
        "untreated_branches_fixed": UNTREATED_BRANCHES == 16
        and UNTREATED_HALVES == {"A": (0, 8), "B": (8, 16)},
        "intervention_branches_fixed": INTERVENTION_BRANCHES == 8,
        "horizon_fixed": HORIZON == 12,
        "cpu_allocation_fixed": CPU_SECONDS == 12 * 3600,
        "future_seed_has_no_arm_or_mechanism": "arm"
        not in inspect.signature(gn_future_seed).parameters
        and "mechanism" not in inspect.signature(gn_future_seed).parameters,
        "explicit_pairs_only": protocol()["remeasurement"]["explicit_transition_pairs"]
        and not protocol()["remeasurement"]["cross_branch_transitions"],
        "draws_fixed": BOOTSTRAP_DRAWS == 4096
        and RANDOMIZATION_DRAWS == 4096,
        "no_outcome_selection": protocol()["no_new_matrix_or_branch_selection"],
        "no_48_matrix_campaign": protocol()["no_48_matrix_campaign"],
        "strict_eight_excluded": "strict-eight is excluded"
        in protocol()["claim_boundary"],
    }


def run_validation() -> dict[str, Any]:
    checks = validation_checks()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_phir_extension_px5.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = {
        "format": "codex-ch5-phir-extension-px5-validation-v1",
        "checks": checks,
        "pytest_returncode": completed.returncode,
        "pytest_stdout": completed.stdout,
        "pytest_stderr": completed.stderr,
        "all_passed": bool(all(checks.values()) and completed.returncode == 0),
        "runtime": runtime_versions(),
    }
    if DEFAULT_VALIDATION.exists():
        shutil.rmtree(DEFAULT_VALIDATION)
    DEFAULT_VALIDATION.mkdir(parents=True)
    atomic_json(DEFAULT_VALIDATION / "validation.json", payload)
    write_checksums(DEFAULT_VALIDATION)
    if not payload["all_passed"]:
        raise AssertionError(f"PX5 validation failed\n{completed.stdout}\n{completed.stderr}")
    return payload


def register_program() -> dict[str, Any]:
    verify_checksums(DEFAULT_VALIDATION)
    verify_checksums(GN_RESULT)
    validation = json.loads((DEFAULT_VALIDATION / "validation.json").read_text())
    if not validation["all_passed"]:
        raise ValueError("PX5 validation did not pass")
    if DEFAULT_REGISTRATION.exists():
        raise FileExistsError(f"PX5 registration exists: {DEFAULT_REGISTRATION}")
    master = json.loads((MASTER_REGISTRATION / "registration.json").read_text())
    gn_manifest = json.loads((GN_RESULT / "manifest.json").read_text())
    body: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "master_registration_id": master["registration_id"],
        "protocol": protocol(),
        "source_hashes": _source_hashes(),
        "runtime": runtime_versions(),
        "gn_registration_id": gn_manifest["registration_id"],
        "gn_manifest_sha256": sha256_file(GN_RESULT / "manifest.json"),
        "selected_archive_prefix_digest": _archive_prefix_digest(),
        "new_scientific_matrices_at_registration": 0,
        "new_scientific_futures_at_registration": 0,
    }
    body["registration_id"] = canonical_digest(body)
    DEFAULT_REGISTRATION.mkdir(parents=True)
    shutil.copy2(ROOT / DOCUMENT, DEFAULT_REGISTRATION / "preregistration.md")
    atomic_json(DEFAULT_REGISTRATION / "protocol.json", body["protocol"])
    atomic_json(DEFAULT_REGISTRATION / "registration.json", body)
    write_checksums(DEFAULT_REGISTRATION)
    _append_ledger(
        f"<!-- phir-extension-px5-registration-{body['registration_id']} -->",
        [
            "## Phi-r extension PX5 registered",
            "",
            f"- Registration: `{body['registration_id']}`.",
            "- The first 24 sealed GN1 matrices and fixed branch prefixes were selected without inspecting new information outcomes.",
            "- All four generative mechanisms will be remeasured with explicit branch pairs.",
        ],
    )
    return body


def verify_registration() -> dict[str, Any]:
    verify_checksums(DEFAULT_REGISTRATION)
    body = json.loads((DEFAULT_REGISTRATION / "registration.json").read_text())
    observed = body.pop("registration_id")
    if body.get("format") != REGISTRATION_FORMAT or observed != canonical_digest(body):
        raise ValueError("PX5 registration identity failed")
    body["registration_id"] = observed
    if body["protocol"] != canonical_json(protocol()):
        raise ValueError("PX5 protocol changed")
    if body["source_hashes"] != _source_hashes():
        raise ValueError("PX5 source changed after registration")
    if body["selected_archive_prefix_digest"] != _archive_prefix_digest():
        raise ValueError("PX5 sealed GN1 archive prefix changed")
    return body


def _group_cases(
    cases: Sequence[NullCase],
) -> dict[int, tuple[tuple[int, NullCase], ...]]:
    grouped: dict[int, list[tuple[int, NullCase]]] = {
        matrix_id: [] for matrix_id in range(MATRICES)
    }
    for index, case in enumerate(cases):
        grouped[case.matrix_id].append((index, case))
    output = {key: tuple(value) for key, value in grouped.items()}
    if any(len(value) != GN_CASES_PER_MATRIX for value in output.values()):
        raise AssertionError("PX5 case grouping changed")
    return output


def _checkpointed(
    spec: PX5Spec,
    registration: Mapping[str, Any],
    directory: Path,
    stage: str,
    groups: Mapping[int, tuple[tuple[int, NullCase], ...]],
    workers: int,
    prior_cpu: float = 0.0,
) -> tuple[list[PX5Batch], float]:
    directory.mkdir(parents=True, exist_ok=True)
    case_digests = {
        str(matrix_id): [gn_state_digest(case) for _, case in groups[matrix_id]]
        for matrix_id in range(spec.matrices)
    }
    contract = {
        "format": CHECKPOINT_FORMAT,
        "stage": stage,
        "registration_id": registration["registration_id"],
        "protocol_id": registration["protocol"]["protocol_id"],
        "spec": asdict(spec),
        "case_digests": case_digests,
        "source_hashes": registration["source_hashes"],
    }
    contract["contract_id"] = canonical_digest(contract)
    contract_path = directory / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != canonical_json(contract):
            raise ValueError("PX5 checkpoint contract changed")
    else:
        atomic_json(contract_path, contract)
    batches: list[PX5Batch | None] = [None] * spec.matrices
    missing: list[int] = []
    cpu = float(prior_cpu)
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:04d}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                batch = pickle.load(handle)
            if not isinstance(batch, PX5Batch) or batch.scientific_digest != _batch_digest(batch):
                raise ValueError(f"invalid PX5 checkpoint: {path}")
            batches[matrix_id] = batch
            cpu += batch.cpu_seconds
        else:
            missing.append(matrix_id)

    started = time.time()

    def status(state: str) -> None:
        complete = sum(item is not None for item in batches)
        elapsed = max(time.time() - started, 1e-9)
        rate = complete / elapsed if complete else 0.0
        atomic_json(
            directory / "status.json",
            {
                "stage": stage,
                "state": state,
                "completed": complete,
                "total": spec.matrices,
                "fraction": complete / spec.matrices,
                "cpu_seconds": cpu,
                "eta_seconds": (spec.matrices - complete) / rate if rate else None,
            },
        )

    status("running")
    arguments = [(matrix_id, spec, groups[matrix_id]) for matrix_id in missing]
    executor: ProcessPoolExecutor | None = None
    generated: Iterable[PX5Batch]
    if workers <= 1:
        generated = map(_run_matrix, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS))
        generated = executor.map(_run_matrix, arguments, chunksize=1)
    try:
        for matrix_id, batch in zip(missing, generated, strict=True):
            if batch.matrix_id != matrix_id or batch.scientific_digest != _batch_digest(batch):
                raise AssertionError("PX5 worker returned invalid batch")
            batches[matrix_id] = batch
            atomic_pickle(directory / f"matrix_{matrix_id:04d}.pkl", batch)
            cpu += batch.cpu_seconds
            status("running")
            print(f"[PX5 {stage}] {sum(item is not None for item in batches)}/{spec.matrices}", flush=True)
            if cpu > spec.cpu_seconds:
                status("paused_cpu_budget")
                raise RuntimeError("PX5 CPU allocation reached; checkpoints retained")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    if any(batch is None for batch in batches):
        raise AssertionError("PX5 checkpoint stage incomplete")
    status("complete")
    return [batch for batch in batches if batch is not None], cpu


def _replay_audit(
    generated: Sequence[PX5Batch], replayed: Sequence[PX5Batch]
) -> dict[str, Any]:
    rows = [
        {
            "matrix_id": left.matrix_id,
            "generated": left.scientific_digest,
            "replay": right.scientific_digest,
            "exact": left.scientific_digest == right.scientific_digest,
        }
        for left, right in zip(generated, replayed, strict=True)
    ]
    archived_mismatches = int(
        sum(
            audit["outcome_mismatches"]
            for batch in generated
            for audit in batch.audit_rows
        )
    )
    return {
        "matrices": rows,
        "archived_outcome_mismatches": archived_mismatches,
        "complete_exact_replay": len(rows) == MATRICES
        and all(row["exact"] for row in rows)
        and archived_mismatches == 0,
    }


def _paired_level_effect(
    frame: pd.DataFrame,
    candidate: str,
    half: str,
    null: str,
    metric: str,
) -> pd.Series:
    selected = frame[
        (frame["context"] == "UNTREATED")
        & (frame["candidate"] == candidate)
        & (frame["half"] == half)
        & frame["mechanism"].isin(("NATURAL_GARD", null))
    ]
    pivot = selected.pivot(
        index=["matrix_id", "landmark"], columns="mechanism", values=metric
    )
    return (
        pivot["NATURAL_GARD"] - pivot[null]
    ).groupby("matrix_id").mean().sort_index()


def _rule_effect(
    frame: pd.DataFrame,
    candidate: str,
    mechanism: str,
    metric: str,
) -> pd.Series:
    selected = frame[
        (frame["context"] == "INTERVENTION")
        & (frame["candidate"] == candidate)
        & (frame["mechanism"] == mechanism)
    ]
    pivot = selected.pivot(
        index=["matrix_id", "landmark"], columns="arm", values=metric
    )
    return (
        pivot["SOURCE_RULE_DOWN"] - pivot["SOURCE_RULE_UP"]
    ).groupby("matrix_id").mean().sort_index()


def _safe_matrix_summary(values: Sequence[float], key: str) -> tuple[dict[str, Any], dict[str, NDArray]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return (
            {
                "effect": float("nan"),
                "ci90": [float("nan"), float("nan")],
                "ci95": [float("nan"), float("nan")],
                "matrices": int(finite.size),
                "positive_sign_randomization_p": float("nan"),
                "two_sided_sign_randomization_p": float("nan"),
            },
            {"matrix_effects": finite},
        )
    return paired_summary(finite, key)


def analyze(
    batches: Sequence[PX5Batch],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    frame = pd.DataFrame([row for batch in batches for row in batch.score_rows])
    audit = pd.DataFrame([row for batch in batches for row in batch.audit_rows])
    frame["candidate"] = frame["candidate"].astype(str).str.zfill(2)
    arrays: dict[str, NDArray] = {}
    level_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    metric_names = [
        f"{representation}_{metric}"
        for representation in REPRESENTATIONS
        for metric in INFORMATION_METRICS
    ]
    for candidate in CANDIDATES:
        for half in UNTREATED_HALVES:
            for null in MECHANISMS:
                if null == "NATURAL_GARD":
                    continue
                for metric in metric_names:
                    effect = _paired_level_effect(frame, candidate, half, null, metric)
                    summary, local = paired_summary(
                        effect.to_numpy(),
                        f"PX5/level/{candidate}/{half}/{null}/{metric}",
                    )
                    summary.update(
                        {
                            "candidate": candidate,
                            "half": half,
                            "null": null,
                            "metric": metric,
                            "contrast": "NATURAL_GARD-null",
                        }
                    )
                    level_rows.append(summary)
                    arrays.update(
                        {
                            f"level__{candidate}__{half}__{null}__{metric}__{name}": value
                            for name, value in local.items()
                        }
                    )
                    for matrix_id, value in effect.items():
                        matrix_rows.append(
                            {
                                "kind": "level",
                                "candidate": candidate,
                                "cell": half,
                                "mechanism": null,
                                "metric": metric,
                                "matrix_id": int(matrix_id),
                                "value": float(value),
                            }
                        )
        for mechanism in MECHANISMS:
            for metric in (*metric_names, "inherited_fraction", "joint_break_run3"):
                effect = _rule_effect(frame, candidate, mechanism, metric)
                # For F12 risk, positive stabilization is RULE_UP minus RULE_DOWN.
                if metric == "joint_break_run3":
                    effect = -effect
                summary, local = paired_summary(
                    effect.to_numpy(),
                    f"PX5/rule/{candidate}/{mechanism}/{metric}",
                )
                summary.update(
                    {
                        "candidate": candidate,
                        "mechanism": mechanism,
                        "metric": metric,
                        "contrast": "stabilizing-direction",
                    }
                )
                rule_rows.append(summary)
                arrays.update(
                    {
                        f"rule__{candidate}__{mechanism}__{metric}__{name}": value
                        for name, value in local.items()
                    }
                )
    apply_holm(level_rows)
    apply_holm(rule_rows)

    reliability_rows: list[dict[str, Any]] = []
    coupling_rows: list[dict[str, Any]] = []
    untreated = frame[frame["context"] == "UNTREATED"].copy()
    for candidate in CANDIDATES:
        for mechanism in MECHANISMS:
            selected = untreated[
                (untreated["candidate"] == candidate)
                & (untreated["mechanism"] == mechanism)
            ]
            for metric in metric_names:
                correlations: list[float] = []
                couplings: list[float] = []
                for _matrix_id, local in selected.groupby("matrix_id", sort=True):
                    pivot = local.pivot(index="landmark", columns="half", values=metric)
                    correlations.append(
                        float(spearmanr(pivot["A"], pivot["B"]).statistic)
                    )
                    couplings.append(
                        float(
                            spearmanr(
                                local[metric].to_numpy(float),
                                local["joint_break_run3"].to_numpy(float),
                            ).statistic
                        )
                    )
                summary, local_arrays = _safe_matrix_summary(
                    correlations, f"PX5/reliability/{candidate}/{mechanism}/{metric}"
                )
                summary.update(
                    {
                        "candidate": candidate,
                        "mechanism": mechanism,
                        "metric": metric,
                    }
                )
                reliability_rows.append(summary)
                arrays.update(
                    {
                        f"reliability__{candidate}__{mechanism}__{metric}__{name}": value
                        for name, value in local_arrays.items()
                    }
                )
                summary, local_arrays = _safe_matrix_summary(
                    couplings, f"PX5/coupling/{candidate}/{mechanism}/{metric}"
                )
                summary.update(
                    {
                        "candidate": candidate,
                        "mechanism": mechanism,
                        "metric": metric,
                    }
                )
                coupling_rows.append(summary)
                arrays.update(
                    {
                        f"coupling__{candidate}__{mechanism}__{metric}__{name}": value
                        for name, value in local_arrays.items()
                    }
                )

    def complete_level(metric: str) -> bool:
        selected = [row for row in level_rows if row["metric"] == metric]
        return bool(
            len(selected) == 12
            and all(row["effect"] > 0 and row["ci95"][0] > 0 for row in selected)
        )

    def natural_rule(metric: str) -> bool:
        selected = [
            row
            for row in rule_rows
            if row["metric"] == metric and row["mechanism"] == "NATURAL_GARD"
        ]
        return bool(
            len(selected) == 2
            and all(row["effect"] > 0 and row["ci95"][0] > 0 for row in selected)
        )

    gates = {
        "material_full_catalytic_enrichment": complete_level("material_full_revised"),
        "material_public_catalytic_enrichment": complete_level("material_public_revised"),
        "natural_full_information_steerability": natural_rule("material_full_revised"),
        "natural_public_information_steerability": natural_rule("material_public_revised"),
        "natural_heredity_steerability": natural_rule("inherited_fraction")
        and natural_rule("joint_break_run3"),
        "archived_outcomes_exact": bool((audit["outcome_mismatches"] == 0).all()),
    }
    classification = (
        "catalytic_enrichment_and_steerability"
        if (gates["material_full_catalytic_enrichment"] or gates["material_public_catalytic_enrichment"])
        and (gates["natural_full_information_steerability"] or gates["natural_public_information_steerability"])
        else "catalytic_enrichment_without_information_steerability"
        if gates["material_full_catalytic_enrichment"] or gates["material_public_catalytic_enrichment"]
        else "information_steerability_without_complete_level_enrichment"
        if gates["natural_full_information_steerability"] or gates["natural_public_information_steerability"]
        else "no_complete_catalytic_information_separation"
    )
    metrics = {
        "format": "codex-ch5-phir-extension-px5-metrics-v1",
        "level_contrasts": level_rows,
        "rule_responses": rule_rows,
        "statewise_reliability": reliability_rows,
        "information_f12_coupling": coupling_rows,
        "gates": gates,
        "classification": classification,
        "no_omnibus_gate": True,
    }
    return metrics, {
        "scores": frame,
        "archive_audit": audit,
        "matrix_effects": pd.DataFrame(matrix_rows),
    }, arrays


def _write_result(
    batches: Sequence[PX5Batch],
    replay: Mapping[str, Any],
    registration: Mapping[str, Any],
    cpu: float,
) -> dict[str, Any]:
    metrics, tables, arrays = analyze(batches)
    temporary = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(temporary / f"{name}.csv.gz", index=False)
    np.savez_compressed(temporary / "inference_arrays.npz", **arrays)
    atomic_json(temporary / "primary_metrics.json", metrics)
    atomic_json(temporary / "replay_audit.json", replay)
    report = [
        "# PX5 generative-null information remeasurement",
        "",
        f"Registration: `{registration['registration_id']}`.",
        "",
        f"Classification: **{metrics['classification']}**.",
        "",
        "## Registered component gates",
        "",
        "```json",
        json.dumps(metrics["gates"], indent=2, sort_keys=True),
        "```",
        "",
        "There is deliberately no omnibus pass gate. Nonzero information under a null is a geometric or sampling floor and must not be relabeled catalytic heredity.",
    ]
    (temporary / "SCIENTIFIC_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (temporary / "LAY_SUMMARY.md").write_text(
        "# PX5 lay summary\n\n"
        "We replayed the same already-sealed futures under ordinary catalysis, uniform catalysis, deliberately misaligned catalysis, and growth-and-division geometry without catalytic state dependence. We then applied the new information measurements to exactly those futures.\n\n"
        f"The descriptive classification is **{metrics['classification']}**. The separate gates show whether ordinary catalytic dynamics add information structure or steerability beyond the geometric floor.\n",
        encoding="utf-8",
    )
    manifest = {
        "format": RESULT_FORMAT,
        "registration_id": registration["registration_id"],
        "matrices": MATRICES,
        "states": MATRICES * GN_CASES_PER_MATRIX,
        "cpu_seconds": cpu,
        "complete_exact_replay": bool(replay["complete_exact_replay"]),
        "complete_readback_exact": False,
        "classification": metrics["classification"],
        "gates": metrics["gates"],
    }
    atomic_json(temporary / "manifest.json", manifest)
    write_checksums(temporary)
    temporary.replace(DEFAULT_OUTPUT)
    verify_checksums(DEFAULT_OUTPUT)
    readback = pd.read_csv(DEFAULT_OUTPUT / "scores.csv.gz")
    exact = len(readback) == len(tables["scores"])
    manifest["complete_readback_exact"] = exact
    atomic_json(DEFAULT_OUTPUT / "manifest.json", manifest)
    atomic_json(DEFAULT_OUTPUT / "readback_audit.json", {"complete": exact})
    write_checksums(DEFAULT_OUTPUT)
    if not exact:
        raise AssertionError("PX5 readback failed")
    _append_ledger(
        f"<!-- phir-extension-px5-result-{registration['registration_id']} -->",
        [
            "## Phi-r extension PX5 completed",
            "",
            "- Result: `results/phir_extension/px5_generative_null_remeasurement`.",
            f"- Classification: `{metrics['classification']}`; no omnibus gate was used.",
            "- Sealed GN1 outcomes and complete replay matched exactly.",
        ],
    )
    return manifest


def _build_groups() -> dict[int, tuple[tuple[int, NullCase], ...]]:
    cases = build_cases(matrices=MATRICES, landmarks=LANDMARKS)
    if len(cases) != MATRICES * GN_CASES_PER_MATRIX:
        raise AssertionError("PX5 regenerated an unexpected number of GN1 cases")
    return _group_cases(cases)


def run_scientific(workers: int = MAX_WORKERS) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"PX5 output exists: {DEFAULT_OUTPUT}")
    if not (RESULT_ROOT / "px4_simulator_moderator" / "manifest.json").exists():
        raise RuntimeError("PX5 is locked until PX4 completes")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("PX5 refused below the sealed disk floor")
    groups = _build_groups()
    generated, cpu = _checkpointed(
        scientific_spec(),
        registration,
        DEFAULT_WORK / "generate",
        "generate",
        groups,
        workers,
    )
    replay_groups = _build_groups()
    replayed, cpu = _checkpointed(
        scientific_spec(),
        registration,
        DEFAULT_WORK / "replay",
        "replay",
        replay_groups,
        workers,
        cpu,
    )
    replay = _replay_audit(generated, replayed)
    if not replay["complete_exact_replay"]:
        raise AssertionError("PX5 exact or archived replay failed")
    return _write_result(generated, replay, registration, cpu)


def run_smoke() -> dict[str, Any]:
    if DEFAULT_SMOKE.exists():
        raise FileExistsError(f"PX5 smoke exists: {DEFAULT_SMOKE}")
    cases = build_cases(matrices=1, landmarks=LANDMARKS)
    first_case = cases[0]
    first = _run_matrix((0, smoke_spec(), ((0, first_case),)))
    second = _run_matrix((0, smoke_spec(), ((0, first_case),)))
    rows = pd.DataFrame(first.score_rows)
    audit = pd.DataFrame(first.audit_rows)
    payload = {
        "format": "codex-ch5-phir-extension-px5-smoke-v1",
        "all_intervention_arms": set(rows[rows.context == "INTERVENTION"].arm)
        == set(INTERVENTION_ARMS),
        "all_untreated_halves": set(rows[rows.context == "UNTREATED"].half)
        == set(UNTREATED_HALVES),
        "explicit_pairs_positive": bool((rows["transition_pairs"] > 0).all()),
        "scores_finite": bool(
            np.isfinite(rows["material_full_revised"]).all()
            and np.isfinite(rows["functional_flux_full_revised"]).all()
        ),
        "archived_outcomes_exact": bool((audit["outcome_mismatches"] == 0).all()),
        "replay_exact": first.scientific_digest == second.scientific_digest,
        "effect_sizes_suppressed": True,
    }
    payload["passed"] = bool(
        all(value for key, value in payload.items() if key != "format")
    )
    DEFAULT_SMOKE.mkdir(parents=True)
    atomic_json(DEFAULT_SMOKE / "smoke.json", payload)
    write_checksums(DEFAULT_SMOKE)
    if not payload["passed"]:
        raise AssertionError("PX5 smoke failed")
    return payload


def launch_detached(workers: int = MAX_WORKERS) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"PX5 output exists: {DEFAULT_OUTPUT}")
    if not (RESULT_ROOT / "px4_simulator_moderator" / "manifest.json").exists():
        raise RuntimeError("PX5 launch is locked until PX4 completes")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("PX5 launch refused below the sealed disk floor")
    DEFAULT_WORK.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run",
        "--user",
        f"--unit={SERVICE_NAME}",
        "--collect",
        "--property",
        f"WorkingDirectory={ROOT}",
        "--property",
        f"StandardOutput=append:{DEFAULT_LOG}",
        "--property",
        f"StandardError=append:{DEFAULT_LOG}",
        sys.executable,
        "-m",
        "plastic_heredity.phir_extension_px5",
        "run",
        "--workers",
        str(min(workers, MAX_WORKERS)),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = {
        "registration_id": registration["registration_id"],
        "service": SERVICE_NAME,
        "workers": min(workers, MAX_WORKERS),
        "launched_at_unix": time.time(),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    atomic_json(DEFAULT_WORK / "detached_launch.json", payload)
    return payload


def status_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phase": "PX5",
        "validation": DEFAULT_VALIDATION.exists(),
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "complete": DEFAULT_OUTPUT.exists(),
        "launch_locked_until_px4": not (
            RESULT_ROOT / "px4_simulator_moderator" / "manifest.json"
        ).exists(),
        "service": SERVICE_NAME,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
    }
    for stage in ("generate", "replay"):
        path = DEFAULT_WORK / stage / "status.json"
        if path.exists():
            payload[stage] = json.loads(path.read_text())
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    commands.add_parser("register")
    commands.add_parser("smoke")
    run = commands.add_parser("run")
    run.add_argument("--workers", type=int, default=MAX_WORKERS)
    launch = commands.add_parser("launch")
    launch.add_argument("--workers", type=int, default=MAX_WORKERS)
    commands.add_parser("status")
    arguments = parser.parse_args(argv)
    if arguments.command == "validate":
        print(json.dumps(run_validation(), indent=2, sort_keys=True))
    elif arguments.command == "register":
        print(json.dumps(register_program(), indent=2, sort_keys=True))
    elif arguments.command == "smoke":
        print(json.dumps(run_smoke(), indent=2, sort_keys=True))
    elif arguments.command == "run":
        print(json.dumps(run_scientific(arguments.workers), indent=2, sort_keys=True))
    elif arguments.command == "launch":
        print(json.dumps(launch_detached(arguments.workers), indent=2, sort_keys=True))
    elif arguments.command == "status":
        print(json.dumps(status_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
