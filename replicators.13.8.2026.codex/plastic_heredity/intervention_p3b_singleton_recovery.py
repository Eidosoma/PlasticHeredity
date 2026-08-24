"""Prospective recovery for P3b states with fewer than two occupied types.

The sealed P3b runner stopped before replay or inference because its balanced
within-PxP random surgery is mathematically undefined on a one-entry block.
This additive module retains every registered state and applies all-arm NOOP
at structurally ineligible states.  It never changes the sealed P3b source.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import intervention_p3b_dose_bridge as original
from . import intervention_replication as base
from .config import CANDIDATES, ExperimentConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .intervention_core import FrozenFullPredictor, simulate_one_shot
from .intervention_metrics import generate_inference_draws
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .seeds import derive_seed


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPOSITORY_ROOT / "results_intervention_replication"
ORIGINAL_REGISTRATION = original.DEFAULT_REGISTRATION
DEFAULT_WORK = original.DEFAULT_WORK
DEFAULT_OUTPUT = original.DEFAULT_OUTPUT
DEFAULT_AMENDMENT = RESULT_ROOT / "p3b_singleton_recovery_amendment"

DOCUMENT = "CODEX_INTERVENTION_P3B_SINGLETON_RECOVERY_AMENDMENT.md"
AMENDMENT_FORMAT = "codex-intervention-p3b-singleton-recovery-amendment-v1"
RECOVERY_CHECKPOINT_FORMAT = (
    "codex-intervention-p3b-singleton-recovery-checkpoint-v1"
)
RESULT_FORMAT = "codex-intervention-p3b-singleton-recovered-result-v1"
EXPECTED_ORIGINAL_REGISTRATION_ID = (
    "c1fe38be6a7e2b71eb5e288c9e238ff45c30d8fe388f2d21a879acea6dd5624e"
)
EXPECTED_FAILURE = "ValueError: P3b surgery requires at least two present types"
EXPECTED_COMPLETED_STATES = 363
EXPECTED_COMPLETED_FUTURES = 81_312
EXPECTED_NEXT_CASE = "INTP3B_DOSE_BRIDGE_V1-c02-m030-g060"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_p3b_singleton_recovery.py",
    "tests/test_intervention_p3b_singleton_recovery.py",
)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _checkpoint_prefix_audit(
    work: Path, *, require_interrupted_boundary: bool
) -> dict[str, Any]:
    generation = work.resolve() / "generate"
    contract_path = generation / "checkpoint_contract.json"
    status_path = generation / "status.json"
    if not contract_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("P3b generation checkpoint metadata is incomplete")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("states_total") != 960 or status.get("futures_total") != 215_040:
        raise ValueError("P3b checkpoint totals changed")
    if require_interrupted_boundary and (
        status.get("states_complete") != EXPECTED_COMPLETED_STATES
        or status.get("futures_complete") != EXPECTED_COMPLETED_FUTURES
    ):
        raise ValueError("interrupted P3b checkpoint count changed")
    if contract["case_ids"][EXPECTED_COMPLETED_STATES] != EXPECTED_NEXT_CASE:
        raise ValueError("the P3b failure boundary changed")
    files = [generation / f"state_{index:04d}.pkl" for index in range(EXPECTED_COMPLETED_STATES)]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("an immutable P3b prefix checkpoint is missing")
    next_checkpoint = generation / f"state_{EXPECTED_COMPLETED_STATES:04d}.pkl"
    if require_interrupted_boundary and next_checkpoint.exists():
        raise ValueError("the failed P3b state unexpectedly has a checkpoint")
    hashes = {path.name: sha256_file(path) for path in files}
    return {
        "format": "codex-intervention-p3b-immutable-prefix-audit-v1",
        "original_checkpoint_contract_sha256": sha256_file(contract_path),
        "immutable_state_checkpoint_count": len(files),
        "immutable_state_checkpoint_hashes": hashes,
        "immutable_prefix_aggregate_sha256": _canonical_digest(hashes),
        "completed_futures": EXPECTED_COMPLETED_FUTURES,
        "next_state_index": EXPECTED_COMPLETED_STATES,
        "next_state_id": EXPECTED_NEXT_CASE,
        "reported_exception": EXPECTED_FAILURE,
        "checkpoints_deserialized": False,
        "branch_outcomes_loaded": False,
    }


def _protocol(prefix: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": AMENDMENT_FORMAT,
        "status": "frozen_after_partial_generation_before_replay_or_inference",
        "original_registration_id": EXPECTED_ORIGINAL_REGISTRATION_ID,
        "failure": {
            "exception": EXPECTED_FAILURE,
            "states_completed": EXPECTED_COMPLETED_STATES,
            "futures_completed": EXPECTED_COMPLETED_FUTURES,
            "next_state_id": EXPECTED_NEXT_CASE,
            "replay_started": False,
            "inference_started": False,
            "result_bundle_created": False,
        },
        "recovery_rule": {
            "eligibility": "number of occupied types >= 2",
            "ineligible_action": "STRUCTURAL_NO_ACTION",
            "all_seven_arm_labels_use_original_beta": True,
            "all_seven_arm_labels_use_same_registered_arm_free_future_seed": True,
            "predictions_must_be_bitwise_identical_across_arms": True,
            "records_must_be_bitwise_identical_across_arms_within_branch": True,
            "paired_contribution_to_every_contrast": 0.0,
            "state_retained": True,
            "matrix_retained": True,
            "future_retried": False,
            "matrix_or_state_replaced": False,
            "applies_to_every_later_state_meeting_the_rule": True,
        },
        "unchanged": {
            "eligible_state_interventions": True,
            "cohort_seed_and_state_order": True,
            "future_seed_and_branch_halves": True,
            "endpoint_and_horizon": True,
            "arms_and_doses": True,
            "inference_and_gates": True,
            "frozen_predictor": True,
            "sealed_p3b_source": True,
        },
        "checkpoint_reuse": {
            "immutable_prefix_count": EXPECTED_COMPLETED_STATES,
            "immutable_prefix_aggregate_sha256": prefix[
                "immutable_prefix_aggregate_sha256"
            ],
            "prefix_checkpoints_loaded_before_seal": False,
            "prefix_checkpoints_regenerated": False,
            "complete_replay_required": True,
        },
        "outcome_firewall": {
            "event_rates_computed_before_seal": False,
            "arm_means_computed_before_seal": False,
            "contrasts_computed_before_seal": False,
            "candidate_differences_computed_before_seal": False,
            "checkpoint_files_hashed_without_deserialization": True,
        },
        "stop_rule": {
            "seal_result_after_recovery": True,
            "launch_next_phase": False,
        },
        "claim_boundary": {
            "recovered_estimand": (
                "natural-cohort intervention policy with structural no-action "
                "where a matched balanced random control is undefined"
            ),
            "prohibited": original._protocol()["claim_boundary"]["prohibited"],
        },
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def _append_amendment_notice(amendment_id: str) -> None:
    path = REPOSITORY_ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- p3b-singleton-recovery-{amendment_id} -->"
    if marker in text:
        return
    lines = [
        "",
        marker,
        "## P3b singleton recovery sealed",
        "",
        f"- Recovery amendment: `{amendment_id}`",
        f"- Interrupted prefix retained byte-for-byte: {EXPECTED_COMPLETED_STATES} states / {EXPECTED_COMPLETED_FUTURES:,} futures.",
        f"- Failure: `{EXPECTED_FAILURE}` at `{EXPECTED_NEXT_CASE}`.",
        "- Universal recovery: states with fewer than two occupied types remain in the cohort and use all-arm structural NOOP.",
        "- No checkpoint outcome was loaded, no inference was run, and no state or matrix was removed or replaced before this amendment was sealed.",
        "",
    ]
    path.write_text(text + "\n".join(lines), encoding="utf-8")


def prepare_amendment(
    work_directory: Path,
    output_directory: Path,
    amendment_directory: Path,
) -> None:
    work = work_directory.resolve()
    output = output_directory.resolve()
    amendment_directory = amendment_directory.resolve()
    if amendment_directory.exists():
        raise FileExistsError(f"refusing to overwrite {amendment_directory}")
    if output.exists():
        raise FileExistsError("P3b result already exists; recovery is not pre-inference")
    if (work / "replay").exists():
        raise FileExistsError("P3b replay already exists; recovery boundary changed")
    registration = original.verify_registration(ORIGINAL_REGISTRATION)
    if registration["registration_id"] != EXPECTED_ORIGINAL_REGISTRATION_ID:
        raise ValueError("unexpected original P3b registration")
    prefix = _checkpoint_prefix_audit(work, require_interrupted_boundary=True)
    protocol = _protocol(prefix)
    command = [
        str(REPOSITORY_ROOT / ".venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "tests/test_intervention_p3b_singleton_recovery.py",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "P3b singleton recovery tests failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    with _atomic_destination(amendment_directory) as directory:
        (directory / "recovery_protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (directory / "checkpoint_prefix_audit.json").write_text(
            json.dumps(prefix, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (directory / "pytest_output.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        payload: dict[str, Any] = {
            "format": AMENDMENT_FORMAT,
            "status": "sealed_after_partial_generation_before_outcome_loading_replay_or_inference",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(directory / "recovery_protocol.json"),
            "checkpoint_prefix_audit_sha256": sha256_file(
                directory / "checkpoint_prefix_audit.json"
            ),
            "source_hashes": _source_hashes(),
            "original_registration_id": registration["registration_id"],
            "original_registration_checksum_manifest_sha256": sha256_file(
                ORIGINAL_REGISTRATION / "SHA256SUMS"
            ),
            "immutable_prefix_aggregate_sha256": prefix[
                "immutable_prefix_aggregate_sha256"
            ],
            "immutable_prefix_checkpoints": EXPECTED_COMPLETED_STATES,
            "completed_scientific_futures": EXPECTED_COMPLETED_FUTURES,
            "checkpoint_outcomes_loaded": False,
            "scientific_effects_computed": False,
            "replay_started": False,
        }
        payload["amendment_id"] = _canonical_digest(payload)
        (directory / "amendment_registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(directory)
    amendment = verify_amendment(amendment_directory)
    _append_amendment_notice(amendment["amendment_id"])
    print(
        f"P3b singleton recovery amendment sealed: {amendment['amendment_id']}",
        flush=True,
    )


def verify_amendment(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads(
        (directory / "amendment_registration.json").read_text(encoding="utf-8")
    )
    identifier = payload.pop("amendment_id")
    if (
        payload.get("format") != AMENDMENT_FORMAT
        or payload.get("status")
        != "sealed_after_partial_generation_before_outcome_loading_replay_or_inference"
        or _canonical_digest(payload) != identifier
    ):
        raise ValueError("invalid P3b singleton recovery amendment")
    payload["amendment_id"] = identifier
    if payload["source_hashes"] != _source_hashes():
        raise ValueError("P3b singleton recovery source changed after sealing")
    registration = original.verify_registration(ORIGINAL_REGISTRATION)
    if registration["registration_id"] != payload["original_registration_id"]:
        raise ValueError("original P3b registration changed")
    prefix = json.loads(
        (directory / "checkpoint_prefix_audit.json").read_text(encoding="utf-8")
    )
    current = _checkpoint_prefix_audit(
        DEFAULT_WORK, require_interrupted_boundary=False
    )
    immutable_fields = (
        "original_checkpoint_contract_sha256",
        "immutable_state_checkpoint_count",
        "immutable_state_checkpoint_hashes",
        "immutable_prefix_aggregate_sha256",
        "completed_futures",
        "next_state_index",
        "next_state_id",
    )
    if any(prefix[name] != current[name] for name in immutable_fields):
        raise ValueError("an immutable pre-recovery checkpoint changed")
    protocol = json.loads(
        (directory / "recovery_protocol.json").read_text(encoding="utf-8")
    )
    if protocol != json.loads(json.dumps(_json_ready(_protocol(prefix)))):
        raise ValueError("P3b singleton recovery protocol changed")
    if (
        protocol["protocol_id"] != payload["protocol_id"]
        or sha256_file(directory / "recovery_protocol.json")
        != payload["protocol_sha256"]
    ):
        raise ValueError("P3b singleton recovery protocol digest changed")
    return payload


def _structural_no_action_batch(
    arguments: tuple[StateCase, ExperimentConfig, original.BridgeSpec, str]
) -> base.PhaseBatch:
    case, experiment, spec, model_path = arguments
    if np.count_nonzero(case.snapshot.composition) >= 2:
        return original._phase_worker(arguments)
    limiter = threadpool_limits(limits=1)
    try:
        predictor = FrozenFullPredictor.load(model_path)
        prediction = predictor.predict_snapshot(
            case.candidate,
            case.snapshot,
            case.beta,
            experiment.gard,
        )
        predictions = np.full(len(spec.arms), prediction, dtype=np.float64)
        arm_outcomes: list[list[Any]] = [[] for _ in spec.arms]
        for branch in range(spec.branches):
            seed = original._future_seed(spec, case, branch)
            for arm_index in range(len(spec.arms)):
                arm_outcomes[arm_index].append(
                    simulate_one_shot(
                        case.snapshot,
                        case.beta,
                        case.candidate,
                        experiment.gard,
                        original.HORIZON,
                        np.random.default_rng(seed),
                        None,
                    )
                )
        outcomes = tuple(tuple(arm) for arm in arm_outcomes)
        for branch in range(spec.branches):
            digests = {outcomes[arm][branch].record_digest for arm in range(len(spec.arms))}
            if len(digests) != 1:
                raise AssertionError("structural no-action arms diverged")
        return base.PhaseBatch(
            state_id=case.state_id,
            state_digest=base._snapshot_digest(case),
            arm_names=spec.arms,
            predictions=predictions,
            selected_edits=tuple(None for _ in spec.arms),
            surgeries=tuple(None for _ in spec.arms),
            scored_edits=tuple(),
            catalytic_support=np.empty(0, dtype=np.float64),
            outcomes=outcomes,
        )
    finally:
        limiter.restore_original_limits()


def _recovery_contract(
    cases: list[StateCase],
    spec: original.BridgeSpec,
    registration_id: str,
    amendment_id: str,
    stage: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": RECOVERY_CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "amendment_id": amendment_id,
        "phase": spec.phase,
        "stage": stage,
        "case_ids": [case.state_id for case in cases],
        "case_digests": [base._snapshot_digest(case) for case in cases],
        "arms": list(spec.arms),
        "branches": spec.branches,
        "structural_no_action_condition": "occupied_types < 2",
        "eligible_worker_unchanged": True,
        "future_seed_includes_arm": False,
        "source_hashes": _source_hashes(),
    }
    value["contract_id"] = _canonical_digest(value)
    return value


def run_recovery_batches(
    cases: list[StateCase],
    experiment: ExperimentConfig,
    spec: original.BridgeSpec,
    model_path: Path,
    registration_id: str,
    amendment_id: str,
    checkpoint_directory: Path,
    workers: int,
    stage: str,
) -> list[base.PhaseBatch]:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    original_contract = original._checkpoint_contract(
        cases, spec, registration_id, stage
    )
    original_path = checkpoint_directory / "checkpoint_contract.json"
    if original_path.exists():
        if json.loads(original_path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(_json_ready(original_contract))
        ):
            raise ValueError("original P3b checkpoint contract changed")
    else:
        base._atomic_json(original_path, original_contract)
    recovery_contract = _recovery_contract(
        cases, spec, registration_id, amendment_id, stage
    )
    recovery_path = checkpoint_directory / "singleton_recovery_contract.json"
    if recovery_path.exists():
        if json.loads(recovery_path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(_json_ready(recovery_contract))
        ):
            raise ValueError("P3b singleton recovery checkpoint contract changed")
    else:
        base._atomic_json(recovery_path, recovery_contract)

    batches: list[base.PhaseBatch | None] = [None] * len(cases)
    missing: list[int] = []
    for index, case in enumerate(cases):
        path = checkpoint_directory / f"state_{index:04d}.pkl"
        if path.is_file():
            with path.open("rb") as handle:
                batch = pickle.load(handle)
            if (
                not isinstance(batch, base.PhaseBatch)
                or batch.state_id != case.state_id
                or batch.state_digest != base._snapshot_digest(case)
                or batch.arm_names != spec.arms
            ):
                raise ValueError(f"invalid recovered P3b checkpoint {path}")
            batches[index] = batch
        else:
            missing.append(index)

    def status(state: str) -> None:
        complete = sum(batch is not None for batch in batches)
        base._atomic_json(
            checkpoint_directory / "status.json",
            {
                "format": RECOVERY_CHECKPOINT_FORMAT,
                "phase": spec.phase,
                "stage": stage,
                "state": state,
                "states_complete": complete,
                "states_total": len(cases),
                "percent_complete": 100.0 * complete / len(cases),
                "futures_complete": complete * len(spec.arms) * spec.branches,
                "futures_total": len(cases) * len(spec.arms) * spec.branches,
                "reused_original_prefix_checkpoints": min(
                    complete, EXPECTED_COMPLETED_STATES
                )
                if stage == "generate"
                else 0,
                "singleton_recovery_amendment_id": amendment_id,
                "checkpoint_directory": str(checkpoint_directory),
            },
        )

    status("running" if missing else "complete")
    arguments = [
        (cases[index], experiment, spec, str(model_path)) for index in missing
    ]
    if workers <= 1:
        generated = map(_structural_no_action_batch, arguments)
        for index, batch in zip(missing, generated, strict=True):
            batches[index] = batch
            base._atomic_pickle(checkpoint_directory / f"state_{index:04d}.pkl", batch)
            status("running")
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            generated = executor.map(_structural_no_action_batch, arguments, chunksize=1)
            for index, batch in zip(missing, generated, strict=True):
                batches[index] = batch
                base._atomic_pickle(
                    checkpoint_directory / f"state_{index:04d}.pkl", batch
                )
                status("running")
    status("complete")
    if any(batch is None for batch in batches):
        raise AssertionError("P3b recovery stage has missing states")
    return [batch for batch in batches if batch is not None]


def _audit_interventions(
    cases: list[StateCase], batches: list[base.PhaseBatch]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    structural_mask = [
        np.count_nonzero(case.snapshot.composition) < 2 for case in cases
    ]
    eligible_cases = [case for case, structural in zip(cases, structural_mask) if not structural]
    eligible_batches = [
        batch for batch, structural in zip(batches, structural_mask) if not structural
    ]
    surgery_rows, eligible_summary = original._surgery_audit(
        eligible_cases, eligible_batches
    )
    structural_rows: list[dict[str, Any]] = []
    for index, (case, batch, structural) in enumerate(
        zip(cases, batches, structural_mask, strict=True)
    ):
        if not structural:
            continue
        predictions_exact = bool(
            np.array_equal(batch.predictions, np.full(len(batch.arm_names), batch.predictions[0]))
        )
        all_surgeries_none = all(surgery is None for surgery in batch.surgeries)
        all_edits_none = all(edit is None for edit in batch.selected_edits)
        branch_exact = True
        for branch in range(len(batch.outcomes[0])):
            records = [arm[branch].record_digest for arm in batch.outcomes]
            branch_exact &= len(set(records)) == 1
        structural_rows.append(
            {
                "state_index": index,
                "state_id": case.state_id,
                "candidate": case.candidate,
                "matrix_id": case.matrix_id,
                "landmark": case.landmark,
                "occupied_types": int(np.count_nonzero(case.snapshot.composition)),
                "mass": int(case.snapshot.composition.sum()),
                "all_arm_predictions_bitwise_exact": predictions_exact,
                "all_arm_surgeries_none": all_surgeries_none,
                "all_arm_edits_none": all_edits_none,
                "all_arm_branch_records_bitwise_exact": branch_exact,
                "zero_contribution_to_every_paired_contrast": bool(
                    predictions_exact and all_surgeries_none and branch_exact
                ),
            }
        )
    structural_frame = pd.DataFrame(structural_rows)
    structural_pass = bool(
        not structural_frame.empty
        and structural_frame[
            [
                "all_arm_predictions_bitwise_exact",
                "all_arm_surgeries_none",
                "all_arm_edits_none",
                "all_arm_branch_records_bitwise_exact",
                "zero_contribution_to_every_paired_contrast",
            ]
        ]
        .to_numpy(dtype=bool)
        .all()
    )
    summary = {
        "format": "codex-intervention-p3b-singleton-recovery-audit-v1",
        "total_states": len(cases),
        "eligible_surgery_states": len(eligible_cases),
        "structural_no_action_states": len(structural_frame),
        "structural_no_action_state_ids": structural_frame["state_id"].tolist(),
        "structural_no_action_audit_pass": structural_pass,
        "eligible_surgery_audit": eligible_summary,
        "maximum_norm_relative_error": eligible_summary[
            "maximum_norm_relative_error"
        ],
        "all_registered_states_retained": len(eligible_cases)
        + len(structural_frame)
        == len(cases),
    }
    if not (
        structural_pass
        and summary["all_registered_states_retained"]
        and eligible_summary["random_control_location_norm_positivity_audit_pass"]
    ):
        raise AssertionError("P3b singleton recovery intervention audit failed")
    return surgery_rows, structural_frame, summary


def _scope_rows(scope: dict[str, Any]) -> list[str]:
    rows = [
        "| Cell | Loosen−tighten | 95% CI | Holm p | Random 90% CI | Pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in scope["cells"]:
        effect = cell["contrasts"]["fable_effect"]
        random_ci = cell["fable_random_noop_equivalence"]["bootstrap_ci90"]
        rows.append(
            f"| {cell['cell']} | {effect['estimate']:.6f} | "
            f"[{effect['bootstrap_ci95'][0]:.6f}, {effect['bootstrap_ci95'][1]:.6f}] | "
            f"{cell['fable_effect_randomization_p_holm']:.6g} | "
            f"[{random_ci[0]:.6f}, {random_ci[1]:.6f}] | "
            f"{cell['primary_fable_cell_pass_without_replay']} |"
        )
    return rows


def _technical_report(
    metrics: dict[str, Any],
    replay: dict[str, Any],
    audit: dict[str, Any],
    registration_id: str,
    amendment_id: str,
) -> str:
    return "\n".join(
        [
            "# P3b beta-surgery dose bridge — singleton-recovered result",
            "",
            "## Recovery disclosure",
            "",
            f"The original run stopped after {EXPECTED_COMPLETED_STATES} states because a balanced random surgery is undefined on a one-entry present-present block. The recovery was sealed before replay or inference. All {audit['structural_no_action_states']} structurally ineligible states remained in the cohort and used unchanged beta in every arm, producing a zero paired contribution.",
            "",
            f"Original registration: `{registration_id}`. Recovery amendment: `{amendment_id}`.",
            "",
            "## Registered outcome",
            "",
            f"Landmark-60 Fable-strength replication gate: **{metrics['primary_replication_gate_pass']}**.",
            f"Five-landmark generalization gate: **{metrics['five_landmark_generalization_gate_pass']}**.",
            f"Landmark-60 two-dose ordering gate: **{metrics['landmark60_two_dose_gate_pass']}**.",
            f"Five-landmark two-dose ordering gate: **{metrics['five_landmark_two_dose_gate_pass']}**.",
            f"Exact replay: **{replay['state_edit_endpoint_and_process_digests_exact']}**.",
            "",
            "## Primary landmark 60",
            "",
            *_scope_rows(metrics["primary"]),
            "",
            "## Five-landmark generalization",
            "",
            *_scope_rows(metrics["generalization"]),
            "",
            "## Audit",
            "",
            f"- Structural no-action audit: **{audit['structural_no_action_audit_pass']}**.",
            f"- Eligible surgery states: {audit['eligible_surgery_states']}.",
            f"- Structural no-action states: {audit['structural_no_action_states']}.",
            "- Every matrix and state was retained; none was replaced or retried.",
            f"- Reused original generation checkpoints: {EXPECTED_COMPLETED_STATES}, unchanged byte-for-byte.",
            "- Every state and arm was included in the complete deterministic replay.",
            "",
            "## Boundary and stop",
            "",
            "The estimand is the registered natural-cohort policy with structural no-action where a matched balanced random control is undefined. This cannot establish life, biological memory, autonomy, real chemistry, Phi/PhiID intervention, or strict-eight control. The result is sealed and no later phase launches automatically.",
            "",
        ]
    )


def _lay_report(
    metrics: dict[str, Any], replay: dict[str, Any], audit: dict[str, Any]
) -> str:
    verdict = (
        "The correctly sized intervention passed all four primary candidate-and-half tests."
        if metrics["primary_replication_gate_pass"]
        else "The correctly sized intervention did not pass all four primary candidate-and-half tests."
    )
    return "\n".join(
        [
            "# Lay summary",
            "",
            f"A small number of saved assemblies had collapsed to fewer than two molecular types. In such a state there is no honest way to make the planned balanced random network change: there is only one active network link. We therefore kept all {audit['structural_no_action_states']} such states but changed nothing in any arm, making their contribution to every comparison exactly zero.",
            "",
            "This rule was written, tested, and sealed before looking at event rates or comparing arms. The 363 already completed states were reused unchanged, and every state was included in a full second replay.",
            "",
            verdict,
            f" Exact replay passed: **{replay['state_edit_endpoint_and_process_digests_exact']}**.",
            "",
            "The result concerns causal control of a simplified simulated heredity process. It does not show that the assemblies are alive or autonomously remember or repair themselves.",
            "",
        ]
    )


def _append_recovered_result_notice(
    output: Path,
    metrics: dict[str, Any],
    audit: dict[str, Any],
    amendment_id: str,
) -> None:
    path = REPOSITORY_ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- sealed-p3b-singleton-recovered-{amendment_id} -->"
    if marker in text:
        return
    lines = [
        "",
        marker,
        "## P3b singleton-recovered result",
        "",
        f"- Recovery amendment: `{amendment_id}`",
        f"- Result bundle: `{output.relative_to(REPOSITORY_ROOT)}`",
        f"- Structural no-action states retained: **{audit['structural_no_action_states']}**",
        f"- Landmark-60 primary gate: **{metrics['primary_replication_gate_pass']}**",
        f"- Five-landmark generalization gate: **{metrics['five_landmark_generalization_gate_pass']}**",
        f"- Landmark-60 two-dose gate: **{metrics['landmark60_two_dose_gate_pass']}**",
        f"- Five-landmark two-dose gate: **{metrics['five_landmark_two_dose_gate_pass']}**",
        "- Exact replay and structural no-action audits passed.",
        "- No later phase launched automatically.",
        "",
    ]
    path.write_text(text + "\n".join(lines), encoding="utf-8")


def recover(
    amendment_directory: Path,
    output_directory: Path,
    work_directory: Path,
    workers: int,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    amendment = verify_amendment(amendment_directory)
    registration = original.verify_registration(ORIGINAL_REGISTRATION)
    output = output_directory.resolve()
    work = work_directory.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    spec = original.phase_spec()
    experiment = original._experiment(spec)
    original._prepare_campaign(work, output, registration, spec)

    print("[p3b recovery 1/8] Rebuilding and verifying the frozen 960-state cohort", flush=True)
    original._campaign_status(work, "running", "rebuilding_and_verifying_frozen_cohort")
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, original.LABEL, experiment.confirmation)
    contract = json.loads(
        (work / "generate/checkpoint_contract.json").read_text(encoding="utf-8")
    )
    if (
        [case.state_id for case in cases] != contract["case_ids"]
        or [base._snapshot_digest(case) for case in cases] != contract["case_digests"]
    ):
        raise ValueError("rebuilt P3b cohort differs from the frozen failed campaign")
    structural_count = sum(
        np.count_nonzero(case.snapshot.composition) < 2 for case in cases
    )
    if structural_count < 1:
        raise AssertionError("P3b recovery found no structural no-action state")

    model_path = ORIGINAL_REGISTRATION / "frozen_full_predictor.npz"
    futures = len(cases) * len(spec.arms) * spec.branches
    print(
        f"[p3b recovery 2/8] Reusing {EXPECTED_COMPLETED_STATES} checkpoints and completing {futures:,} primary futures",
        flush=True,
    )
    original._campaign_status(work, "running", "resuming_primary_generation")
    generated = run_recovery_batches(
        cases,
        experiment,
        spec,
        model_path,
        registration["registration_id"],
        amendment["amendment_id"],
        work / "generate",
        workers,
        "generate",
    )
    verify_amendment(amendment_directory)

    print(f"[p3b recovery 3/8] Replaying all {futures:,} futures", flush=True)
    original._campaign_status(work, "running", "complete_exact_replay")
    replayed = run_recovery_batches(
        cases,
        experiment,
        spec,
        model_path,
        registration["registration_id"],
        amendment["amendment_id"],
        work / "replay",
        workers,
        "replay",
    )
    replay = base.replay_audit(generated, replayed)
    replay_exact = replay["state_edit_endpoint_and_process_digests_exact"]
    if not replay_exact:
        raise AssertionError("P3b singleton recovery replay failed")

    print("[p3b recovery 4/8] Auditing interventions and computing frozen inference", flush=True)
    original._campaign_status(work, "running", "auditing_and_computing_inference")
    arrays = base._outcome_arrays(cases, generated, spec)
    draws = generate_inference_draws(
        spec.matrices,
        original.BOOTSTRAP_REPETITIONS,
        original.RANDOMIZATION_REPETITIONS,
        np.random.default_rng(
            derive_seed(spec.bootstrap_seed, f"{original.LABEL}.bootstrap")
        ),
        np.random.default_rng(
            derive_seed(spec.randomization_seed, f"{original.LABEL}.randomization")
        ),
    )
    metrics, matrix_rows = original.compute_bridge_inference(
        cases, arrays["targets"], arrays["predictions"], draws
    )
    original.add_replay_gates(metrics, replay_exact)
    secondary = base._secondary_descriptives(cases, arrays, spec)
    surgery_rows, structural_rows, recovery_audit = _audit_interventions(
        cases, generated
    )

    print("[p3b recovery 5/8] Writing and readback-checking complete artifacts", flush=True)
    original._campaign_status(work, "running", "writing_and_readback_checking_artifacts")
    with _atomic_destination(output) as directory:
        np.savez_compressed(directory / "branch_arrays.npz", **arrays)
        base._write_branch_table(directory / "branches.csv.gz", cases, generated)
        base._write_state_artifacts(directory, cases, generated, arrays)
        base._write_selection_artifacts(directory, cases, generated, spec)
        surgery_rows.to_csv(directory / "surgery_norm_audit.csv", index=False)
        structural_rows.to_csv(
            directory / "structural_no_action_states.csv", index=False
        )
        (directory / "recovery_audit.json").write_text(
            json.dumps(_json_ready(recovery_audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        original._write_inference_arrays(
            directory / "inference_arrays.npz", draws, metrics
        )
        pd.DataFrame(matrix_rows).to_csv(directory / "matrix_effects.csv", index=False)
        (directory / "primary_metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (directory / "secondary_outcomes.json").write_text(
            json.dumps(_json_ready(secondary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (directory / "replay_audit.json").write_text(
            json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        readback = original._readback_metrics(
            directory, cases, metrics, matrix_rows, replay_exact
        )
        (directory / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (directory / "SCIENTIFIC_REPORT.md").write_text(
            _technical_report(
                metrics,
                replay,
                recovery_audit,
                registration["registration_id"],
                amendment["amendment_id"],
            ),
            encoding="utf-8",
        )
        (directory / "LAY_SUMMARY.md").write_text(
            _lay_report(metrics, replay, recovery_audit), encoding="utf-8"
        )
        supported: list[str] = []
        failed: list[str] = []
        decisions = (
            (
                "landmark-60 qualitative cross-clean-room beta-surgery replication",
                metrics["primary_replication_gate_pass"],
            ),
            (
                "five-landmark beta-surgery generalization",
                metrics["five_landmark_generalization_gate_pass"],
            ),
            (
                "landmark-60 graded two-dose ordering",
                metrics["landmark60_two_dose_gate_pass"],
            ),
            (
                "five-landmark graded two-dose ordering",
                metrics["five_landmark_two_dose_gate_pass"],
            ),
        )
        for statement, passed in decisions:
            (supported if passed else failed).append(statement)
        claim_boundary = {
            "supported_claims": supported,
            "failed_predictions": failed,
            "deviations": [
                "prospectively sealed structural no-action recovery for states with fewer than two occupied types"
            ],
            "unresolved_questions": [
                "whether network surgery acts mainly on break resistance or post-break renewal",
                "whether repeated intervention can maintain heredity over longer horizons",
            ],
            "prohibited_interpretations": original._protocol()["claim_boundary"][
                "prohibited"
            ],
        }
        (directory / "claim_boundaries.json").write_text(
            json.dumps(claim_boundary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format": RESULT_FORMAT,
            "phase": spec.phase,
            "role": spec.role,
            "original_registration_id": registration["registration_id"],
            "singleton_recovery_amendment_id": amendment["amendment_id"],
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "landmarks": list(original.LANDMARKS),
            "states": len(cases),
            "structural_no_action_states": structural_count,
            "arms": list(spec.arms),
            "branches_per_arm_per_state": spec.branches,
            "primary_futures": futures,
            "replay_futures": futures,
            "reused_original_generation_checkpoints": EXPECTED_COMPLETED_STATES,
            "landmark60_primary_gate": metrics["primary_replication_gate_pass"],
            "five_landmark_generalization_gate": metrics[
                "five_landmark_generalization_gate_pass"
            ],
            "landmark60_two_dose_gate": metrics["landmark60_two_dose_gate_pass"],
            "five_landmark_two_dose_gate": metrics[
                "five_landmark_two_dose_gate_pass"
            ],
            "exact_replay": replay_exact,
            "complete_readback_exact": True,
            "eligible_surgery_audit_pass": True,
            "structural_no_action_audit_pass": recovery_audit[
                "structural_no_action_audit_pass"
            ],
            "all_registered_states_retained": True,
            "no_future_retries": True,
            "no_matrix_replacement": True,
            "no_refitting_or_recalibration": True,
            "mandatory_stop_after_this_stage": True,
            "next_scientific_phase_launched": False,
            "runtime": _runtime_manifest(),
            "checkpoint_audit": {
                "work_directory": str(work),
                "immutable_prefix_aggregate_sha256": amendment[
                    "immutable_prefix_aggregate_sha256"
                ],
                "generate_recovery_contract_sha256": sha256_file(
                    work / "generate/singleton_recovery_contract.json"
                ),
                "replay_recovery_contract_sha256": sha256_file(
                    work / "replay/singleton_recovery_contract.json"
                ),
            },
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (directory / "CUMULATIVE_RESULTS_LEDGER.md").write_text(
            "\n".join(
                [
                    "# Intervention result ledger snapshot",
                    "",
                    "Phase: `p3b_dose_bridge` (singleton-recovered)",
                    f"Original registration: `{registration['registration_id']}`",
                    f"Recovery amendment: `{amendment['amendment_id']}`",
                    f"Structural no-action states: **{structural_count}**",
                    f"Landmark-60 primary gate: **{metrics['primary_replication_gate_pass']}**",
                    f"Five-landmark generalization gate: **{metrics['five_landmark_generalization_gate_pass']}**",
                    "Exact replay: **True**",
                    "Next phase: not launched; mandatory review stop.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print("[p3b recovery 6/8] Sealing and checksum-verifying result", flush=True)
        write_checksums(directory)
    verify_checksums(output)
    verify_amendment(amendment_directory)
    original._append_result_ledger(
        output, metrics, replay, registration["registration_id"]
    )
    _append_recovered_result_notice(
        output, metrics, recovery_audit, amendment["amendment_id"]
    )
    original._campaign_status(work, "sealed_complete", "singleton_recovered_mandatory_review_stop")
    print(f"[p3b recovery 7/8] Result sealed: {output}", flush=True)
    print("[p3b recovery 8/8] STOPPED as amended; no later phase launched", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3b singleton recovery")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    verify = commands.add_parser("verify")
    verify.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    run = commands.add_parser("recover")
    run.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    run.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    status = commands.add_parser("status")
    status.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        prepare_amendment(arguments.work_dir, arguments.output, arguments.amendment)
    elif arguments.command == "verify":
        print(
            json.dumps(
                verify_amendment(arguments.amendment), indent=2, sort_keys=True
            )
        )
    elif arguments.command == "recover":
        recover(
            arguments.amendment,
            arguments.output,
            arguments.work_dir,
            arguments.workers,
        )
    elif arguments.command == "status":
        print(json.dumps(base.read_status(arguments.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
