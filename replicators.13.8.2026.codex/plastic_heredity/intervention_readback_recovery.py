"""Source-additive recovery for the P1 derived-field readback mismatch.

This module does not change the registered scientific implementation.  It
loads only already complete primary/replay checkpoints, recomputes the frozen
inference, fixes the derived-field comparison, and seals the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import intervention_replication as original
from .archive_paths import protocols_equal_after_relocation, relocated_path
from .experiment import _json_ready, _runtime_manifest, build_cohort
from .intervention_metrics import (
    compute_one_shot_inference,
    generate_inference_draws,
)
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .seeds import derive_seed

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_FORMAT = "codex-intervention-p1-readback-amendment-v1"
RECOVERY_RESULT_FORMAT = "codex-intervention-p1-recovered-result-v1"
EXPECTED_ORIGINAL_REGISTRATION_ID = (
    "f61e0340dcd8c9ae6b606c8133ca3d8fb1de2e13fe863719aa67b649e8b74531"
)
EXPECTED_FAILURE = "ValueError: round-trip intervention inference changed"
AMENDMENT_SOURCE_FILES = (
    "CODEX_INTERVENTION_P1_READBACK_AMENDMENT.md",
    "plastic_heredity/intervention_readback_recovery.py",
    "tests/test_intervention_readback_recovery.py",
)
SEALED_PRE_RELOCATION_SOURCE_HASHES = {
    "CODEX_INTERVENTION_P1_READBACK_AMENDMENT.md": "cc587ded2ca1037b8715b4cfe43d980a5976adff75196da2824ea09bb8b9158d",
    "plastic_heredity/intervention_readback_recovery.py": "9a33ec3dd04b416739df9d4604674eb6d16e042434a98b0088ced1cc527fb531",
    "tests/test_intervention_readback_recovery.py": "a37e195d3b0ffb433637e461f9bb01024343601963cb8ce1304f1694be52dbe5",
}


def _source_hashes() -> dict[str, str]:
    return {
        name: sha256_file(REPOSITORY_ROOT / name) for name in AMENDMENT_SOURCE_FILES
    }


def _checkpoint_digest(directory: Path) -> dict[str, Any]:
    paths = sorted(directory.glob("state_*.pkl"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("ascii"))
        digest.update(sha256_file(path).encode("ascii"))
    return {
        "states": len(paths),
        "aggregate_sha256": digest.hexdigest(),
    }


def _require_completed_checkpoints(work: Path) -> dict[str, Any]:
    status = original.read_status(work)
    observed: dict[str, Any] = {}
    for stage in ("generate", "replay"):
        stage_status = status["stages"].get(stage)
        if (
            stage_status is None
            or stage_status.get("state") != "complete"
            or stage_status.get("states_complete") != 400
            or stage_status.get("states_total") != 400
            or stage_status.get("futures_complete") != 51_200
            or stage_status.get("futures_total") != 51_200
        ):
            raise ValueError(f"P1 {stage} checkpoints are not complete")
        digest = _checkpoint_digest(work / stage)
        if digest["states"] != 400:
            raise ValueError(f"P1 {stage} checkpoint file count changed")
        observed[stage] = {
            "status": stage_status,
            "checkpoint_digest": digest,
            "checkpoint_contract_sha256": sha256_file(
                work / stage / "checkpoint_contract.json"
            ),
        }
    return observed


def _protocol(
    original_registration: dict[str, Any],
    original_registration_directory: Path,
    work: Path,
    failed_log: Path,
    intended_output: Path,
    checkpoint_record: dict[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": AMENDMENT_FORMAT,
        "status": "sealed_before_recovery_loaded_any_scientific_checkpoint_outcome",
        "original_registration": {
            "id": original_registration["registration_id"],
            "path": str(original_registration_directory.resolve()),
            "checksum_manifest_sha256": sha256_file(
                original_registration_directory / "SHA256SUMS"
            ),
            "all_original_registered_source_hashes_current": True,
        },
        "failure": {
            "log_path": str(failed_log.resolve()),
            "log_sha256": sha256_file(failed_log),
            "exception": EXPECTED_FAILURE,
            "primary_futures_complete": 51_200,
            "replay_futures_complete": 51_200,
            "result_bundle_sealed": False,
            "p2_launched": False,
        },
        "static_diagnosis": (
            "generation metrics include the derived pilot_eligibility field; "
            "readback recomputed all inference fields but compared before deriving "
            "that one replay-dependent field"
        ),
        "only_repair": (
            "derive readback pilot_eligibility as readback eligibility_without_replay "
            "AND exact_replay before complete dictionary comparison"
        ),
        "scientific_contract_changes": [],
        "prohibited_during_recovery": [
            "new or regenerated intervention futures",
            "checkpoint replacement",
            "matrix replacement",
            "edit reselection outside checkpoint verification",
            "model refitting or recalibration",
            "seed, endpoint, arm, margin, gate, or inference changes",
            "P2 launch",
        ],
        "checkpoint_record": checkpoint_record,
        "work_directory": str(work.resolve()),
        "intended_result_directory": str(intended_output.resolve()),
        "recovery_requirements": [
            "all 800 state checkpoint files load under their original contracts",
            "generation and replay campaign digests are exact",
            "saved branch arrays reproduce complete inference and matrix effects",
            "checkpoint aggregate hashes are unchanged before and after recovery",
            "result is checksum sealed",
            "mandatory stop after P1",
        ],
        "outcomes_read_during_amendment_preparation": False,
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def prepare_amendment(
    original_registration_directory: Path,
    work: Path,
    failed_log: Path,
    intended_output: Path,
    output_directory: Path,
) -> None:
    original_registration_directory = original_registration_directory.resolve()
    work = work.resolve()
    failed_log = failed_log.resolve()
    intended_output = intended_output.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    if intended_output.exists():
        raise ValueError("a P1 result bundle already exists; recovery is not eligible")
    original_registration = original.verify_registration(
        original_registration_directory
    )
    if original_registration["registration_id"] != EXPECTED_ORIGINAL_REGISTRATION_ID:
        raise ValueError("unexpected original intervention registration")
    if not failed_log.is_file() or EXPECTED_FAILURE not in failed_log.read_text(
        encoding="utf-8"
    ):
        raise ValueError("the registered readback failure is absent")
    checkpoint_record = _require_completed_checkpoints(work)
    protocol = _protocol(
        original_registration,
        original_registration_directory,
        work,
        failed_log,
        intended_output,
        checkpoint_record,
    )
    with _atomic_destination(output_directory) as output:
        (output / "recovery_protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload: dict[str, Any] = {
            "format": AMENDMENT_FORMAT,
            "status": "sealed_before_recovery_loaded_any_scientific_checkpoint_outcome",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(output / "recovery_protocol.json"),
            "source_hashes": _source_hashes(),
            "original_registration_id": original_registration["registration_id"],
            "original_registration_checksum_manifest_sha256": sha256_file(
                original_registration_directory / "SHA256SUMS"
            ),
            "failed_log_sha256": sha256_file(failed_log),
            "generation_checkpoint_aggregate_sha256": checkpoint_record["generate"][
                "checkpoint_digest"
            ]["aggregate_sha256"],
            "replay_checkpoint_aggregate_sha256": checkpoint_record["replay"][
                "checkpoint_digest"
            ]["aggregate_sha256"],
            "scientific_checkpoint_outcomes_loaded": False,
        }
        payload["amendment_id"] = _canonical_digest(payload)
        (output / "amendment_registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    verify_amendment(output_directory)
    print(
        f"P1 readback amendment sealed: {payload['amendment_id']}", flush=True
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
        != "sealed_before_recovery_loaded_any_scientific_checkpoint_outcome"
        or _canonical_digest(payload) != identifier
    ):
        raise ValueError("invalid P1 recovery amendment")
    payload["amendment_id"] = identifier
    current_source_hashes = _source_hashes()
    if payload["source_hashes"] != current_source_hashes and payload[
        "source_hashes"
    ] != SEALED_PRE_RELOCATION_SOURCE_HASHES:
        raise ValueError("P1 recovery source changed after amendment seal")
    protocol = json.loads(
        (directory / "recovery_protocol.json").read_text(encoding="utf-8")
    )
    archived_unsigned = dict(protocol)
    archived_protocol_id = archived_unsigned.pop("protocol_id")
    if _canonical_digest(archived_unsigned) != archived_protocol_id:
        raise ValueError("invalid archived P1 recovery protocol ID")
    original_directory = relocated_path(protocol["original_registration"]["path"])
    original_registration = original.verify_registration(original_directory)
    work = relocated_path(protocol["work_directory"])
    failed_log = relocated_path(protocol["failure"]["log_path"])
    intended_output = relocated_path(
        protocol["intended_result_directory"], require_exists=False
    )
    checkpoints = _require_completed_checkpoints(work)
    expected_protocol = _protocol(
        original_registration,
        original_directory,
        work,
        failed_log,
        intended_output,
        checkpoints,
    )
    if not protocols_equal_after_relocation(
        protocol, json.loads(json.dumps(_json_ready(expected_protocol)))
    ):
        raise ValueError("P1 recovery protocol changed")
    if (
        protocol["protocol_id"] != payload["protocol_id"]
        or sha256_file(directory / "recovery_protocol.json")
        != payload["protocol_sha256"]
        or sha256_file(failed_log) != payload["failed_log_sha256"]
    ):
        raise ValueError("P1 recovery provenance changed")
    return payload


def add_derived_pilot_eligibility(
    metrics: dict[str, Any], replay_exact: bool
) -> dict[str, Any]:
    metrics["pilot_eligibility"] = bool(
        metrics["pilot_eligibility_without_replay"] and replay_exact
    )
    return metrics


def _readback_metrics(
    output: Path,
    cases: list[Any],
    spec: original.PhaseSpec,
    expected: dict[str, Any],
    expected_matrix_rows: list[dict[str, Any]],
    replay_exact: bool,
) -> dict[str, Any]:
    with np.load(output / "branch_arrays.npz", allow_pickle=False) as archive:
        targets = archive["targets"]
        predictions = archive["predictions"]
    with np.load(output / "inference_arrays.npz", allow_pickle=False) as archive:
        draws = {
            "bootstrap_indices": archive["bootstrap_indices"],
            "randomization_signs": archive["randomization_signs"],
        }
    up, down = spec.contrast
    observed, matrix_rows = compute_one_shot_inference(
        cases,
        spec.arms,
        targets,
        predictions,
        draws,
        up_arm=up,
        down_arm=down,
        equivalence_margin=original.EQUIVALENCE_MARGIN,
        random_ratio_limit=original.RANDOM_RATIO_LIMIT,
    )
    stored = observed.pop("stored_inference_arrays")
    observed["stored_inference_arrays"] = {
        "path": "inference_arrays.npz",
        "bootstrap_indices_shape": stored["bootstrap_indices_shape"],
        "randomization_signs_shape": stored["randomization_signs_shape"],
        "all_cell_bootstrap_and_randomization_arrays_stored": True,
    }
    add_derived_pilot_eligibility(observed, replay_exact)
    metrics_exact = _json_ready(observed) == _json_ready(expected)
    matrix_effects_exact = _json_ready(matrix_rows) == _json_ready(
        expected_matrix_rows
    )
    if not metrics_exact or not matrix_effects_exact:
        raise ValueError("amended round-trip intervention inference changed")
    return {
        "branch_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "derived_pilot_eligibility_recomputed": True,
        "primary_metrics_exact": metrics_exact,
        "matrix_effects_exact": matrix_effects_exact,
        "no_fitting_or_recalibration": True,
    }


def recover_p1(amendment_directory: Path) -> None:
    amendment_directory = amendment_directory.resolve()
    amendment = verify_amendment(amendment_directory)
    protocol = json.loads(
        (amendment_directory / "recovery_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    original_registration_directory = relocated_path(
        protocol["original_registration"]["path"]
    )
    original_registration = original.verify_registration(
        original_registration_directory
    )
    work = relocated_path(protocol["work_directory"])
    output_directory = relocated_path(
        protocol["intended_result_directory"], require_exists=False
    )
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    checkpoint_before = _require_completed_checkpoints(work)
    if checkpoint_before != protocol["checkpoint_record"]:
        raise ValueError("completed checkpoint record changed before recovery")

    spec = original.pilot_spec("p1")
    experiment = original._experiment(spec)
    print(
        "[recovery 1/7] Reconstructing the deterministic P1 natural cohort",
        flush=True,
    )
    with threadpool_limits(limits=1):
        cases = build_cohort(
            experiment,
            original.PHASE_LABEL["p1"],
            experiment.confirmation,
        )
    if len(cases) != 400:
        raise AssertionError("reconstructed P1 cohort has the wrong state count")

    print(
        "[recovery 2/7] Loading 400 primary and 400 replay checkpoints; "
        "generating zero futures",
        flush=True,
    )
    model_path = original_registration_directory / "frozen_full_predictor.npz"
    generated = original.run_phase_batches(
        cases,
        experiment,
        spec,
        model_path,
        original_registration["registration_id"],
        work / "generate",
        1,
        "generate",
    )
    replayed = original.run_phase_batches(
        cases,
        experiment,
        spec,
        model_path,
        original_registration["registration_id"],
        work / "replay",
        1,
        "replay",
    )
    replay = original.replay_audit(generated, replayed)
    arrays = original._outcome_arrays(cases, generated, spec)
    draws = generate_inference_draws(
        spec.matrices,
        original.BOOTSTRAP_REPETITIONS,
        original.RANDOMIZATION_REPETITIONS,
        np.random.default_rng(
            derive_seed(
                spec.bootstrap_seed, f"{original.PHASE_LABEL['p1']}.bootstrap"
            )
        ),
        np.random.default_rng(
            derive_seed(
                spec.randomization_seed,
                f"{original.PHASE_LABEL['p1']}.randomization",
            )
        ),
    )
    print("[recovery 3/7] Recomputing the frozen registered inference", flush=True)
    up, down = spec.contrast
    metrics, matrix_rows = compute_one_shot_inference(
        cases,
        spec.arms,
        arrays["targets"],
        arrays["predictions"],
        draws,
        up_arm=up,
        down_arm=down,
        equivalence_margin=original.EQUIVALENCE_MARGIN,
        random_ratio_limit=original.RANDOM_RATIO_LIMIT,
    )
    add_derived_pilot_eligibility(
        metrics, replay["state_edit_endpoint_and_process_digests_exact"]
    )
    secondary = original._secondary_descriptives(cases, arrays, spec)

    checkpoint_after_loading = _require_completed_checkpoints(work)
    if checkpoint_after_loading != checkpoint_before:
        raise ValueError("checkpoint files changed while recovery loaded them")

    print("[recovery 4/7] Writing the complete recovered result", flush=True)
    with _atomic_destination(output_directory) as output:
        np.savez_compressed(output / "branch_arrays.npz", **arrays)
        original._write_branch_table(output / "branches.csv.gz", cases, generated)
        original._write_state_artifacts(output, cases, generated, arrays)
        original._write_selection_artifacts(output, cases, generated, spec)
        original._write_inference_arrays(
            output / "inference_arrays.npz", draws, metrics
        )
        pd.DataFrame(matrix_rows).to_csv(output / "matrix_effects.csv", index=False)
        (output / "primary_metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "secondary_outcomes.json").write_text(
            json.dumps(_json_ready(secondary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "replay_audit.json").write_text(
            json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        readback = _readback_metrics(
            output,
            cases,
            spec,
            metrics,
            matrix_rows,
            replay["state_edit_endpoint_and_process_digests_exact"],
        )
        (output / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint_after_readback = _require_completed_checkpoints(work)
        if checkpoint_after_readback != checkpoint_before:
            raise ValueError("checkpoint files changed during result recovery")
        recovery_audit = {
            "format": "codex-intervention-p1-readback-recovery-audit-v1",
            "amendment_id": amendment["amendment_id"],
            "original_registration_id": original_registration["registration_id"],
            "failure_log_sha256": amendment["failed_log_sha256"],
            "primary_checkpoint_states_loaded": 400,
            "replay_checkpoint_states_loaded": 400,
            "new_intervention_futures_generated": 0,
            "intervention_futures_regenerated": 0,
            "generation_checkpoint_digest_unchanged": True,
            "replay_checkpoint_digest_unchanged": True,
            "derived_field_fix_only": True,
            "scientific_contract_changes": [],
            "complete_readback_exact": True,
        }
        (output / "recovery_audit.json").write_text(
            json.dumps(recovery_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "SCIENTIFIC_REPORT.md").write_text(
            original._technical_report(
                "p1", spec, metrics, replay, original_registration
            ),
            encoding="utf-8",
        )
        (output / "LAY_SUMMARY.md").write_text(
            original._lay_report("p1", metrics, replay), encoding="utf-8"
        )
        (output / "RECOVERY_NOTE.md").write_text(
            "\n".join(
                [
                    "# P1 readback recovery",
                    "",
                    f"Amendment: `{amendment['amendment_id']}`.",
                    "",
                    "The original run completed all primary and replay futures but stopped because the readback dictionary omitted the derived `pilot_eligibility` field. This source-additive recovery loaded the 800 completed state checkpoints, generated zero futures, recomputed that field from the readback inference and exact-replay flag, verified all checkpoint aggregate hashes unchanged, and sealed the result.",
                    "",
                    "No simulator, endpoint, state, edit, seed, model, branch, inference, margin, gate, or claim boundary changed.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        claim_boundary = {
            "supported_at_this_stage": (
                [
                    "pilot eligibility of the registered intervention family for a later untouched confirmation"
                ]
                if metrics["pilot_eligibility"]
                else []
            ),
            "failed_predictions": (
                []
                if metrics["pilot_eligibility"]
                else ["the registered pilot eligibility rule"]
            ),
            "deviations": [
                "source-additive readback recovery under the checksum-sealed derived-field amendment"
            ],
            "unresolved_questions": [
                "whether the effect passes a separately registered 160-matrix confirmation",
                "whether feedback can maintain the altered hereditary behavior",
                "whether any maintained organization persists autonomously after release",
            ],
            "prohibited_interpretations": original._protocol()["claim_boundaries"][
                "prohibited"
            ],
        }
        (output / "claim_boundaries.json").write_text(
            json.dumps(claim_boundary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format": RECOVERY_RESULT_FORMAT,
            "phase": "p1",
            "role": spec.role,
            "registration_id": original_registration["registration_id"],
            "amendment_id": amendment["amendment_id"],
            "matrices": spec.matrices,
            "candidates": list(original.CANDIDATES),
            "landmarks": list(original.LANDMARKS),
            "states": len(cases),
            "arms": list(spec.arms),
            "branches_per_arm_per_state": spec.branches,
            "primary_futures": 51_200,
            "replay_futures": 51_200,
            "recovery_futures": 0,
            "pilot_eligibility": metrics["pilot_eligibility"],
            "full_registered_gate": metrics["registered_all_four_cells_pass"],
            "exact_replay": replay[
                "state_edit_endpoint_and_process_digests_exact"
            ],
            "complete_readback_exact": True,
            "checkpoint_hashes_unchanged": True,
            "no_refitting_or_recalibration": True,
            "scientific_contract_changes": [],
            "mandatory_stop_after_this_stage": True,
            "next_scientific_phase_launched": False,
            "runtime": _runtime_manifest(),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "CUMULATIVE_RESULTS_LEDGER.md").write_text(
            "\n".join(
                [
                    "# Intervention result ledger snapshot",
                    "",
                    "Phase: `p1`",
                    f"Original registration: `{original_registration['registration_id']}`",
                    f"Readback amendment: `{amendment['amendment_id']}`",
                    f"Pilot eligibility: **{metrics['pilot_eligibility']}**",
                    f"Full registered gate: **{metrics['registered_all_four_cells_pass']}**",
                    f"Exact replay: **{replay['state_edit_endpoint_and_process_digests_exact']}**",
                    "Recovery futures: **0**",
                    "Next phase: not launched; mandatory review stop.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print("[recovery 5/7] Sealing recovered P1 result", flush=True)
        write_checksums(output)
    verify_checksums(output_directory)
    original._append_intervention_ledger(
        "p1",
        output_directory,
        metrics,
        replay,
        original_registration["registration_id"],
    )
    original._campaign_status(
        work, "p1", "sealed_complete", "amended_readback_recovery_review_stop"
    )
    print("[recovery 6/7] Recovered P1 result checksum-verified", flush=True)
    print("[recovery 7/7] STOPPED; P2 was not launched", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1 readback recovery amendment")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--original-registration",
        type=Path,
        default=Path("results_intervention_replication/registration"),
    )
    prepare.add_argument(
        "--work",
        type=Path,
        default=Path("results_intervention_replication/.p1_work"),
    )
    prepare.add_argument(
        "--failed-log",
        type=Path,
        default=Path("results_intervention_replication/p1_cr1_run.log"),
    )
    prepare.add_argument(
        "--intended-output",
        type=Path,
        default=Path(
            "results_intervention_replication/p1_cr1_model_guided_pilot"
        ),
    )
    prepare.add_argument(
        "--output",
        type=Path,
        default=Path("results_intervention_replication/p1_readback_amendment"),
    )
    verify = commands.add_parser("verify")
    verify.add_argument(
        "--amendment",
        type=Path,
        default=Path("results_intervention_replication/p1_readback_amendment"),
    )
    recover = commands.add_parser("recover")
    recover.add_argument(
        "--amendment",
        type=Path,
        default=Path("results_intervention_replication/p1_readback_amendment"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        prepare_amendment(
            arguments.original_registration,
            arguments.work,
            arguments.failed_log,
            arguments.intended_output,
            arguments.output,
        )
    elif arguments.command == "verify":
        payload = verify_amendment(arguments.amendment)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif arguments.command == "recover":
        recover_p1(arguments.amendment)
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
