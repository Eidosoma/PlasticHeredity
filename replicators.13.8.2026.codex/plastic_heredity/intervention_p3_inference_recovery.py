"""Zero-future recovery for P3's semantic random-arm routing failure.

The registered P3 simulation and replay checkpoints were already complete
when the original runner supplied the molecular-phase default ``RANDOM`` to
the generic inference routine.  This additive module seals that failure and
routes P3's registered ``RANDOM_SURGERY`` control explicitly.  It never edits
or regenerates a scientific future.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import intervention_p3_lifecycle as lifecycle
from . import intervention_replication as original
from .archive_paths import protocols_equal_after_relocation, relocated_path
from .experiment import _json_ready, _runtime_manifest, build_cohort
from .intervention_metrics import (
    compute_one_shot_inference,
    generate_inference_draws,
)
from .intervention_readback_recovery import (
    _require_completed_checkpoints,
    add_derived_pilot_eligibility,
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
RESULT_ROOT = REPOSITORY_ROOT / "results_intervention_replication"
ORIGINAL_REGISTRATION = RESULT_ROOT / "registration"
LIFECYCLE_AMENDMENT = RESULT_ROOT / "p3_lifecycle_amendment"
DEFAULT_WORK = RESULT_ROOT / ".p3_work"
DEFAULT_FAILED_LOG = RESULT_ROOT / "p3_cr4_run.log"
DEFAULT_OUTPUT = RESULT_ROOT / "p3_cr4_beta_surgery_pilot"
DEFAULT_AMENDMENT = RESULT_ROOT / "p3_inference_recovery_amendment"
DEFAULT_AUDIT = RESULT_ROOT / "p3_cr4_beta_surgery_pilot_lifecycle_audit"

DOCUMENT = "CODEX_INTERVENTION_P3_INFERENCE_RECOVERY_AMENDMENT.md"
AMENDMENT_FORMAT = "codex-intervention-p3-inference-recovery-amendment-v1"
RESULT_FORMAT = "codex-intervention-p3-inference-recovered-result-v1"
AUDIT_FORMAT = "codex-intervention-p3-recovered-result-audit-v1"
EXPECTED_ORIGINAL_REGISTRATION_ID = (
    "f61e0340dcd8c9ae6b606c8133ca3d8fb1de2e13fe863719aa67b649e8b74531"
)
EXPECTED_LIFECYCLE_AMENDMENT_ID = (
    "679449881a33f0f40a50ca7e9de8849a1996321492a1b8190f8007f6cc22637c"
)
EXPECTED_FAILURE = "ValueError: missing registered arms: ['RANDOM']"
RANDOM_ARM = "RANDOM_SURGERY"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_p3_inference_recovery.py",
    "tests/test_intervention_p3_inference_recovery.py",
)
SEALED_PRE_RELOCATION_SOURCE_HASHES = {
    "CODEX_INTERVENTION_P3_INFERENCE_RECOVERY_AMENDMENT.md": "a12103b44223f49b664f16c4ed8bd4af161c4a31717a8ff86702bb2d3454cff7",
    "plastic_heredity/intervention_p3_inference_recovery.py": "34c418c2db20caa613a7afab6e2741887db24bcb1071e71c90ddf02c0468efd6",
    "tests/test_intervention_p3_inference_recovery.py": "9b4f01d4f59287321066cad9fb6c02e269e543f54ab14fe2df20834800d0b05c",
}


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def compute_registered_p3_inference(
    cases: list[Any],
    spec: original.PhaseSpec,
    targets: np.ndarray,
    predictions: np.ndarray,
    draws: dict[str, np.ndarray],
    *,
    inference_function: Callable[..., tuple[dict[str, Any], list[dict[str, Any]]]]
    | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the unchanged inference with P3's registered random-control name."""

    if spec.phase != "p3":
        raise ValueError("P3 recovery cannot analyze another phase")
    if spec.arms != ("LOOSEN", "TIGHTEN", RANDOM_ARM, "NOOP"):
        raise ValueError("the registered P3 arm contract changed")
    if spec.contrast != ("LOOSEN", "TIGHTEN"):
        raise ValueError("the registered P3 contrast changed")
    infer = compute_one_shot_inference if inference_function is None else inference_function
    up, down = spec.contrast
    return infer(
        cases,
        spec.arms,
        targets,
        predictions,
        draws,
        up_arm=up,
        down_arm=down,
        random_arm=RANDOM_ARM,
        equivalence_margin=original.EQUIVALENCE_MARGIN,
        random_ratio_limit=original.RANDOM_RATIO_LIMIT,
    )


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
    observed, matrix_rows = compute_registered_p3_inference(
        cases, spec, targets, predictions, draws
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
        raise ValueError("P3 recovery round-trip inference changed")
    return {
        "branch_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "semantic_random_arm": RANDOM_ARM,
        "semantic_random_arm_explicit_in_primary_and_readback": True,
        "derived_pilot_eligibility_recomputed": True,
        "primary_metrics_exact": metrics_exact,
        "matrix_effects_exact": matrix_effects_exact,
        "no_fitting_or_recalibration": True,
        "lifecycle_amendment": lifecycle.AMENDMENT_FORMAT,
        "inference_recovery_amendment": AMENDMENT_FORMAT,
    }


def _protocol(
    registration: dict[str, Any],
    lifecycle_registration: dict[str, Any],
    work: Path,
    failed_log: Path,
    intended_output: Path,
    audit_directory: Path,
    checkpoint_record: dict[str, Any],
) -> dict[str, Any]:
    spec = original.pilot_spec("p3")
    value: dict[str, Any] = {
        "format": AMENDMENT_FORMAT,
        "status": "sealed_after_complete_p3_futures_before_outcome_loading",
        "phase": "p3",
        "original_registration": {
            "id": registration["registration_id"],
            "path": str(ORIGINAL_REGISTRATION.resolve()),
            "checksum_manifest_sha256": sha256_file(
                ORIGINAL_REGISTRATION / "SHA256SUMS"
            ),
        },
        "prospective_lifecycle_amendment": {
            "id": lifecycle_registration["amendment_id"],
            "path": str(LIFECYCLE_AMENDMENT.resolve()),
            "checksum_manifest_sha256": sha256_file(
                LIFECYCLE_AMENDMENT / "SHA256SUMS"
            ),
        },
        "failure": {
            "log_path": str(failed_log.resolve()),
            "log_sha256": sha256_file(failed_log),
            "exception": EXPECTED_FAILURE,
            "stage": "first whole-matrix inference call",
            "primary_futures_complete": 51_200,
            "replay_futures_complete": 51_200,
            "scientific_estimates_computed": False,
            "result_bundle_sealed": False,
        },
        "static_diagnosis": (
            "P3 registers RANDOM_SURGERY, but its generic inference caller "
            "implicitly used the molecular-phase default RANDOM"
        ),
        "only_repair": (
            "pass random_arm='RANDOM_SURGERY' explicitly to the unchanged "
            "primary and readback inference routine"
        ),
        "scientific_contract_changes": [],
        "registered_scientific_design": {
            "arms": list(spec.arms),
            "contrast": list(spec.contrast),
            "matrices": spec.matrices,
            "branches": spec.branches,
            "landmarks": list(original.LANDMARKS),
            "horizon": original.HORIZON,
            "bootstrap_repetitions": original.BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": original.RANDOMIZATION_REPETITIONS,
            "equivalence_margin": original.EQUIVALENCE_MARGIN,
            "random_ratio_limit": original.RANDOM_RATIO_LIMIT,
        },
        "checkpoint_record": checkpoint_record,
        "work_directory": str(work.resolve()),
        "intended_result_directory": str(intended_output.resolve()),
        "audit_directory": str(audit_directory.resolve()),
        "recovery_futures": 0,
        "checkpoint_outcomes_loaded_during_amendment_preparation": False,
        "mandatory_stop_after_recovery": True,
        "confirmation_launched": False,
        "prohibited_during_recovery": [
            "new or regenerated intervention futures",
            "checkpoint or matrix replacement",
            "beta-surgery reselection",
            "seed, endpoint, arm, contrast, inference, margin, or gate changes",
            "model refitting or recalibration",
            "confirmation launch",
        ],
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def prepare_amendment(
    work: Path,
    failed_log: Path,
    intended_output: Path,
    audit_directory: Path,
    output_directory: Path,
) -> None:
    work = work.resolve()
    failed_log = failed_log.resolve()
    intended_output = intended_output.resolve()
    audit_directory = audit_directory.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    if intended_output.exists() or audit_directory.exists():
        raise ValueError("a P3 result or audit already exists; recovery is ineligible")
    registration = original.verify_registration(ORIGINAL_REGISTRATION)
    if registration["registration_id"] != EXPECTED_ORIGINAL_REGISTRATION_ID:
        raise ValueError("unexpected original intervention registration")
    lifecycle_registration = lifecycle.verify_amendment(LIFECYCLE_AMENDMENT)
    if (
        lifecycle_registration["amendment_id"]
        != EXPECTED_LIFECYCLE_AMENDMENT_ID
    ):
        raise ValueError("unexpected prospective P3 lifecycle amendment")
    if not failed_log.is_file() or EXPECTED_FAILURE not in failed_log.read_text(
        encoding="utf-8"
    ):
        raise ValueError("the registered P3 arm-routing failure is absent")
    checkpoint_record = _require_completed_checkpoints(work)

    command = [
        str(REPOSITORY_ROOT / ".venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "tests/test_intervention_p3_inference_recovery.py",
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
            "P3 inference recovery validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )

    protocol = _protocol(
        registration,
        lifecycle_registration,
        work,
        failed_log,
        intended_output,
        audit_directory,
        checkpoint_record,
    )
    with _atomic_destination(output_directory) as output:
        (output / "recovery_protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "pytest_output.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        payload: dict[str, Any] = {
            "format": AMENDMENT_FORMAT,
            "status": "sealed_after_complete_p3_futures_before_outcome_loading",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(output / "recovery_protocol.json"),
            "source_hashes": _source_hashes(),
            "original_registration_id": registration["registration_id"],
            "lifecycle_amendment_id": lifecycle_registration["amendment_id"],
            "failed_log_sha256": sha256_file(failed_log),
            "generation_checkpoint_aggregate_sha256": checkpoint_record[
                "generate"
            ]["checkpoint_digest"]["aggregate_sha256"],
            "replay_checkpoint_aggregate_sha256": checkpoint_record["replay"][
                "checkpoint_digest"
            ]["aggregate_sha256"],
            "p3_checkpoint_outcomes_loaded": False,
            "scientific_estimates_computed": False,
        }
        payload["amendment_id"] = _canonical_digest(payload)
        (output / "amendment_registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    verified = verify_amendment(output_directory)
    print(f"P3 inference recovery amendment sealed: {verified['amendment_id']}")


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
        != "sealed_after_complete_p3_futures_before_outcome_loading"
        or _canonical_digest(payload) != identifier
    ):
        raise ValueError("invalid P3 inference recovery amendment")
    payload["amendment_id"] = identifier
    current_source_hashes = _source_hashes()
    if payload["source_hashes"] != current_source_hashes and payload[
        "source_hashes"
    ] != SEALED_PRE_RELOCATION_SOURCE_HASHES:
        raise ValueError("P3 inference recovery source changed after sealing")

    protocol = json.loads(
        (directory / "recovery_protocol.json").read_text(encoding="utf-8")
    )
    archived_unsigned = dict(protocol)
    archived_protocol_id = archived_unsigned.pop("protocol_id")
    if _canonical_digest(archived_unsigned) != archived_protocol_id:
        raise ValueError("invalid archived P3 recovery protocol ID")
    registration = original.verify_registration(
        relocated_path(protocol["original_registration"]["path"])
    )
    lifecycle_registration = lifecycle.verify_amendment(
        relocated_path(protocol["prospective_lifecycle_amendment"]["path"])
    )
    checkpoint_record = _require_completed_checkpoints(
        relocated_path(protocol["work_directory"])
    )
    expected = _protocol(
        registration,
        lifecycle_registration,
        relocated_path(protocol["work_directory"]),
        relocated_path(protocol["failure"]["log_path"]),
        relocated_path(protocol["intended_result_directory"]),
        relocated_path(protocol["audit_directory"]),
        checkpoint_record,
    )
    if not protocols_equal_after_relocation(
        protocol, json.loads(json.dumps(_json_ready(expected)))
    ):
        raise ValueError("P3 inference recovery protocol changed")
    if (
        protocol["protocol_id"] != payload["protocol_id"]
        or sha256_file(directory / "recovery_protocol.json")
        != payload["protocol_sha256"]
        or sha256_file(relocated_path(protocol["failure"]["log_path"]))
        != payload["failed_log_sha256"]
    ):
        raise ValueError("P3 inference recovery provenance changed")
    return payload


def _append_recovery_notice(
    amendment: dict[str, Any], output_directory: Path, metrics: dict[str, Any]
) -> None:
    path = REPOSITORY_ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- p3-inference-recovery-{amendment['amendment_id']} -->"
    if marker in text:
        return
    lines = [
        "",
        marker,
        "## P3 inference-routing recovery",
        "",
        f"- Amendment: `{amendment['amendment_id']}`",
        f"- Result: `{output_directory.relative_to(REPOSITORY_ROOT)}`",
        "- Repair: explicitly route the registered `RANDOM_SURGERY` arm in primary and readback inference.",
        "- New or regenerated futures: **0**.",
        f"- Pilot eligibility: **{metrics['pilot_eligibility']}**.",
        f"- Full registered gate: **{metrics['registered_all_four_cells_pass']}**.",
        "- Status: checksum-sealed and stopped before confirmation.",
        "",
    ]
    path.write_text(text + "\n".join(lines), encoding="utf-8")


def _write_result_audit(
    amendment: dict[str, Any], result: Path, audit_directory: Path
) -> None:
    if audit_directory.exists():
        raise FileExistsError(f"refusing to overwrite {audit_directory}")
    verify_checksums(result)
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    replay = json.loads((result / "replay_audit.json").read_text(encoding="utf-8"))
    readback = json.loads((result / "readback_audit.json").read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "original_registration_id": amendment["original_registration_id"],
        "prospective_lifecycle_amendment_id": amendment[
            "lifecycle_amendment_id"
        ],
        "inference_recovery_amendment_id": amendment["amendment_id"],
        "result_directory": str(result.resolve()),
        "result_checksum_manifest_sha256": sha256_file(result / "SHA256SUMS"),
        "phase": manifest["phase"],
        "primary_futures": manifest["primary_futures"],
        "replay_futures": manifest["replay_futures"],
        "recovery_futures": manifest["recovery_futures"],
        "exact_replay": replay["state_edit_endpoint_and_process_digests_exact"],
        "semantic_random_arm": readback["semantic_random_arm"],
        "complete_readback_exact": bool(
            readback["primary_metrics_exact"]
            and readback["matrix_effects_exact"]
            and readback["derived_pilot_eligibility_recomputed"]
        ),
        "checkpoint_hashes_unchanged": manifest["checkpoint_hashes_unchanged"],
        "scientific_contract_changes": [],
        "confirmation_launched": False,
    }
    payload["audit_id"] = _canonical_digest(payload)
    with _atomic_destination(audit_directory) as output:
        (output / "result_audit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    verify_checksums(audit_directory)


def recover(amendment_directory: Path) -> None:
    amendment_directory = amendment_directory.resolve()
    amendment = verify_amendment(amendment_directory)
    protocol = json.loads(
        (amendment_directory / "recovery_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    work = relocated_path(protocol["work_directory"])
    output_directory = relocated_path(
        protocol["intended_result_directory"], require_exists=False
    )
    audit_directory = relocated_path(
        protocol["audit_directory"], require_exists=False
    )
    if output_directory.exists() or audit_directory.exists():
        raise FileExistsError("P3 recovered result or audit already exists")
    checkpoint_before = _require_completed_checkpoints(work)
    if checkpoint_before != protocol["checkpoint_record"]:
        raise ValueError("completed P3 checkpoint record changed before recovery")

    registration = original.verify_registration(
        relocated_path(protocol["original_registration"]["path"])
    )
    spec = original.pilot_spec("p3")
    experiment = original._experiment(spec)
    print("[p3 recovery 1/8] Reconstructing deterministic natural cohort", flush=True)
    with threadpool_limits(limits=1):
        cases = build_cohort(
            experiment, original.PHASE_LABEL["p3"], experiment.confirmation
        )
    if len(cases) != 400:
        raise AssertionError("reconstructed P3 cohort has the wrong state count")

    print(
        "[p3 recovery 2/8] Loading 400 primary and 400 replay checkpoints; generating zero futures",
        flush=True,
    )
    model_path = (
        relocated_path(protocol["original_registration"]["path"])
        / "frozen_full_predictor.npz"
    )
    generated = original.run_phase_batches(
        cases,
        experiment,
        spec,
        model_path,
        registration["registration_id"],
        work / "generate",
        1,
        "generate",
    )
    replayed = original.run_phase_batches(
        cases,
        experiment,
        spec,
        model_path,
        registration["registration_id"],
        work / "replay",
        1,
        "replay",
    )
    replay = original.replay_audit(generated, replayed)
    if not replay["state_edit_endpoint_and_process_digests_exact"]:
        raise ValueError("P3 completed checkpoints fail exact replay")
    arrays = original._outcome_arrays(cases, generated, spec)
    draws = generate_inference_draws(
        spec.matrices,
        original.BOOTSTRAP_REPETITIONS,
        original.RANDOMIZATION_REPETITIONS,
        np.random.default_rng(
            derive_seed(spec.bootstrap_seed, f"{original.PHASE_LABEL['p3']}.bootstrap")
        ),
        np.random.default_rng(
            derive_seed(
                spec.randomization_seed,
                f"{original.PHASE_LABEL['p3']}.randomization",
            )
        ),
    )

    print(
        "[p3 recovery 3/8] Computing frozen inference with RANDOM_SURGERY explicitly routed",
        flush=True,
    )
    metrics, matrix_rows = compute_registered_p3_inference(
        cases, spec, arrays["targets"], arrays["predictions"], draws
    )
    add_derived_pilot_eligibility(metrics, True)
    secondary = original._secondary_descriptives(cases, arrays, spec)
    if _require_completed_checkpoints(work) != checkpoint_before:
        raise ValueError("P3 checkpoints changed while recovery loaded them")

    print("[p3 recovery 4/8] Writing complete machine-readable result", flush=True)
    with _atomic_destination(output_directory) as output:
        np.savez_compressed(output / "branch_arrays.npz", **arrays)
        original._write_branch_table(output / "branches.csv.gz", cases, generated)
        original._write_state_artifacts(output, cases, generated, arrays)
        original._write_selection_artifacts(output, cases, generated, spec)
        original._write_inference_arrays(output / "inference_arrays.npz", draws, metrics)
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
            output, cases, spec, metrics, matrix_rows, replay_exact=True
        )
        (output / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if _require_completed_checkpoints(work) != checkpoint_before:
            raise ValueError("P3 checkpoints changed during result readback")
        recovery_audit = {
            "format": "codex-intervention-p3-inference-recovery-audit-v1",
            "amendment_id": amendment["amendment_id"],
            "original_registration_id": registration["registration_id"],
            "prospective_lifecycle_amendment_id": amendment[
                "lifecycle_amendment_id"
            ],
            "failure_log_sha256": amendment["failed_log_sha256"],
            "primary_checkpoint_states_loaded": 400,
            "replay_checkpoint_states_loaded": 400,
            "new_intervention_futures_generated": 0,
            "intervention_futures_regenerated": 0,
            "generation_checkpoint_digest_unchanged": True,
            "replay_checkpoint_digest_unchanged": True,
            "semantic_random_arm": RANDOM_ARM,
            "semantic_random_arm_explicit_in_primary_and_readback": True,
            "derived_pilot_eligibility_recomputed": True,
            "scientific_contract_changes": [],
            "complete_readback_exact": True,
        }
        (output / "recovery_audit.json").write_text(
            json.dumps(recovery_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "SCIENTIFIC_REPORT.md").write_text(
            original._technical_report("p3", spec, metrics, replay, registration),
            encoding="utf-8",
        )
        (output / "LAY_SUMMARY.md").write_text(
            original._lay_report("p3", metrics, replay), encoding="utf-8"
        )
        (output / "RECOVERY_NOTE.md").write_text(
            "\n".join(
                [
                    "# P3 inference-routing recovery",
                    "",
                    f"Recovery amendment: `{amendment['amendment_id']}`.",
                    f"Prospective lifecycle amendment: `{amendment['lifecycle_amendment_id']}`.",
                    "",
                    "The original P3 execution completed all primary and replay futures, then stopped before inference because its generic caller looked for `RANDOM` instead of the registered `RANDOM_SURGERY` arm. This checksum-sealed recovery explicitly routed `RANDOM_SURGERY`, loaded all completed checkpoints, generated zero futures, verified exact replay and unchanged checkpoint hashes, round-tripped the complete inference, and sealed the result.",
                    "",
                    "No scientific design, data, estimator, threshold, gate, or claim boundary changed. This remains a developmental pilot and stops before confirmation.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        claim_boundary = {
            "supported_at_this_stage": (
                [
                    "pilot eligibility of registered fixed-composition beta surgery for a later untouched confirmation"
                ]
                if metrics["pilot_eligibility"]
                else []
            ),
            "failed_predictions": (
                []
                if metrics["pilot_eligibility"]
                else ["the registered P3 pilot eligibility rule"]
            ),
            "deviations": [
                "source-additive zero-future recovery after explicitly routing the already registered RANDOM_SURGERY control in inference"
            ],
            "unresolved_questions": [
                "whether one selected mechanism passes a separately registered 160-matrix confirmation",
                "whether feedback can maintain altered hereditary behavior",
                "whether maintained organization persists after release",
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
            "format": RESULT_FORMAT,
            "phase": "p3",
            "role": spec.role,
            "registration_id": registration["registration_id"],
            "prospective_lifecycle_amendment_id": amendment[
                "lifecycle_amendment_id"
            ],
            "inference_recovery_amendment_id": amendment["amendment_id"],
            "matrices": spec.matrices,
            "candidates": list(original.CANDIDATES),
            "landmarks": list(original.LANDMARKS),
            "states": len(cases),
            "arms": list(spec.arms),
            "semantic_random_arm": RANDOM_ARM,
            "branches_per_arm_per_state": spec.branches,
            "primary_futures": 51_200,
            "replay_futures": 51_200,
            "recovery_futures": 0,
            "pilot_eligibility": metrics["pilot_eligibility"],
            "full_registered_gate": metrics["registered_all_four_cells_pass"],
            "exact_replay": True,
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
                    "Phase: `p3`",
                    f"Original registration: `{registration['registration_id']}`",
                    f"Prospective lifecycle amendment: `{amendment['lifecycle_amendment_id']}`",
                    f"Inference recovery amendment: `{amendment['amendment_id']}`",
                    f"Pilot eligibility: **{metrics['pilot_eligibility']}**",
                    f"Full registered gate: **{metrics['registered_all_four_cells_pass']}**",
                    "Exact replay: **True**",
                    "Recovery futures: **0**",
                    "Next phase: not launched; mandatory review stop.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print("[p3 recovery 5/8] Sealing recovered result", flush=True)
        write_checksums(output)
    verify_checksums(output_directory)

    print("[p3 recovery 6/8] Writing linked lifecycle audit", flush=True)
    _write_result_audit(amendment, output_directory, audit_directory)
    _append_recovery_notice(amendment, output_directory, metrics)
    original._campaign_status(
        work,
        "p3",
        "sealed_complete",
        "amended_inference_recovery_mandatory_review_stop",
    )
    print("[p3 recovery 7/8] Result and lifecycle audit checksum-verified", flush=True)
    print("[p3 recovery 8/8] STOPPED; confirmation not launched", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3 inference-routing recovery")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--work", type=Path, default=DEFAULT_WORK)
    prepare.add_argument("--failed-log", type=Path, default=DEFAULT_FAILED_LOG)
    prepare.add_argument("--intended-output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    prepare.add_argument("--output", type=Path, default=DEFAULT_AMENDMENT)
    verify = commands.add_parser("verify")
    verify.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    recovery = commands.add_parser("recover")
    recovery.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        prepare_amendment(
            arguments.work,
            arguments.failed_log,
            arguments.intended_output,
            arguments.audit,
            arguments.output,
        )
    elif arguments.command == "verify":
        print(
            json.dumps(
                verify_amendment(arguments.amendment), indent=2, sort_keys=True
            )
        )
    elif arguments.command == "recover":
        recover(arguments.amendment)
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
