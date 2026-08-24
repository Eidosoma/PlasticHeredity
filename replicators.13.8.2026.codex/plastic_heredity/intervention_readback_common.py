"""Shared artifact recovery for completed intervention-pilot checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import intervention_replication as original
from .experiment import _json_ready, _runtime_manifest, build_cohort
from .intervention_metrics import (
    compute_one_shot_inference,
    generate_inference_draws,
)
from .intervention_readback_recovery import (
    _readback_metrics,
    _require_completed_checkpoints,
    add_derived_pilot_eligibility,
)
from .mechanistic import _atomic_destination, verify_checksums, write_checksums
from .seeds import derive_seed


def recover_completed_pilot(
    *,
    phase: str,
    amendment_id: str,
    original_registration_directory: Path,
    work: Path,
    output_directory: Path,
    registered_checkpoint_record: dict[str, Any],
    failure_log_sha256: str,
) -> None:
    """Seal one already completed pilot without creating missing checkpoints."""

    original_registration_directory = original_registration_directory.resolve()
    work = work.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    original_registration = original.verify_registration(
        original_registration_directory
    )
    checkpoint_before = _require_completed_checkpoints(work)
    if checkpoint_before != registered_checkpoint_record:
        raise ValueError("completed checkpoint record changed before recovery")

    spec = original.pilot_spec(phase)
    experiment = original._experiment(spec)
    print(
        f"[{phase} recovery 1/7] Reconstructing the deterministic natural cohort",
        flush=True,
    )
    with threadpool_limits(limits=1):
        cases = build_cohort(
            experiment,
            original.PHASE_LABEL[phase],
            experiment.confirmation,
        )
    if len(cases) != 400:
        raise AssertionError("reconstructed pilot cohort has the wrong state count")

    print(
        f"[{phase} recovery 2/7] Loading 400 primary and 400 replay "
        "checkpoints; generating zero futures",
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
                spec.bootstrap_seed,
                f"{original.PHASE_LABEL[phase]}.bootstrap",
            )
        ),
        np.random.default_rng(
            derive_seed(
                spec.randomization_seed,
                f"{original.PHASE_LABEL[phase]}.randomization",
            )
        ),
    )
    print(
        f"[{phase} recovery 3/7] Recomputing the frozen registered inference",
        flush=True,
    )
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

    print(
        f"[{phase} recovery 4/7] Writing the complete recovered result", flush=True
    )
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
            "format": f"codex-intervention-{phase}-readback-recovery-audit-v1",
            "amendment_id": amendment_id,
            "original_registration_id": original_registration["registration_id"],
            "failure_log_sha256": failure_log_sha256,
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
                phase, spec, metrics, replay, original_registration
            ),
            encoding="utf-8",
        )
        (output / "LAY_SUMMARY.md").write_text(
            original._lay_report(phase, metrics, replay), encoding="utf-8"
        )
        (output / "RECOVERY_NOTE.md").write_text(
            "\n".join(
                [
                    f"# {phase.upper()} readback recovery",
                    "",
                    f"Amendment: `{amendment_id}`.",
                    "",
                    "The original run completed every primary and replay future but stopped because readback omitted the derived `pilot_eligibility` field. This source-additive recovery loaded all 800 completed state checkpoints, generated zero futures, recomputed that field from readback inference and exact replay, verified unchanged checkpoint aggregates, and sealed the result.",
                    "",
                    "No simulator, endpoint, state, intervention, seed, model, branch, inference, margin, gate, or claim boundary changed.",
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
                "source-additive readback recovery under a checksum-sealed derived-field amendment"
            ],
            "unresolved_questions": [
                "whether the effect passes a separately registered untouched confirmation",
                "whether feedback can maintain the altered hereditary behavior",
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
            "format": f"codex-intervention-{phase}-recovered-result-v1",
            "phase": phase,
            "role": spec.role,
            "registration_id": original_registration["registration_id"],
            "amendment_id": amendment_id,
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
                    f"Phase: `{phase}`",
                    f"Original registration: `{original_registration['registration_id']}`",
                    f"Readback amendment: `{amendment_id}`",
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
        print(f"[{phase} recovery 5/7] Sealing recovered result", flush=True)
        write_checksums(output)
    verify_checksums(output_directory)
    original._append_intervention_ledger(
        phase,
        output_directory,
        metrics,
        replay,
        original_registration["registration_id"],
    )
    original._campaign_status(
        work,
        phase,
        "sealed_complete",
        "amended_readback_recovery_review_stop",
    )
    print(
        f"[{phase} recovery 6/7] Recovered result checksum-verified", flush=True
    )
    print(f"[{phase} recovery 7/7] STOPPED; next phase not launched", flush=True)

