"""Administrative CR7 extension-readback correction and checkpoint-only seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from . import intervention_cr7_steering as cr7
from .experiment import StateCase, _json_ready
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
DOCUMENT = "CODEX_INTERVENTION_CR7_READBACK_AMENDMENT.md"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr7_readback_amendment.py",
    "tests/test_intervention_cr7_readback_amendment.py",
)
DEFAULT_VALIDATION = RESULT_ROOT / "cr7_readback_amendment_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr7_readback_amendment_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr7_readback_amendment_smoke"
ORIGINAL_VALIDATION = cr7.DEFAULT_VALIDATION
ORIGINAL_REGISTRATION = cr7.DEFAULT_REGISTRATION
ORIGINAL_SMOKE = cr7.DEFAULT_SMOKE
WORK = cr7.DEFAULT_WORK
OUTPUT = cr7.DEFAULT_OUTPUT
FAILED_LOG = RESULT_ROOT / "cr7_closed_loop_steering.log"
ORIGINAL_REGISTRATION_ID = (
    "41cf815a63129f40c04c7fb260f0f90c713adb9743eaae8479a5f6046e826e70"
)
VALIDATION_FORMAT = "codex-intervention-cr7-readback-amendment-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-cr7-readback-amendment-registration-v1"
SMOKE_FORMAT = "codex-intervention-cr7-readback-amendment-smoke-v1"


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def _install_checkpoint_pickle_aliases() -> None:
    """Expose the registered runner's dataclasses under its ``-m`` pickle name."""

    main_module = sys.modules["__main__"]
    setattr(main_module, "SteeringBatch", cr7.SteeringBatch)
    setattr(main_module, "LineageSummary", cr7.LineageSummary)


def _rebuild_matrix_cases(matrix_id: int) -> list[StateCase]:
    """Reconstruct one matrix's two natural launch states by the sealed rules."""

    limiter = threadpool_limits(limits=1)
    try:
        current_experiment = cr7.experiment()
        beta = cr7.generate_beta(
            current_experiment.gard,
            np.random.default_rng(
                cr7.derive_seed(
                    cr7.SEEDS["matrix_generation"],
                    f"{cr7.LABEL}.beta",
                    matrix_id,
                )
            ),
        )
        initial = cr7.generate_initial_composition(
            current_experiment.gard,
            np.random.default_rng(
                cr7.derive_seed(
                    cr7.SEEDS["initial_composition"],
                    f"{cr7.LABEL}.initial",
                    matrix_id,
                )
            ),
        )
        cases: list[StateCase] = []
        for candidate, contract in cr7.CANDIDATES.items():
            lineage = None
            for attempt in range(100):
                rng = np.random.default_rng(
                    cr7.derive_seed(
                        cr7.SEEDS["main_trajectory"],
                        f"{cr7.LABEL}.natural_main_path",
                        candidate,
                        matrix_id,
                        attempt,
                    )
                )
                try:
                    lineage = cr7.simulate_lineage(
                        initial,
                        beta,
                        current_experiment.gard,
                        contract,
                        rng,
                    )
                    break
                except cr7.SimulationError:
                    continue
            if lineage is None:
                raise cr7.SimulationError(
                    f"failed to reconstruct CR7 launch state for candidate "
                    f"{candidate}, matrix {matrix_id}"
                )
            by_generation = {snapshot.generation: snapshot for snapshot in lineage}
            cases.append(
                StateCase(
                    state_id=(
                        f"{cr7.LABEL}-c{candidate}-m{matrix_id:03d}-"
                        f"g{cr7.LANDMARK:03d}"
                    ),
                    cohort=cr7.LABEL,
                    candidate=candidate,
                    matrix_id=matrix_id,
                    landmark=cr7.LANDMARK,
                    beta=beta,
                    snapshot=by_generation[cr7.LANDMARK],
                )
            )
        return cases
    finally:
        limiter.restore_original_limits()


def rebuild_launch_cases(workers: int = min(14, cr7.os.cpu_count() or 1)) -> list[StateCase]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        grouped = [_rebuild_matrix_cases(matrix_id) for matrix_id in range(cr7.MATRICES)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            grouped = list(executor.map(_rebuild_matrix_cases, range(cr7.MATRICES)))
    return [case for group in grouped for case in group]


@lru_cache(maxsize=1)
def _load_completed_checkpoints() -> tuple[
    list[StateCase],
    list[cr7.SteeringBatch],
    list[cr7.SteeringBatch],
    list[cr7.SteeringBatch],
    list[cr7.SteeringBatch],
]:
    registration = cr7.verify_registration(ORIGINAL_REGISTRATION)
    _install_checkpoint_pickle_aliases()
    cases = rebuild_launch_cases()
    if len(cases) != 2 * cr7.MATRICES:
        raise ValueError("CR7 deterministic launch cohort is incomplete")
    primary_generated: list[cr7.SteeringBatch] = []
    primary_replayed: list[cr7.SteeringBatch] = []
    extension_generated: list[cr7.SteeringBatch] = []
    extension_replayed: list[cr7.SteeringBatch] = []
    for case in cases:
        generated = cr7._read_checkpoint(
            cr7._checkpoint_path(WORK / "primary" / "generate", case),
            case,
            registration["registration_id"],
        )
        replayed = cr7._read_checkpoint(
            cr7._checkpoint_path(WORK / "primary" / "replay", case),
            case,
            registration["registration_id"],
        )
        if generated is None or replayed is None:
            raise ValueError(f"missing completed primary checkpoint for {case.state_id}")
        extension = cr7._read_extension_checkpoint(
            cr7._checkpoint_path(WORK / "extension" / "generate", case),
            case,
            generated,
            registration["registration_id"],
        )
        extension_replay = cr7._read_extension_checkpoint(
            cr7._checkpoint_path(WORK / "extension" / "replay", case),
            case,
            generated,
            registration["registration_id"],
        )
        if extension is None or extension_replay is None:
            raise ValueError(f"missing completed extension checkpoint for {case.state_id}")
        primary_generated.append(generated)
        primary_replayed.append(replayed)
        extension_generated.append(extension)
        extension_replayed.append(extension_replay)
    return (
        cases,
        primary_generated,
        primary_replayed,
        extension_generated,
        extension_replayed,
    )


def reporting_extension_batches(
    cases: list[StateCase],
    primary: list[cr7.SteeringBatch],
    extension: list[cr7.SteeringBatch],
) -> list[cr7.SteeringBatch]:
    """Create metadata-only views accepted by the primary table helper.

    The persisted extension identity is verified before this function is
    called. Only the reporting copy's case_digest changes; lineage objects are
    retained by identity and no checkpoint is rewritten.
    """

    if not (len(cases) == len(primary) == len(extension)):
        raise ValueError("CR7 amendment inputs do not align")
    output: list[cr7.SteeringBatch] = []
    for case, parent, batch in zip(cases, primary, extension, strict=True):
        expected_extension_digest = hashlib.sha256(
            (cr7._case_digest(case) + cr7.batch_digest(parent)).encode()
        ).hexdigest()
        if batch.case_digest != expected_extension_digest:
            raise ValueError("extension checkpoint lost its parent-bound identity")
        view = replace(batch, case_digest=cr7._case_digest(case))
        if view.lineages is not batch.lineages:
            raise AssertionError("metadata view copied or changed scientific lineages")
        output.append(view)
    return output


def _checkpoint_hash_summary() -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for name, directory in (
        ("primary_generate", WORK / "primary" / "generate"),
        ("primary_replay", WORK / "primary" / "replay"),
        ("extension_generate", WORK / "extension" / "generate"),
        ("extension_replay", WORK / "extension" / "replay"),
    ):
        files = sorted(directory.glob("*.pkl"))
        stages[name] = {
            "checkpoint_files": len(files),
            "checkpoint_hash_digest": _canonical_digest(
                {path.name: sha256_file(path) for path in files}
            ),
        }
    return stages


def metadata_audit() -> dict[str, Any]:
    (
        cases,
        primary_generated,
        primary_replayed,
        extension_generated,
        extension_replayed,
    ) = _load_completed_checkpoints()
    primary_replay = cr7.replay_audit(primary_generated, primary_replayed)
    extension_replay = cr7.replay_audit(extension_generated, extension_replayed)
    first_case = cases[0]
    first_extension = extension_generated[0]
    original_rejection = False
    try:
        cr7._lineage_and_matrix_tables([first_case], [first_extension])
    except ValueError as error:
        original_rejection = str(error) == "CR7 batch no longer matches its launch state"
    views = reporting_extension_batches(
        cases, primary_generated, extension_generated
    )
    only_case_digest_changed = all(
        view.lineages is source.lineages
        and view.format == source.format
        and view.registration_id == source.registration_id
        and view.mode == source.mode
        and view.state_id == source.state_id
        and view.candidate == source.candidate
        and view.matrix_id == source.matrix_id
        and view.landmark == source.landmark
        and view.case_digest == cr7._case_digest(case)
        and source.case_digest
        == hashlib.sha256(
            (cr7._case_digest(case) + cr7.batch_digest(parent)).encode()
        ).hexdigest()
        for case, parent, source, view in zip(
            cases, primary_generated, extension_generated, views, strict=True
        )
    )
    return {
        "state_batches": len(cases),
        "all_four_checkpoint_sets_complete": all(
            value["checkpoint_files"] == 2 * cr7.MATRICES
            for value in _checkpoint_hash_summary().values()
        ),
        "primary_generate_replay_exact": primary_replay[
            "exact_state_edit_endpoint_process_and_rng"
        ],
        "extension_generate_replay_exact": extension_replay[
            "exact_state_edit_endpoint_process_and_rng"
        ],
        "original_generic_metadata_rejection_reproduced": original_rejection,
        "reporting_view_changes_only_case_digest": only_case_digest_changed,
        "lineage_tuples_retained_by_identity": all(
            view.lineages is source.lineages
            for source, view in zip(extension_generated, views, strict=True)
        ),
        "checkpoint_files_rewritten": 0,
        "effect_sizes_arm_means_and_gate_values_inspected": False,
    }


def validation_checks() -> dict[str, Any]:
    original = cr7.verify_registration(ORIGINAL_REGISTRATION)
    verify_checksums(ORIGINAL_VALIDATION)
    verify_checksums(ORIGINAL_SMOKE)
    audit = metadata_audit()
    log_text = FAILED_LOG.read_text()
    checkpoint_summary = _checkpoint_hash_summary()
    checks = {
        "original_registration_exact": original["registration_id"]
        == ORIGINAL_REGISTRATION_ID,
        "original_validation_and_smoke_checksums_exact": True,
        "original_registered_source_unchanged": original["source_hashes"]
        == cr7.source_hashes(),
        "failure_is_extension_table_case_digest_check": (
            "CR7 batch no longer matches its launch state" in log_text
            and "extension_summary" in log_text
            and "_lineage_and_matrix_tables" in log_text
        ),
        "all_four_checkpoint_sets_have_96_files": all(
            value["checkpoint_files"] == 96 for value in checkpoint_summary.values()
        ),
        "all_checkpoints_load_under_registered_contracts": audit[
            "all_four_checkpoint_sets_complete"
        ],
        "primary_replay_exact": audit["primary_generate_replay_exact"],
        "extension_replay_exact": audit["extension_generate_replay_exact"],
        "original_metadata_rejection_reproduced": audit[
            "original_generic_metadata_rejection_reproduced"
        ],
        "reporting_view_changes_only_case_digest": audit[
            "reporting_view_changes_only_case_digest"
        ]
        and audit["lineage_tuples_retained_by_identity"],
        "no_final_result_exists": not OUTPUT.exists(),
        "no_checkpoint_rewritten": audit["checkpoint_files_rewritten"] == 0,
        "no_effect_sizes_inspected": not audit[
            "effect_sizes_arm_means_and_gate_values_inspected"
        ],
    }
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "check_count": len(checks),
        "all_checks_passed": bool(all(checks.values())),
        "metadata_audit": audit,
        "checkpoint_summary": checkpoint_summary,
        "scientific_lineages_regenerated": 0,
        "scientific_effect_sizes_inspected": False,
    }


def validate(output: Path = DEFAULT_VALIDATION) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    validation = validation_checks()
    if not validation["all_checks_passed"]:
        raise AssertionError(
            {name: value for name, value in validation["checks"].items() if not value}
        )
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CR7 readback-amendment repository validation failed\n"
            + completed.stdout
            + completed.stderr
        )
    with _atomic_destination(output) as destination:
        payload = dict(validation)
        payload["source_hashes"] = source_hashes()
        payload["failed_log_sha256"] = sha256_file(FAILED_LOG)
        (destination / "validation.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"
        )
        (destination / "pytest_output.txt").write_text(
            "$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR7 readback amendment validation sealed: {output}", flush=True)


def _append_ledger(marker: str, lines: list[str]) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    current = path.read_text(encoding="utf-8").rstrip() + "\n"
    if marker in current:
        return
    path.write_text(current + "\n" + marker + "\n" + "\n".join(lines))


def register(
    validation_directory: Path = DEFAULT_VALIDATION,
    output: Path = DEFAULT_REGISTRATION,
) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    verify_checksums(validation_directory)
    validation = json.loads((validation_directory / "validation.json").read_text())
    if not validation["all_checks_passed"]:
        raise ValueError("CR7 readback amendment validation did not pass")
    if validation["source_hashes"] != source_hashes():
        raise ValueError("CR7 readback amendment source changed after validation")
    original = cr7.verify_registration(ORIGINAL_REGISTRATION)
    payload: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "status": "sealed_before_cr7_checkpoint_only_result_assembly",
        "original_registration_id": original["registration_id"],
        "original_registration_checksum_manifest_sha256": sha256_file(
            ORIGINAL_REGISTRATION / "SHA256SUMS"
        ),
        "source_hashes": source_hashes(),
        "validation_checksum_manifest_sha256": sha256_file(
            validation_directory / "SHA256SUMS"
        ),
        "failed_log_sha256": sha256_file(FAILED_LOG),
        "checkpoint_summary": _checkpoint_hash_summary(),
        "scientific_changes": [],
        "administrative_change": (
            "after parent-bound extension identity and exact replay pass, create "
            "a temporary reporting view whose case_digest is the original case "
            "digest required by the generic primary table helper"
        ),
        "scientific_effect_sizes_inspected_before_amendment": False,
        "original_model_states_seeds_actions_outcomes_inference_and_gates_unchanged": True,
        "scientific_lineages_to_regenerate": 0,
    }
    payload["amendment_id"] = _canonical_digest(_json_ready(payload))
    with _atomic_destination(output) as destination:
        (destination / "registration.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"
        )
        (destination / "amendment.md").write_text((ROOT / DOCUMENT).read_text())
        write_checksums(destination)
    verify_registration(output)
    _append_ledger(
        f"<!-- cr7-readback-amendment-{payload['amendment_id']} -->",
        [
            "## CR7 administrative extension-readback amendment sealed",
            "",
            f"- Amendment: `{payload['amendment_id']}`.",
            f"- Original scientific registration remains `{ORIGINAL_REGISTRATION_ID}`.",
            "- Primary, primary-replay, extension, and extension-replay checkpoint sets were each complete at 96/96 and exact replays passed.",
            "- The correction creates a temporary metadata-only reporting view after the stronger parent-bound extension identity has passed.",
            "- No lineage, model, state, seed, action, outcome, inference draw, gate, or numerical tolerance changes; no scientific lineage is rerun.",
            "- No effect size, arm mean, or gate value was inspected before this amendment was sealed.",
            "",
        ],
    )
    print(f"CR7 readback amendment registered: {payload['amendment_id']}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text())
    if payload.get("format") != REGISTRATION_FORMAT:
        raise ValueError("unsupported CR7 readback amendment registration")
    amendment_id = payload.pop("amendment_id")
    if amendment_id != _canonical_digest(_json_ready(payload)):
        raise ValueError("CR7 readback amendment ID changed")
    payload["amendment_id"] = amendment_id
    if payload["source_hashes"] != source_hashes():
        raise ValueError("CR7 readback amendment source changed")
    original = cr7.verify_registration(ORIGINAL_REGISTRATION)
    if original["registration_id"] != payload["original_registration_id"]:
        raise ValueError("CR7 original registration changed")
    return payload


def smoke(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> None:
    registration = verify_registration(registration_directory)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    audit = metadata_audit()
    passed = bool(
        audit["primary_generate_replay_exact"]
        and audit["extension_generate_replay_exact"]
        and audit["original_generic_metadata_rejection_reproduced"]
        and audit["reporting_view_changes_only_case_digest"]
        and audit["lineage_tuples_retained_by_identity"]
    )
    if not passed:
        raise AssertionError("CR7 readback amendment smoke failed")
    with _atomic_destination(output) as destination:
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "format": SMOKE_FORMAT,
                    "amendment_id": registration["amendment_id"],
                    "metadata_and_integrity_only": True,
                    "scientific_effect_sizes_disclosed": False,
                    "scientific_lineages_regenerated": 0,
                    "passed": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR7 readback amendment smoke passed: {output}", flush=True)


def _apply_result_provenance(output: Path, registration: dict[str, Any]) -> None:
    verify_checksums(output)
    provenance_path = output / "READBACK_AMENDMENT.json"
    expected = {
        "format": "codex-intervention-cr7-readback-amendment-result-v1",
        "amendment_id": registration["amendment_id"],
        "original_registration_id": registration["original_registration_id"],
        "administrative_only": True,
        "scientific_changes": [],
        "scientific_lineages_regenerated": 0,
        "corrected_operation": registration["administrative_change"],
        "all_original_scientific_gates_unchanged": True,
    }
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != expected:
            raise ValueError("CR7 result amendment provenance changed")
        verify_checksums(output)
        return
    provenance_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    manifest["readback_amendment_id"] = registration["amendment_id"]
    manifest["original_scientific_registration_id"] = registration[
        "original_registration_id"
    ]
    manifest["administrative_extension_readback_amendment_only"] = True
    manifest["scientific_lineages_regenerated_during_amendment"] = 0
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_checksums(output)
    verify_checksums(output)


def run(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = OUTPUT,
    work: Path = WORK,
) -> None:
    registration = verify_registration(registration_directory)
    verify_checksums(DEFAULT_SMOKE)
    output = output.resolve()
    if work.resolve() != WORK.resolve():
        raise ValueError("CR7 amendment may only use the sealed original work directory")
    if not output.exists():
        (
            cases,
            primary_generated,
            primary_replayed,
            extension_generated,
            extension_replayed,
        ) = _load_completed_checkpoints()
        primary_replay = cr7.replay_audit(primary_generated, primary_replayed)
        extension_replay = cr7.replay_audit(extension_generated, extension_replayed)
        if not primary_replay["exact_state_edit_endpoint_process_and_rng"]:
            raise AssertionError("CR7 primary replay changed during amendment resume")
        if not extension_replay["exact_state_edit_endpoint_process_and_rng"]:
            raise AssertionError("CR7 extension replay changed during amendment resume")
        _, matrix_table, _ = cr7._lineage_and_matrix_tables(
            cases, primary_generated
        )
        noop_exact = all(
            lineage.noop_plain_bitwise_exact
            for batch in primary_generated
            for lineage in batch.lineages
            if lineage.controller == "NOOP"
        )
        draws = cr7.inference_draws()
        metrics, stored = cr7.compute_inference(
            matrix_table,
            draws,
            replay_exact=True,
            noop_plain_exact=noop_exact,
        )
        if not metrics["conditional_extension_authorized"]:
            raise ValueError(
                "completed extension exists but recomputed original primary gate is false"
            )
        reporting_extension = reporting_extension_batches(
            cases, primary_generated, extension_generated
        )
        extension_metrics, extension_lineage, extension_edits = cr7.extension_summary(
            cases, primary_generated, reporting_extension, draws
        )
        cr7._write_result(
            output,
            cr7.verify_registration(ORIGINAL_REGISTRATION),
            cases,
            primary_generated,
            primary_replay,
            metrics,
            stored,
            extension_metrics,
            extension_lineage,
            extension_edits,
            extension_replay,
        )
    _apply_result_provenance(output, registration)
    manifest = json.loads((output / "manifest.json").read_text())
    _append_ledger(
        f"<!-- sealed-cr7-readback-amended-{registration['amendment_id']} -->",
        [
            "## CR7 result sealed under the administrative extension-readback amendment",
            "",
            f"- Amendment: `{registration['amendment_id']}`.",
            f"- Result: `{output.relative_to(ROOT)}`.",
            f"- Complete original CR7 60-fission gate: **{manifest['complete_cr7_60_fission_gate']}**.",
            f"- Conditional continued-active-control extension launched and replayed: **{manifest['conditional_active_extension_launched']}**.",
            "- Original scientific registration, model, states, seeds, actions, outcomes, inference, and gates remained unchanged; zero scientific lineages were rerun.",
            "- CR8 and CR9 were not launched; mandatory review stop observed.",
            "",
        ],
    )
    cr7._write_status(
        WORK,
        "sealed_complete_mandatory_review_stop",
        2 * cr7.MATRICES,
        2 * cr7.MATRICES,
        output=str(output),
        complete_cr7_gate=manifest["complete_cr7_60_fission_gate"],
        extension_launched=manifest["conditional_active_extension_launched"],
        readback_amendment_id=registration["amendment_id"],
    )
    print("CR7 amended checkpoint-only result sealed; STOPPED before CR8/CR9", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").add_argument(
        "--output", type=Path, default=DEFAULT_VALIDATION
    )
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    register_parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    commands.add_parser("verify").add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    smoke_parser.add_argument("--output", type=Path, default=DEFAULT_SMOKE)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run_parser.add_argument("--output", type=Path, default=OUTPUT)
    run_parser.add_argument("--work-dir", type=Path, default=WORK)
    commands.add_parser("status").add_argument("--work-dir", type=Path, default=WORK)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        validate(arguments.output)
    elif arguments.command == "register":
        register(arguments.validation, arguments.output)
    elif arguments.command == "verify":
        print(json.dumps(verify_registration(arguments.registration), indent=2, sort_keys=True))
    elif arguments.command == "smoke":
        smoke(arguments.registration, arguments.output)
    elif arguments.command == "run":
        run(arguments.registration, arguments.output, arguments.work_dir)
    elif arguments.command == "status":
        print(json.dumps(cr7.read_status(arguments.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
