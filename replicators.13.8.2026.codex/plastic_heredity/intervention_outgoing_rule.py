"""Prospectively sealed correction pilot for Fable's outgoing C3 rule.

The original P2 pilot remains immutable.  It tested incoming target support
(`beta @ x`).  This additive module implements the externally disambiguated
frozen Fable rule, outgoing source influence (`x @ beta`, or `beta.T @ x` in
Codex column-vector notation), with new seeds, states, futures, and artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from . import intervention_replication as base
from .config import CANDIDATES
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .intervention_core import (
    FrozenFullPredictor,
    MolecularEdit,
    enumerate_legal_edits,
    simulate_one_shot,
)
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
RESULT_ROOT = REPOSITORY_ROOT / "results_intervention_replication"
ORIGINAL_REGISTRATION = RESULT_ROOT / "registration"
ORIGINAL_P2_RESULT = RESULT_ROOT / "p2_cr3_physical_rule_pilot"
DEFAULT_VALIDATION = RESULT_ROOT / "p2b_outgoing_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "p2b_outgoing_registration"
DEFAULT_OUTPUT = RESULT_ROOT / "p2b_cr3_outgoing_rule_pilot"
DEFAULT_WORK = RESULT_ROOT / ".p2b_outgoing_work"

DOCUMENT = "CODEX_INTERVENTION_OUTGOING_RULE_CORRECTION_PREREGISTRATION.md"
PROGRAM_FORMAT = "codex-intervention-outgoing-rule-correction-v1"
VALIDATION_FORMAT = "codex-intervention-outgoing-rule-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-outgoing-rule-registration-v1"
RESULT_FORMAT = "codex-intervention-outgoing-rule-pilot-result-v1"
CHECKPOINT_FORMAT = "codex-intervention-outgoing-checkpoint-v1"
LABEL = "INTP2B_OUTGOING_V1"
EXPECTED_ORIGINAL_REGISTRATION_ID = (
    "f61e0340dcd8c9ae6b606c8133ca3d8fb1de2e13fe863719aa67b649e8b74531"
)

SOURCE_FILES = (
    DOCUMENT,
    "conftest.py",
    "plastic_heredity/intervention_outgoing_rule.py",
    "tests/test_intervention_outgoing_rule.py",
)


def _seed_value(label: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-outgoing-rule-correction-v1::{label}".encode("utf-8")
    ).hexdigest()


SEED_DOMAINS = {
    name: _seed_value(name)
    for name in (
        "validation",
        "cohort",
        "selection",
        "future",
        "bootstrap",
        "randomization",
        "replay",
    )
}


def phase_spec() -> base.PhaseSpec:
    # phase='p2' deliberately reuses only the established arm/contrast names.
    # Every stochastic domain and the scientific label are new.
    return base.PhaseSpec(
        phase="p2",
        role="CR3 corrected outgoing catalytic-influence rule pilot",
        matrices=base.PILOT_MATRICES,
        branches=base.PILOT_BRANCHES,
        cohort_seed=SEED_DOMAINS["cohort"],
        selection_seed=SEED_DOMAINS["selection"],
        future_seed=SEED_DOMAINS["future"],
        bootstrap_seed=SEED_DOMAINS["bootstrap"],
        randomization_seed=SEED_DOMAINS["randomization"],
    )


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def outgoing_catalytic_influence(
    composition: NDArray, beta: NDArray
) -> NDArray[np.float64]:
    """Return Fable C3 source influence: x @ beta == beta.T @ x.

    Codex stores beta[target, catalyst].  Consequently, output element ``t``
    is the abundance-weighted influence that candidate catalyst ``t`` exerts
    on the types currently present, not the incoming boost received by ``t``.
    """

    values = np.asarray(composition, dtype=np.float64)
    matrix = np.asarray(beta, dtype=np.float64)
    if values.ndim != 1 or matrix.shape != (values.size, values.size):
        raise ValueError("beta and composition dimensions differ")
    mass = float(values.sum())
    if mass <= 0.0:
        raise ValueError("outgoing influence is undefined for an empty assembly")
    fraction = values / mass
    return np.asarray(fraction @ matrix, dtype=np.float64)


def select_outgoing_rule_edits(
    composition: NDArray, beta: NDArray
) -> dict[str, MolecularEdit]:
    """Select the exact frozen C3 extrema over every legal substitution."""

    influence = outgoing_catalytic_influence(composition, beta)
    legal = enumerate_legal_edits(composition)
    if not legal:
        raise ValueError("restored state has no legal molecular substitutions")
    differences = np.asarray(
        [influence[item.add_type] - influence[item.remove_type] for item in legal],
        dtype=np.float64,
    )
    return {
        "RULE_DOWN": legal[int(np.flatnonzero(differences == differences.max())[0])],
        "RULE_UP": legal[int(np.flatnonzero(differences == differences.min())[0])],
    }


def _future_seed(spec: base.PhaseSpec, case: StateCase, branch: int) -> int:
    return derive_seed(
        spec.future_seed,
        f"{LABEL}.future",
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def _selection_seed(spec: base.PhaseSpec, case: StateCase) -> int:
    return derive_seed(
        spec.selection_seed,
        f"{LABEL}.selection.random_arm",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def _phase_worker(
    arguments: tuple[StateCase, Any, base.PhaseSpec, str]
) -> base.PhaseBatch:
    case, experiment, spec, model_path = arguments
    limiter = threadpool_limits(limits=1)
    try:
        predictor = FrozenFullPredictor.load(model_path)
        rules = select_outgoing_rule_edits(case.snapshot.composition, case.beta)
        legal = enumerate_legal_edits(case.snapshot.composition)
        random_rng = np.random.default_rng(_selection_seed(spec, case))
        random_edit = legal[int(random_rng.integers(0, len(legal)))]
        by_name = {
            "RULE_UP": rules["RULE_UP"],
            "RULE_DOWN": rules["RULE_DOWN"],
            "RANDOM": random_edit,
        }
        edits = tuple(
            None if arm == "NOOP" else by_name[arm] for arm in spec.arms
        )
        predictions = base._predict_edit_arms(
            predictor,
            case.candidate,
            case.snapshot,
            case.beta,
            experiment.gard,
            edits,
        )
        arm_outcomes: list[list[Any | None]] = [
            [None] * spec.branches for _ in spec.arms
        ]
        for branch in range(spec.branches):
            seed = _future_seed(spec, case, branch)
            for arm_index, _arm in enumerate(spec.arms):
                arm_outcomes[arm_index][branch] = simulate_one_shot(
                    case.snapshot,
                    case.beta,
                    case.candidate,
                    experiment.gard,
                    base.HORIZON,
                    np.random.default_rng(seed),
                    edits[arm_index],
                )
        outcomes = tuple(
            tuple(item for item in arm if item is not None) for arm in arm_outcomes
        )
        if any(len(arm) != spec.branches for arm in outcomes):
            raise AssertionError("outgoing-rule worker dropped an outcome")
        return base.PhaseBatch(
            state_id=case.state_id,
            state_digest=base._snapshot_digest(case),
            arm_names=spec.arms,
            predictions=predictions,
            selected_edits=edits,
            surgeries=tuple(None for _ in spec.arms),
            scored_edits=tuple(),
            catalytic_support=outgoing_catalytic_influence(
                case.snapshot.composition, case.beta
            ),
            outcomes=outcomes,
        )
    finally:
        limiter.restore_original_limits()


def _checkpoint_contract(
    cases: list[StateCase],
    spec: base.PhaseSpec,
    registration_id: str,
    stage: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "scientific_label": LABEL,
        "phase": "p2b_outgoing",
        "role": spec.role,
        "stage": stage,
        "matrices": spec.matrices,
        "branches": spec.branches,
        "horizon": base.HORIZON,
        "arms": list(spec.arms),
        "case_ids": [case.state_id for case in cases],
        "case_digests": [base._snapshot_digest(case) for case in cases],
        "future_seed": spec.future_seed,
        "future_seed_includes_arm": False,
        "selection_seed": spec.selection_seed,
        "rule_expression": "x @ beta == beta.T @ x",
        "source_hashes": _source_hashes(),
    }
    value["contract_id"] = _canonical_digest(value)
    return value


def run_phase_batches(
    cases: list[StateCase],
    experiment: Any,
    spec: base.PhaseSpec,
    model_path: Path,
    registration_id: str,
    checkpoint_directory: Path,
    workers: int,
    stage: str,
) -> list[base.PhaseBatch]:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(cases, spec, registration_id, stage)
    contract_path = checkpoint_directory / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(_json_ready(contract))
        ):
            raise ValueError(f"checkpoint contract changed: {checkpoint_directory}")
    else:
        base._atomic_json(contract_path, contract)

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
                raise ValueError(f"invalid checkpoint {path}")
            batches[index] = batch
        else:
            missing.append(index)

    def status(state: str) -> None:
        complete = sum(batch is not None for batch in batches)
        base._atomic_json(
            checkpoint_directory / "status.json",
            {
                "format": CHECKPOINT_FORMAT,
                "phase": "p2b_outgoing",
                "stage": stage,
                "state": state,
                "states_complete": complete,
                "states_total": len(cases),
                "percent_complete": 100.0 * complete / len(cases),
                "futures_complete": complete * len(spec.arms) * spec.branches,
                "futures_total": len(cases) * len(spec.arms) * spec.branches,
                "checkpoint_directory": str(checkpoint_directory),
            },
        )

    status("running" if missing else "complete")
    arguments = [
        (cases[index], experiment, spec, str(model_path)) for index in missing
    ]
    if workers <= 1:
        generated = map(_phase_worker, arguments)
        for index, batch in zip(missing, generated, strict=True):
            batches[index] = batch
            base._atomic_pickle(
                checkpoint_directory / f"state_{index:04d}.pkl", batch
            )
            status("running")
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            generated = executor.map(_phase_worker, arguments, chunksize=1)
            for index, batch in zip(missing, generated, strict=True):
                batches[index] = batch
                base._atomic_pickle(
                    checkpoint_directory / f"state_{index:04d}.pkl", batch
                )
                status("running")
    status("complete")
    if any(batch is None for batch in batches):
        raise AssertionError("checkpointed outgoing phase has missing states")
    return [batch for batch in batches if batch is not None]


def add_derived_pilot_eligibility(
    metrics: dict[str, Any], replay_exact: bool
) -> dict[str, Any]:
    metrics["pilot_eligibility"] = bool(
        metrics["pilot_eligibility_without_replay"] and replay_exact
    )
    return metrics


def _readback_metrics(
    output: Path,
    cases: list[StateCase],
    spec: base.PhaseSpec,
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
        equivalence_margin=base.EQUIVALENCE_MARGIN,
        random_ratio_limit=base.RANDOM_RATIO_LIMIT,
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
        raise ValueError("outgoing-rule round-trip intervention inference changed")
    return {
        "branch_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "derived_pilot_eligibility_recomputed": True,
        "primary_metrics_exact": metrics_exact,
        "matrix_effects_exact": matrix_effects_exact,
        "no_fitting_or_recalibration": True,
    }


def _protocol() -> dict[str, Any]:
    spec = phase_spec()
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_p2b_scientific_matrix",
        "scientific_label": LABEL,
        "correction_basis": {
            "external_frozen_expression": "x @ beta",
            "codex_equivalent": "beta.T @ x",
            "codex_storage": "beta[target,catalyst]",
            "wrong_for_fable_c3": "beta @ x",
            "received_after_original_p2_was_sealed": True,
        },
        "original_p2": {
            "result": str(ORIGINAL_P2_RESULT.relative_to(REPOSITORY_ROOT)),
            "scientific_result_unchanged": True,
            "classification": "incoming-support negative control; not Fable C3",
            "pilot_eligibility": False,
            "exact_replay": True,
        },
        "disclosed_post_p2_prediction_only_diagnostic": {
            "new_futures_generated": 0,
            "candidate_02_mean_predicted_up_down": 0.08955050900014036,
            "candidate_03_mean_predicted_up_down": 0.10084118732222329,
            "directional_agreement_each_candidate": 0.995,
            "not_an_outcome_or_gate": True,
        },
        "design": {
            "role": spec.role,
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "landmarks": list(base.LANDMARKS),
            "states": 2 * spec.matrices * len(base.LANDMARKS),
            "arms": list(spec.arms),
            "branches_per_arm_per_state": spec.branches,
            "branch_halves": {"A": [0, 15], "B": [16, 31]},
            "horizon_fissions": base.HORIZON,
            "primary_futures": 51_200,
            "replay_futures": 51_200,
            "fresh_matrices_states_and_seed_domains": True,
            "future_seed_includes_arm": False,
            "intervention_future_retries": 0,
        },
        "rule": {
            "normalization": "x = n / sum(n)",
            "score": "x @ beta == beta.T @ x",
            "rule_up": "remove maximum-influence present; add minimum-influence different type",
            "rule_down": "remove minimum-influence present; add maximum-influence different type",
            "selection": "exhaustive legal substitutions with deterministic lexicographic ties",
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": base.BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": base.RANDOMIZATION_REPETITIONS,
            "holm_family": ["c02_A", "c02_B", "c03_A", "c03_B"],
            "equivalence_margin": base.EQUIVALENCE_MARGIN,
            "random_ratio_limit": base.RANDOM_RATIO_LIMIT,
            "full_gates_unchanged": True,
            "pilot_eligibility_gates_unchanged": True,
        },
        "lifecycle": {
            "readback_derives_pilot_eligibility_before_comparison": True,
            "mandatory_stop_after_seal": True,
            "does_not_launch_p3_or_confirmation": True,
        },
        "claim_boundary": {
            "pilot_only": True,
            "cross_clean_room_confirmation": False,
            "prohibited": base._protocol()["claim_boundaries"]["prohibited"],
        },
        "seed_domains": SEED_DOMAINS,
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def validation_checks() -> dict[str, Any]:
    original = base.validation_checks()
    checks = dict(original["checks"])

    def record(name: str, condition: bool, detail: Any = None) -> None:
        if not condition:
            raise AssertionError(f"validation failed: {name}: {detail}")
        checks[name] = {"passed": True, "detail": detail}

    composition = np.asarray([3, 1, 0, 2], dtype=np.int64)
    beta = np.asarray(
        [
            [1.0, 40.0, 2.0, 3.0],
            [7.0, 1.0, 5.0, 2.0],
            [80.0, 4.0, 1.0, 9.0],
            [2.0, 3.0, 60.0, 1.0],
        ],
        dtype=np.float64,
    )
    x = composition / composition.sum()
    outgoing = outgoing_catalytic_influence(composition, beta)
    incoming = beta @ x
    record(
        "27_outgoing_formula_exact",
        np.array_equal(outgoing, x @ beta)
        and np.array_equal(outgoing, beta.T @ x),
        {"expression": "x @ beta == beta.T @ x"},
    )
    record(
        "28_outgoing_is_not_incoming_on_asymmetric_fixture",
        not np.array_equal(outgoing, incoming),
    )
    rules = select_outgoing_rule_edits(composition, beta)
    legal = enumerate_legal_edits(composition)
    differences = np.asarray(
        [outgoing[item.add_type] - outgoing[item.remove_type] for item in legal]
    )
    record(
        "29_outgoing_extrema_exact_and_legal",
        rules["RULE_DOWN"]
        == legal[int(np.flatnonzero(differences == differences.max())[0])]
        and rules["RULE_UP"]
        == legal[int(np.flatnonzero(differences == differences.min())[0])],
    )
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    permuted = outgoing_catalytic_influence(
        composition[permutation], beta[np.ix_(permutation, permutation)]
    )
    record(
        "30_outgoing_permutation_equivariant",
        np.array_equal(permuted, outgoing[permutation]),
    )
    spec = phase_spec()
    dummy = StateCase(
        "outgoing-fixture", "FIX", "02", 7, 20, beta, base._fixture_snapshot()
    )
    future = [_future_seed(spec, dummy, branch) for branch in range(4)]
    record(
        "31_fresh_future_streams_unique_and_arm_free",
        len(set(future)) == len(future),
        {"arm_identity_in_key": False},
    )
    record(
        "32_new_domains_do_not_collide_with_original",
        set(SEED_DOMAINS.values()).isdisjoint(base.SEED_DOMAINS.values())
        and len(set(SEED_DOMAINS.values())) == len(SEED_DOMAINS),
    )
    metrics = {"pilot_eligibility_without_replay": True}
    add_derived_pilot_eligibility(metrics, True)
    record(
        "33_readback_field_is_derived_before_comparison",
        metrics["pilot_eligibility"] is True,
    )
    verify_checksums(ORIGINAL_P2_RESULT)
    p2_manifest = json.loads(
        (ORIGINAL_P2_RESULT / "manifest.json").read_text(encoding="utf-8")
    )
    record(
        "34_original_p2_preserved_as_failed_incoming_control",
        p2_manifest["pilot_eligibility"] is False
        and p2_manifest["exact_replay"] is True,
    )
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "original_required_checks_passed": original["required_checks_passed"],
        "all_checks_passed": all(value["passed"] for value in checks.values()),
        "check_count": len(checks),
        "scientific_cohort_generated": False,
    }


def run_validation(output_directory: Path) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    validation = validation_checks()
    command = [
        str(REPOSITORY_ROOT / ".venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "tests/test_intervention_outgoing_rule.py",
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
            "outgoing-rule pytest validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    with _atomic_destination(output_directory) as output:
        (output / "validation.json").write_text(
            json.dumps(_json_ready(validation), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "pytest_output.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        write_checksums(output)
    verify_checksums(output_directory)
    print(f"Outgoing-rule validation passed: {output_directory}", flush=True)


def _append_registration_notice(registration_id: str) -> None:
    path = REPOSITORY_ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- p2b-outgoing-registered-{registration_id} -->"
    if marker in text:
        return
    lines = [
        "",
        marker,
        "## P2 orientation correction registered",
        "",
        "- The sealed P2 result is unchanged.",
        "- P2 tested incoming target support (`beta @ x`) and is retained as an incoming-support negative control.",
        "- External frozen-code clarification established that Fable C3 uses outgoing source influence (`x @ beta`, equivalently `beta.T @ x`).",
        "- P2 therefore did not test Fable C3 and is not classified as a failed C3 replication.",
        f"- Corrected P2b registration: `{registration_id}`.",
        "- P2b uses entirely fresh matrices, states, and seed domains and stops after its pilot seal.",
        "",
    ]
    path.write_text(text + "\n".join(lines), encoding="utf-8")


def register_program(validation_directory: Path, output_directory: Path) -> None:
    validation_directory = validation_directory.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    verify_checksums(validation_directory)
    validation = json.loads(
        (validation_directory / "validation.json").read_text(encoding="utf-8")
    )
    if not validation["all_checks_passed"] or validation["scientific_cohort_generated"]:
        raise ValueError("outgoing-rule validation is not registration-eligible")
    original = base.verify_registration(ORIGINAL_REGISTRATION)
    if original["registration_id"] != EXPECTED_ORIGINAL_REGISTRATION_ID:
        raise ValueError("unexpected original intervention registration")
    verify_checksums(ORIGINAL_P2_RESULT)
    protocol = _protocol()
    with _atomic_destination(output_directory) as output:
        (output / "outgoing_rule_protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "outgoing_rule_seed_registry.json").write_text(
            json.dumps(
                {
                    "domains": SEED_DOMAINS,
                    "all_values_unique": len(set(SEED_DOMAINS.values()))
                    == len(SEED_DOMAINS),
                    "disjoint_from_original": set(SEED_DOMAINS.values()).isdisjoint(
                        base.SEED_DOMAINS.values()
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.copy2(validation_directory / "validation.json", output / "validation.json")
        shutil.copy2(
            validation_directory / "pytest_output.txt", output / "pytest_output.txt"
        )
        payload: dict[str, Any] = {
            "format": REGISTRATION_FORMAT,
            "status": "sealed_before_any_p2b_scientific_matrix",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(output / "outgoing_rule_protocol.json"),
            "seed_registry_sha256": sha256_file(
                output / "outgoing_rule_seed_registry.json"
            ),
            "source_hashes": _source_hashes(),
            "original_registration_id": original["registration_id"],
            "original_registration_checksum_manifest_sha256": sha256_file(
                ORIGINAL_REGISTRATION / "SHA256SUMS"
            ),
            "original_p2_checksum_manifest_sha256": sha256_file(
                ORIGINAL_P2_RESULT / "SHA256SUMS"
            ),
            "frozen_predictor_sha256": base.EXPECTED_MODEL_SHA256,
            "validation_checksum_manifest_sha256": sha256_file(
                validation_directory / "SHA256SUMS"
            ),
            "p2b_scientific_matrices_generated": False,
        }
        payload["registration_id"] = _canonical_digest(payload)
        (output / "registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    registered = verify_registration(output_directory)
    _append_registration_notice(registered["registration_id"])
    print(
        f"Outgoing-rule correction sealed: {registered['registration_id']}",
        flush=True,
    )


def verify_registration(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    identifier = payload.pop("registration_id")
    if (
        payload.get("format") != REGISTRATION_FORMAT
        or payload.get("status") != "sealed_before_any_p2b_scientific_matrix"
        or _canonical_digest(payload) != identifier
    ):
        raise ValueError("invalid outgoing-rule registration")
    payload["registration_id"] = identifier
    if payload["source_hashes"] != _source_hashes():
        raise ValueError("outgoing-rule source changed after sealing")
    original = base.verify_registration(ORIGINAL_REGISTRATION)
    if original["registration_id"] != payload["original_registration_id"]:
        raise ValueError("original intervention registration changed")
    verify_checksums(ORIGINAL_P2_RESULT)
    protocol = json.loads(
        (directory / "outgoing_rule_protocol.json").read_text(encoding="utf-8")
    )
    if protocol != json.loads(json.dumps(_json_ready(_protocol()))):
        raise ValueError("outgoing-rule protocol implementation diverged")
    if (
        protocol["protocol_id"] != payload["protocol_id"]
        or sha256_file(directory / "outgoing_rule_protocol.json")
        != payload["protocol_sha256"]
    ):
        raise ValueError("outgoing-rule protocol digest changed")
    seeds = json.loads(
        (directory / "outgoing_rule_seed_registry.json").read_text(encoding="utf-8")
    )
    if (
        seeds["domains"] != SEED_DOMAINS
        or not seeds["all_values_unique"]
        or not seeds["disjoint_from_original"]
    ):
        raise ValueError("outgoing-rule seed registry changed")
    validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    if not validation["all_checks_passed"]:
        raise ValueError("outgoing-rule validation no longer passes")
    return payload


def _campaign_status(work: Path, state: str, detail: str) -> None:
    base._atomic_json(
        work / "campaign_status.json",
        {
            "format": CHECKPOINT_FORMAT,
            "phase": "p2b_outgoing",
            "state": state,
            "detail": detail,
            "mandatory_stop_after_seal": True,
        },
    )


def _prepare_campaign(
    work: Path,
    output: Path,
    registration: dict[str, Any],
    spec: base.PhaseSpec,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    work.mkdir(parents=True, exist_ok=True)
    contract: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration["registration_id"],
        "phase": "p2b_outgoing",
        "output": str(output),
        "matrices": spec.matrices,
        "branches": spec.branches,
        "arms": list(spec.arms),
        "horizon": base.HORIZON,
        "rule_expression": "x @ beta == beta.T @ x",
        "source_hashes": _source_hashes(),
    }
    contract["campaign_id"] = _canonical_digest(contract)
    path = work / "campaign_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(_json_ready(contract))
        ):
            raise ValueError("work directory belongs to another campaign")
    else:
        base._atomic_json(path, contract)
    _campaign_status(work, "running", "campaign_initialized")


def _technical_report(
    metrics: dict[str, Any], replay: dict[str, Any], registration_id: str
) -> str:
    rows = []
    for cell in metrics["cells"]:
        effect = cell["contrasts"]["up_minus_down"]
        random_effect = cell["contrasts"]["random_minus_noop"]
        rows.append(
            "| {cell} | {effect:.6f} | [{low:.6f}, {high:.6f}] | "
            "{p:.6g} | {random:.6f} | {full} |".format(
                cell=cell["cell"],
                effect=effect["estimate"],
                low=effect["bootstrap_ci95"][0],
                high=effect["bootstrap_ci95"][1],
                p=cell["up_down_randomization_p_holm"],
                random=random_effect["estimate"],
                full=cell["registered_cell_pass"],
            )
        )
    return "\n".join(
        [
            "# P2b / corrected Fable C3 outgoing-rule pilot",
            "",
            "## Outcome",
            "",
            f"Pilot eligibility: **{metrics['pilot_eligibility']}**.",
            f"Full four-cell gate: **{metrics['registered_all_four_cells_pass']}**.",
            f"Exact replay: **{replay['state_edit_endpoint_and_process_digests_exact']}**.",
            "",
            "P2b uses `x @ beta` (`beta.T @ x`), the externally disambiguated frozen Fable C3 source/outgoing influence. The original sealed P2 used `beta @ x` and remains an incoming-support negative control.",
            "",
            "| Cell | Up−down | 95% matrix-bootstrap CI | Holm p | Random−no-op | Full gate |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "The catalytic matrix was the inference unit. Branch halves and candidates were evaluated separately with shared whole-matrix draws.",
            "",
            "## Design and audit",
            "",
            "- 40 fresh matrices, two candidates, five landmarks, and 400 states.",
            "- 51,200 primary F12 futures and complete deterministic replay.",
            "- Common random streams across arms; arm identity absent from future seeds.",
            f"- Registration: `{registration_id}`.",
            "",
            "## Boundary",
            "",
            "This is a developmental correction pilot, not an untouched confirmation. It cannot establish life, biological memory, autonomy, Phi/PhiID, real chemistry, or a universal origin-of-life mechanism.",
            "",
            "## Mandatory stop",
            "",
            "The result is sealed. P3 and confirmation remain unlaunched pending review.",
            "",
        ]
    )


def _lay_report(metrics: dict[str, Any], replay: dict[str, Any]) -> str:
    verdict = (
        "The corrected rule moved the event in the required direction in every registered cell and is eligible for a later untouched confirmation."
        if metrics["pilot_eligibility"]
        else "The corrected rule did not satisfy the prewritten eligibility condition in every registered cell."
    )
    return "\n".join(
        [
            "# Lay summary",
            "",
            "The earlier P2 experiment ranked molecules by how much catalytic help they received. Fable's frozen rule actually ranks them by how much help they give to the molecules already present. P2b tests that corrected outgoing direction using entirely new simulated assemblies and random futures.",
            "",
            verdict,
            f" Exact replay passed: **{replay['state_edit_endpoint_and_process_digests_exact']}**.",
            "",
            "This remains a pilot in a computer model. Only a separately registered untouched confirmation could establish a cross-clean-room causal replication.",
            "",
        ]
    )


def _append_result_ledger(
    output: Path,
    metrics: dict[str, Any],
    replay: dict[str, Any],
    registration_id: str,
) -> None:
    path = REPOSITORY_ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- sealed-p2b-outgoing-{registration_id} -->"
    if marker in text:
        return
    lines = [
        "",
        marker,
        "## P2b corrected outgoing-rule pilot",
        "",
        f"- Registration: `{registration_id}`",
        f"- Result bundle: `{output.relative_to(REPOSITORY_ROOT)}`",
        f"- Pilot eligibility: **{metrics['pilot_eligibility']}**",
        f"- Full four-cell gate: **{metrics['registered_all_four_cells_pass']}**",
        f"- Exact replay: **{replay['state_edit_endpoint_and_process_digests_exact']}**",
        "- Rule: `x @ beta`, equivalently `beta.T @ x` under Codex storage.",
        "- Boundary: developmental correction pilot, not untouched confirmation.",
        "",
        "| Cell | Up−down | 95% whole-matrix CI | Holm p | Random−no-op |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in metrics["cells"]:
        effect = cell["contrasts"]["up_minus_down"]
        random_effect = cell["contrasts"]["random_minus_noop"]["estimate"]
        lines.append(
            f"| {cell['cell']} | {effect['estimate']:.6f} | "
            f"[{effect['bootstrap_ci95'][0]:.6f}, {effect['bootstrap_ci95'][1]:.6f}] | "
            f"{cell['up_down_randomization_p_holm']:.6g} | {random_effect:.6f} |"
        )
    path.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")


def run_pilot(
    registration_directory: Path,
    output_directory: Path,
    workers: int,
    work_directory: Path,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output_directory = output_directory.resolve()
    work = work_directory.resolve()
    registration = verify_registration(registration_directory)
    spec = phase_spec()
    experiment = base._experiment(spec)
    _prepare_campaign(work, output_directory, registration, spec)

    print(
        "[p2b 1/8] Building 40 fresh matrices and 400 natural restored states",
        flush=True,
    )
    _campaign_status(work, "running", "building_natural_trajectories")
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, LABEL, experiment.confirmation)
    if len(cases) != 400:
        raise AssertionError("fresh outgoing-rule cohort has the wrong state count")

    model_path = ORIGINAL_REGISTRATION / "frozen_full_predictor.npz"
    futures = len(cases) * len(spec.arms) * spec.branches
    print(f"[p2b 2/8] Selecting outgoing-rule arms and shooting {futures:,} futures", flush=True)
    _campaign_status(work, "running", "selecting_and_shooting_futures")
    generated = run_phase_batches(
        cases,
        experiment,
        spec,
        model_path,
        registration["registration_id"],
        work / "generate",
        workers,
        "generate",
    )
    print(f"[p2b 3/8] Replaying all {futures:,} futures", flush=True)
    _campaign_status(work, "running", "exact_replay")
    replayed = run_phase_batches(
        cases,
        experiment,
        spec,
        model_path,
        registration["registration_id"],
        work / "replay",
        workers,
        "replay",
    )
    replay = base.replay_audit(generated, replayed)
    arrays = base._outcome_arrays(cases, generated, spec)
    draws = generate_inference_draws(
        spec.matrices,
        base.BOOTSTRAP_REPETITIONS,
        base.RANDOMIZATION_REPETITIONS,
        np.random.default_rng(
            derive_seed(spec.bootstrap_seed, f"{LABEL}.bootstrap")
        ),
        np.random.default_rng(
            derive_seed(spec.randomization_seed, f"{LABEL}.randomization")
        ),
    )
    print("[p2b 4/8] Computing frozen whole-matrix inference", flush=True)
    _campaign_status(work, "running", "whole_matrix_inference")
    up, down = spec.contrast
    metrics, matrix_rows = compute_one_shot_inference(
        cases,
        spec.arms,
        arrays["targets"],
        arrays["predictions"],
        draws,
        up_arm=up,
        down_arm=down,
        equivalence_margin=base.EQUIVALENCE_MARGIN,
        random_ratio_limit=base.RANDOM_RATIO_LIMIT,
    )
    add_derived_pilot_eligibility(
        metrics, replay["state_edit_endpoint_and_process_digests_exact"]
    )
    secondary = base._secondary_descriptives(cases, arrays, spec)

    print("[p2b 5/8] Writing and readback-checking artifacts", flush=True)
    with _atomic_destination(output_directory) as output:
        np.savez_compressed(output / "branch_arrays.npz", **arrays)
        base._write_branch_table(output / "branches.csv.gz", cases, generated)
        base._write_state_artifacts(output, cases, generated, arrays)
        base._write_selection_artifacts(output, cases, generated, spec)
        np.savez_compressed(
            output / "outgoing_influence.npz",
            state_ids=np.asarray([case.state_id for case in cases]),
            outgoing_influence=np.vstack(
                [batch.catalytic_support for batch in generated]
            ),
        )
        (output / "rule_definition.json").write_text(
            json.dumps(
                {
                    "fable_expression": "x @ beta",
                    "codex_expression": "beta.T @ x",
                    "storage": "beta[target,catalyst]",
                    "legacy_selection_arrays_field": (
                        "selection_arrays.npz::catalytic_support stores outgoing influence"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        base._write_inference_arrays(output / "inference_arrays.npz", draws, metrics)
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
        (output / "SCIENTIFIC_REPORT.md").write_text(
            _technical_report(metrics, replay, registration["registration_id"]),
            encoding="utf-8",
        )
        (output / "LAY_SUMMARY.md").write_text(
            _lay_report(metrics, replay), encoding="utf-8"
        )
        claim_boundary = {
            "supported_at_this_stage": (
                ["pilot eligibility of the corrected outgoing-rule family"]
                if metrics["pilot_eligibility"]
                else []
            ),
            "failed_predictions": (
                []
                if metrics["pilot_eligibility"]
                else ["the corrected outgoing-rule pilot eligibility condition"]
            ),
            "deviations": [],
            "prohibited_interpretations": base._protocol()["claim_boundaries"][
                "prohibited"
            ],
        }
        (output / "claim_boundaries.json").write_text(
            json.dumps(claim_boundary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format": RESULT_FORMAT,
            "phase": "p2b_outgoing",
            "role": spec.role,
            "registration_id": registration["registration_id"],
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "landmarks": list(base.LANDMARKS),
            "states": len(cases),
            "arms": list(spec.arms),
            "branches_per_arm_per_state": spec.branches,
            "primary_futures": futures,
            "replay_futures": futures,
            "rule_expression": "x @ beta == beta.T @ x",
            "pilot_eligibility": metrics["pilot_eligibility"],
            "full_registered_gate": metrics["registered_all_four_cells_pass"],
            "exact_replay": replay[
                "state_edit_endpoint_and_process_digests_exact"
            ],
            "complete_readback_exact": True,
            "no_future_retries": True,
            "no_matrix_replacement": True,
            "no_refitting_or_recalibration": True,
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
                    "Phase: `p2b_outgoing`",
                    f"Registration: `{registration['registration_id']}`",
                    f"Pilot eligibility: **{metrics['pilot_eligibility']}**",
                    f"Full registered gate: **{metrics['registered_all_four_cells_pass']}**",
                    f"Exact replay: **{replay['state_edit_endpoint_and_process_digests_exact']}**",
                    "Next phase: not launched; mandatory review stop.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print("[p2b 6/8] Sealing and checksum-verifying result", flush=True)
        write_checksums(output)
    verify_checksums(output_directory)
    _append_result_ledger(
        output_directory, metrics, replay, registration["registration_id"]
    )
    _campaign_status(work, "sealed", f"result={output_directory}")
    print("[p2b 7/8] Exact readback and result checksums passed", flush=True)
    print("[p2b 8/8] Mandatory stop: P3 and confirmation not launched", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Corrected outgoing C3 rule pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--output", type=Path, default=DEFAULT_VALIDATION)
    register = commands.add_parser("register")
    register.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    register.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    verify = commands.add_parser("verify")
    verify.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run = commands.add_parser("run")
    run.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    run.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    status = commands.add_parser("status")
    status.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        run_validation(arguments.output)
    elif arguments.command == "register":
        register_program(arguments.validation, arguments.output)
    elif arguments.command == "verify":
        print(
            json.dumps(
                verify_registration(arguments.registration), indent=2, sort_keys=True
            )
        )
    elif arguments.command == "run":
        run_pilot(
            arguments.registration,
            arguments.output,
            arguments.workers,
            arguments.work_dir,
        )
    elif arguments.command == "status":
        print(
            json.dumps(base.read_status(arguments.work_dir), indent=2, sort_keys=True)
        )
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
