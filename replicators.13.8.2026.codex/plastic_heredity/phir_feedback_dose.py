"""Fresh 24-matrix feedback-strength reconciliation for Chapter 5 Phi-r.

The campaign is deliberately separate from the completed Chapter 5 pilot,
the pooled/rolling window bridge, and the locked 48-matrix confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
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
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, GardConfig
from .intervention_core import (
    FrozenFullPredictor,
    MolecularEdit,
    ScoredEdit,
    _records_digest,
    apply_molecular_edit,
    edited_snapshot,
    enumerate_legal_edits,
    score_legal_edits,
)
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .mechanistic_metrics import holm_adjust
from .phir_ch5 import (
    AUTHORIZATION as CH5_CONFIRMATION_AUTHORIZATION,
    DEFAULT_CONFIRMATION as CH5_CONFIRMATION,
    DEFAULT_CONFIRMATION_WORK as CH5_CONFIRMATION_WORK,
    DEFAULT_PILOT as CH5_PILOT,
    DEFAULT_REGISTRATION as CH5_REGISTRATION,
    EXPECTED_MODEL_SHA256,
    BufferState,
    _append_ledger,
    _append_observation,
    _buffer_state,
    _json_ready,
    _restore_buffer,
    _snapshot_after_record,
    verify_result as verify_ch5_result,
)
from .phir_instruments import (
    ATOM_NAMES,
    _canonical_array_digest,
    advance_fission_traced,
    records_equal,
    rng_states_equal,
    score_phi_window,
)
from .phir_window_bridge import (
    DEFAULT_OUTPUT as WINDOW_BRIDGE_OUTPUT,
    _jsonify_table,
    _nan_score,
    _runtime_versions,
    _safe_score,
    _score_fields,
    _score_suffixes,
    score_counts,
    verify_result as verify_window_bridge_result,
)
from .seeds import derive_seed
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    advance_fission,
    generate_beta,
    generate_initial_composition,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCUMENT = "CODEX_CH5_PHIR_FEEDBACK_DOSE_PREREGISTRATION.md"

DEFAULT_VALIDATION = RESULTS / "phir_feedback_dose_validation"
DEFAULT_REGISTRATION = RESULTS / "phir_feedback_dose_registration"
DEFAULT_SMOKE = RESULTS / "phir_feedback_dose_smoke"
DEFAULT_OUTPUT = RESULTS / "phir_feedback_dose24"
DEFAULT_WORK = RESULTS / ".phir_feedback_dose24_work"
DEFAULT_LOG = RESULTS / "phir_feedback_dose24.log"

LABEL = "CODEX_CH5_PHIR_FEEDBACK_DOSE_V1"
PROGRAM_FORMAT = "codex-ch5-phir-feedback-dose-program-v1"
REGISTRATION_FORMAT = "codex-ch5-phir-feedback-dose-registration-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-feedback-dose-checkpoint-v1"
RESULT_FORMAT = "codex-ch5-phir-feedback-dose-result-v1"
STATUS_FORMAT = "codex-ch5-phir-feedback-dose-status-v1"
SERVICE_NAME = "codex-phir-feedback-dose24-20260818"

MATRICES = 24
REPLICATES = 2
NATURAL_GENERATIONS = 60
CONTROL_HORIZON = 60
POOLED30_START = 31
ROLLING_WINDOW = 512
BOOTSTRAP_REPETITIONS = 4096
RANDOMIZATION_REPETITIONS = 4096
MINIMUM_FREE_DISK_BYTES = 1_500_000_000
DOSES = (0.25, 0.50, 0.75, 1.00)
PREPROCESSINGS = ("clr", "raw_count")


def _dose_tag(value: float) -> str:
    return f"{int(round(100.0 * value)):02d}"


DIRECTED_ARMS = tuple(
    f"{direction}_{_dose_tag(dose)}"
    for direction in ("STABILIZE", "DESTABILIZE")
    for dose in DOSES
)
ARMS = ("NOOP", "RANDOM", "NEUTRAL", *DIRECTED_ARMS)

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_feedback_dose.py",
    "tests/test_phir_feedback_dose.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_window_bridge.py",
    "plastic_heredity/phir_ch5.py",
    "plastic_heredity/config.py",
    "plastic_heredity/features.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/mechanistic.py",
    "plastic_heredity/mechanistic_metrics.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/simulator.py",
)


def _seed_value(name: str) -> str:
    return hashlib.sha256(f"{LABEL}::{name}".encode("utf-8")).hexdigest()


SEED_DOMAINS = {
    name: _seed_value(name)
    for name in (
        "matrix",
        "initial",
        "main_path",
        "future",
        "random_action",
        "bootstrap",
        "randomization",
        "validation",
        "smoke",
    )
}


@dataclass(frozen=True)
class RunSpec:
    label: str
    matrices: int
    replicates: int
    natural_generations: int
    control_horizon: int
    pooled30_start: int
    rolling_window: int
    bootstrap_repetitions: int
    randomization_repetitions: int


@dataclass(frozen=True)
class NaturalLaunch:
    candidate: str
    replicate: int
    snapshot: Snapshot
    buffer: BufferState
    record_digest: str
    path_attempt: int


@dataclass(frozen=True)
class DoseChoice:
    arm: str
    edit: MolecularEdit | None
    requested_alpha: float
    achieved_alpha: float
    noop_probability: float
    neutral_probability: float
    minimum_probability: float
    maximum_probability: float
    target_probability: float
    selected_probability: float


@dataclass(frozen=True)
class DoseBatch:
    matrix_id: int
    beta: NDArray[np.float64]
    initial_composition: NDArray[np.int16]
    lineage_rows: tuple[dict[str, Any], ...]
    rolling_rows: tuple[dict[str, Any], ...]
    selected_edit_rows: tuple[dict[str, Any], ...]
    scientific_digest: str


def scientific_spec() -> RunSpec:
    return RunSpec(
        label="feedback_dose24",
        matrices=MATRICES,
        replicates=REPLICATES,
        natural_generations=NATURAL_GENERATIONS,
        control_horizon=CONTROL_HORIZON,
        pooled30_start=POOLED30_START,
        rolling_window=ROLLING_WINDOW,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        randomization_repetitions=RANDOMIZATION_REPETITIONS,
    )


def smoke_spec() -> RunSpec:
    return RunSpec(
        label="smoke",
        matrices=1,
        replicates=1,
        natural_generations=8,
        control_horizon=6,
        pooled30_start=4,
        rolling_window=64,
        bootstrap_repetitions=32,
        randomization_repetitions=32,
    )


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol() -> dict[str, Any]:
    pilot = verify_ch5_result(CH5_PILOT)
    bridge = verify_window_bridge_result(WINDOW_BRIDGE_OUTPUT)
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "program": "fresh 24-matrix repeated-feedback strength reconciliation",
        "completed_ch5_pilot_registration_id": pilot["registration_id"],
        "completed_window_bridge_registration_id": bridge["registration_id"],
        "original_confirmation_locked": bool(
            not CH5_CONFIRMATION.exists()
            and not CH5_CONFIRMATION_WORK.exists()
            and not CH5_CONFIRMATION_AUTHORIZATION.exists()
        ),
        "spec": asdict(scientific_spec()),
        "candidates": list(CANDIDATES),
        "doses": list(DOSES),
        "arms": list(ARMS),
        "selector": {
            "enumeration": "every legal mass-preserving one-molecule substitution",
            "neutral": "legal prediction closest to no-op",
            "targets": "linear interpolation from neutral to each legal extreme",
            "ties": "remove_type then add_type",
            "edit_frequency": "one edit after every completed fission in edited arms",
        },
        "primary": [
            "inherited fraction over controlled fissions 31-60",
            "pooled-final-30 CLR/drop-last revised nine-atom Phi-r",
            "all 16 pooled-final-30 CLR PhiID atoms",
        ],
        "secondary": [
            "rolling-512 CLR and raw-count readings over fissions 31-60",
            "pooled full-dimensional and macro typeset",
            "pooled normalized-full ratio",
            "random-minus-noop and neutral-minus-noop",
        ],
        "frozen_model_sha256": EXPECTED_MODEL_SHA256,
        "seed_domains": SEED_DOMAINS,
        "inference": {
            "unit": "whole catalytic matrix",
            "candidate_pooling": False,
            "replicate_pooling": False,
            "bootstrap": BOOTSTRAP_REPETITIONS,
            "sign_randomization": RANDOMIZATION_REPETITIONS,
            "holm_across_four_cells_within_family": True,
        },
        "classification": {
            "strict": "dose-valid; alpha 0.5 positive; alpha 1 negative; paired half-minus-high positive in every cell",
            "partial": "dose-valid and paired half-minus-high positive, without strict sign crossing",
            "none": "neither registered classification passes",
        },
        "external_context_only": {
            "fable_heredity_effect_approximate_range": [0.11, 0.16],
            "fable_revised_sign": "positive under pooled and rolling readings",
            "imported_as_fitting_target": False,
        },
        "replay": "complete deterministic regeneration of all 24 matrices",
        "raw_molecular_trajectories_persisted": False,
        "cannot_authorize_or_launch_original_confirmation": True,
        "no_48_matrix_continuation": True,
        "claim_boundary": [
            "a dose response does not select a uniquely correct Phi-r",
            "gauge response is not hereditary control or a cause",
            "no consciousness, life, agency, or biological-memory claim",
            "no universal origin-of-life mechanism or Platonic-space portal",
        ],
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(value), sort_keys=True, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=5)
    temporary.replace(path)


def _batch_digest(batch: DoseBatch) -> str:
    blank = DoseBatch(
        matrix_id=batch.matrix_id,
        beta=batch.beta,
        initial_composition=batch.initial_composition,
        lineage_rows=batch.lineage_rows,
        rolling_rows=batch.rolling_rows,
        selected_edit_rows=batch.selected_edit_rows,
        scientific_digest="",
    )
    return _canonical_digest(_json_ready(asdict(blank)))


def _directed_arm(direction: str, dose: float) -> str:
    if direction not in {"STABILIZE", "DESTABILIZE"} or dose not in DOSES:
        raise ValueError("unsupported direction or dose")
    return f"{direction}_{_dose_tag(dose)}"


def _tie_key(item: ScoredEdit) -> tuple[int, int]:
    return item.edit.remove_type, item.edit.add_type


def _nearest(scores: Sequence[ScoredEdit], target: float) -> ScoredEdit:
    return min(
        scores,
        key=lambda item: (
            abs(item.predicted_probability - target),
            item.edit.remove_type,
            item.edit.add_type,
        ),
    )


def select_dose_choices(
    noop_probability: float, scores: Sequence[ScoredEdit]
) -> dict[str, DoseChoice]:
    """Select the frozen neutral-to-extreme ladder from exact legal scores."""

    if not scores:
        raise ValueError("dose selection requires legal edits")
    ordered = tuple(scores)
    neutral = min(
        ordered,
        key=lambda item: (
            abs(item.predicted_probability - noop_probability),
            item.edit.remove_type,
            item.edit.add_type,
        ),
    )
    minimum_value = min(item.predicted_probability for item in ordered)
    maximum_value = max(item.predicted_probability for item in ordered)
    minimum = min(
        (item for item in ordered if item.predicted_probability == minimum_value),
        key=_tie_key,
    )
    maximum = min(
        (item for item in ordered if item.predicted_probability == maximum_value),
        key=_tie_key,
    )
    choices: dict[str, DoseChoice] = {
        "NEUTRAL": DoseChoice(
            arm="NEUTRAL",
            edit=neutral.edit,
            requested_alpha=0.0,
            achieved_alpha=0.0,
            noop_probability=float(noop_probability),
            neutral_probability=float(neutral.predicted_probability),
            minimum_probability=float(minimum.predicted_probability),
            maximum_probability=float(maximum.predicted_probability),
            target_probability=float(neutral.predicted_probability),
            selected_probability=float(neutral.predicted_probability),
        )
    }
    for direction, extreme in (
        ("STABILIZE", minimum),
        ("DESTABILIZE", maximum),
    ):
        denominator = abs(extreme.predicted_probability - neutral.predicted_probability)
        for dose in DOSES:
            arm = _directed_arm(direction, dose)
            target = neutral.predicted_probability + dose * (
                extreme.predicted_probability - neutral.predicted_probability
            )
            selected = extreme if dose == 1.0 else _nearest(ordered, target)
            achieved = (
                abs(selected.predicted_probability - neutral.predicted_probability)
                / denominator
                if denominator > 0.0
                else 0.0
            )
            choices[arm] = DoseChoice(
                arm=arm,
                edit=selected.edit,
                requested_alpha=float(dose),
                achieved_alpha=float(achieved),
                noop_probability=float(noop_probability),
                neutral_probability=float(neutral.predicted_probability),
                minimum_probability=float(minimum.predicted_probability),
                maximum_probability=float(maximum.predicted_probability),
                target_probability=float(target),
                selected_probability=float(selected.predicted_probability),
            )
    return choices


def _matrix_seed(spec: RunSpec, matrix_id: int, purpose: str) -> int:
    domain = SEED_DOMAINS["smoke"] if spec.label == "smoke" else SEED_DOMAINS[purpose]
    return derive_seed(domain, LABEL, spec.label, matrix_id)


def _future_seed(spec: RunSpec, candidate: str, matrix_id: int, replicate: int) -> int:
    domain = SEED_DOMAINS["smoke"] if spec.label == "smoke" else SEED_DOMAINS["future"]
    return derive_seed(domain, LABEL, spec.label, candidate, matrix_id, replicate)


def _action_seed(spec: RunSpec, candidate: str, matrix_id: int, replicate: int) -> int:
    domain = (
        SEED_DOMAINS["smoke"]
        if spec.label == "smoke"
        else SEED_DOMAINS["random_action"]
    )
    return derive_seed(domain, LABEL, spec.label, "uniform_random_edit", candidate, matrix_id, replicate)


def _natural_launch(
    matrix_id: int,
    beta: NDArray,
    initial: NDArray,
    candidate: str,
    replicate: int,
    spec: RunSpec,
) -> NaturalLaunch:
    config = GardConfig()
    domain = SEED_DOMAINS["smoke"] if spec.label == "smoke" else SEED_DOMAINS["main_path"]
    for attempt in range(100):
        rng = np.random.default_rng(
            derive_seed(
                domain,
                LABEL,
                spec.label,
                "main_path",
                candidate,
                matrix_id,
                replicate,
                attempt,
            )
        )
        snapshot = Snapshot(np.asarray(initial, dtype=np.int64).copy(), 0, (), ())
        observations: list[NDArray[np.int64]] = [snapshot.composition.copy()]
        kinds: list[int] = []
        daughters: list[NDArray[np.int64]] = []
        records: list[FissionRecord] = []
        try:
            for _ in range(spec.natural_generations):
                traced = advance_fission_traced(
                    snapshot.composition,
                    beta,
                    config,
                    CANDIDATES[candidate],
                    rng,
                )
                for composition in traced.growth_observations:
                    _append_observation(observations, kinds, composition, 0)
                _append_observation(observations, kinds, traced.record.daughter, 1)
                records.append(traced.record)
                snapshot = _snapshot_after_record(snapshot, traced.record)
                daughters.append(snapshot.composition.copy())
            return NaturalLaunch(
                candidate=candidate,
                replicate=replicate,
                snapshot=snapshot,
                buffer=_buffer_state(observations, kinds, daughters),
                record_digest=_records_digest(records),
                path_attempt=attempt,
            )
        except SimulationError:
            continue
    raise SimulationError(
        f"no complete natural launch c{candidate} m{matrix_id} r{replicate}"
    )


def _select_arm_choice(
    arm: str,
    predictor: FrozenFullPredictor,
    candidate: str,
    snapshot: Snapshot,
    beta: NDArray,
    action_rng: np.random.Generator,
) -> DoseChoice:
    config = GardConfig()
    if arm == "NOOP":
        before = predictor.predict_snapshot(candidate, snapshot, beta, config)
        return DoseChoice(
            arm=arm,
            edit=None,
            requested_alpha=float("nan"),
            achieved_alpha=float("nan"),
            noop_probability=before,
            neutral_probability=float("nan"),
            minimum_probability=float("nan"),
            maximum_probability=float("nan"),
            target_probability=before,
            selected_probability=before,
        )
    if arm == "RANDOM":
        before = predictor.predict_snapshot(candidate, snapshot, beta, config)
        legal = enumerate_legal_edits(snapshot.composition)
        edit = legal[int(action_rng.integers(0, len(legal)))]
        after = predictor.predict_snapshot(
            candidate, edited_snapshot(snapshot, edit), beta, config
        )
        return DoseChoice(
            arm=arm,
            edit=edit,
            requested_alpha=float("nan"),
            achieved_alpha=float("nan"),
            noop_probability=before,
            neutral_probability=float("nan"),
            minimum_probability=float("nan"),
            maximum_probability=float("nan"),
            target_probability=float("nan"),
            selected_probability=after,
        )
    noop, scores = score_legal_edits(predictor, candidate, snapshot, beta, config)
    return select_dose_choices(noop, scores)[arm]


def _append_both(
    rolling_observations: list[NDArray[np.int64]],
    rolling_kinds: list[int],
    controlled_observations: list[NDArray[np.int64]],
    controlled_kinds: list[int],
    composition: NDArray,
    kind: int,
) -> None:
    _append_observation(rolling_observations, rolling_kinds, composition, kind)
    _append_observation(controlled_observations, controlled_kinds, composition, kind)


def _run_arm(
    matrix_id: int,
    launch: NaturalLaunch,
    beta: NDArray,
    predictor: FrozenFullPredictor,
    spec: RunSpec,
    arm: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = GardConfig()
    rolling_observations, rolling_kinds, _ = _restore_buffer(launch.buffer)
    controlled_observations: list[NDArray[np.int64]] = [
        launch.snapshot.composition.copy()
    ]
    controlled_kinds: list[int] = []
    snapshot = launch.snapshot
    records: list[FissionRecord] = []
    inherited: list[int] = []
    rolling_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    boundary30: int | None = None
    rng = np.random.default_rng(
        _future_seed(spec, launch.candidate, matrix_id, launch.replicate)
    )
    action_rng = np.random.default_rng(
        _action_seed(spec, launch.candidate, matrix_id, launch.replicate)
    )
    for step in range(1, spec.control_horizon + 1):
        try:
            traced = advance_fission_traced(
                snapshot.composition,
                beta,
                config,
                CANDIDATES[launch.candidate],
                rng,
            )
        except SimulationError:
            break
        for composition in traced.growth_observations:
            _append_both(
                rolling_observations,
                rolling_kinds,
                controlled_observations,
                controlled_kinds,
                composition,
                0,
            )
        _append_both(
            rolling_observations,
            rolling_kinds,
            controlled_observations,
            controlled_kinds,
            traced.record.daughter,
            1,
        )
        records.append(traced.record)
        inherited.append(int(traced.record.h > config.inheritance_threshold))
        snapshot = _snapshot_after_record(snapshot, traced.record)
        choice = _select_arm_choice(
            arm,
            predictor,
            launch.candidate,
            snapshot,
            beta,
            action_rng,
        )
        if choice.edit is not None:
            snapshot = edited_snapshot(snapshot, choice.edit)
            _append_both(
                rolling_observations,
                rolling_kinds,
                controlled_observations,
                controlled_kinds,
                snapshot.composition,
                2,
            )
        action = {
            "matrix_id": matrix_id,
            "candidate": launch.candidate,
            "replicate": launch.replicate,
            "arm": arm,
            "step": step,
            "edit_applied": int(choice.edit is not None),
            "remove_type": choice.edit.remove_type if choice.edit is not None else -1,
            "add_type": choice.edit.add_type if choice.edit is not None else -1,
            "requested_alpha": choice.requested_alpha,
            "achieved_alpha": choice.achieved_alpha,
            "noop_probability": choice.noop_probability,
            "neutral_probability": choice.neutral_probability,
            "minimum_probability": choice.minimum_probability,
            "maximum_probability": choice.maximum_probability,
            "target_probability": choice.target_probability,
            "selected_probability": choice.selected_probability,
        }
        action["action_digest"] = _canonical_digest(_json_ready(action))
        edit_rows.append(action)
        if step == spec.pooled30_start - 1:
            boundary30 = len(controlled_observations) - 1
        if step >= spec.pooled30_start:
            counts = np.asarray(
                rolling_observations[-spec.rolling_window :], dtype=np.int64
            )
            row: dict[str, Any] = {
                "matrix_id": matrix_id,
                "candidate": launch.candidate,
                "replicate": launch.replicate,
                "arm": arm,
                "step": step,
            }
            for preprocessing in PREPROCESSINGS:
                row.update(
                    _score_fields(
                        preprocessing,
                        _safe_score(
                            counts,
                            preprocessing,
                            include_full_typeset=False,
                        ),
                    )
                )
            rolling_rows.append(row)
    complete = len(records) == spec.control_horizon
    if complete and boundary30 is not None:
        pooled_counts = np.asarray(
            controlled_observations[boundary30:], dtype=np.int64
        )
    else:
        pooled_counts = np.empty((0, config.n_types), dtype=np.int64)
    lineage: dict[str, Any] = {
        "matrix_id": matrix_id,
        "candidate": launch.candidate,
        "replicate": launch.replicate,
        "arm": arm,
        "completed_horizon": int(complete),
        "information_eligible": int(complete),
        "completed_fissions": len(records),
        "extinct": int(not complete),
        "inherited_31_60": float(
            sum(inherited[spec.pooled30_start - 1 :])
            / (spec.control_horizon - spec.pooled30_start + 1)
        ),
        "natural_record_digest": launch.record_digest,
        "controlled_record_digest": _records_digest(records),
        "final_rng_state_digest": _canonical_digest(
            _json_ready(rng.bit_generator.state)
        ),
        "controlled_observation_digest": _canonical_array_digest(
            np.asarray(controlled_observations, dtype=np.int64),
            np.asarray(controlled_kinds, dtype=np.int8),
        ),
        "final_composition": snapshot.composition.astype(int).tolist(),
        "path_attempt": launch.path_attempt,
    }
    for preprocessing in PREPROCESSINGS:
        score = (
            _safe_score(pooled_counts, preprocessing, include_full_typeset=True)
            if len(pooled_counts) >= 3
            else _nan_score(len(pooled_counts))
        )
        lineage.update(_score_fields(f"pooled30_{preprocessing}", score))
    return lineage, rolling_rows, edit_rows


def _run_matrix(args: tuple[int, RunSpec, str]) -> DoseBatch:
    matrix_id, spec, model_path = args
    with threadpool_limits(limits=1):
        config = GardConfig()
        beta = generate_beta(
            config,
            np.random.default_rng(_matrix_seed(spec, matrix_id, "matrix")),
        )
        initial = generate_initial_composition(
            config,
            np.random.default_rng(_matrix_seed(spec, matrix_id, "initial")),
        )
        predictor = FrozenFullPredictor.load(model_path)
        lineage_rows: list[dict[str, Any]] = []
        rolling_rows: list[dict[str, Any]] = []
        edit_rows: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            for replicate in range(spec.replicates):
                launch = _natural_launch(
                    matrix_id, beta, initial, candidate, replicate, spec
                )
                for arm in ARMS:
                    lineage, rolling, edits = _run_arm(
                        matrix_id, launch, beta, predictor, spec, arm
                    )
                    lineage_rows.append(lineage)
                    rolling_rows.extend(rolling)
                    edit_rows.extend(edits)
        provisional = DoseBatch(
            matrix_id=matrix_id,
            beta=np.asarray(beta, dtype=np.float64),
            initial_composition=np.asarray(initial, dtype=np.int16),
            lineage_rows=tuple(lineage_rows),
            rolling_rows=tuple(rolling_rows),
            selected_edit_rows=tuple(edit_rows),
            scientific_digest="",
        )
        return DoseBatch(
            **{**asdict(provisional), "scientific_digest": _batch_digest(provisional)}
        )


def _checkpoint_contract(spec: RunSpec, registration_id: str, stage: str) -> dict[str, Any]:
    value = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
        "stage": stage,
        "spec": asdict(spec),
        "source_hashes": _source_hashes(),
    }
    value["contract_id"] = _canonical_digest(_json_ready(value))
    return value


def _write_status(work: Path, stage: str, completed: int, total: int, **extra: Any) -> None:
    safe = stage.replace("/", "_")
    started = work / f"started_at_{safe}.txt"
    if not started.exists():
        started.parent.mkdir(parents=True, exist_ok=True)
        started.write_text(str(time.time()), encoding="ascii")
    elapsed = max(0.0, time.time() - float(started.read_text(encoding="ascii")))
    rate = completed / elapsed if completed and elapsed else 0.0
    _atomic_json(
        work / "campaign_status.json",
        {
            "format": STATUS_FORMAT,
            "stage": stage,
            "completed": completed,
            "total": total,
            "fraction": completed / total if total else 1.0,
            "elapsed_seconds": elapsed,
            "eta_seconds": (total - completed) / rate if rate else None,
            "pid": os.getpid(),
            **extra,
        },
    )


def _run_checkpointed(
    spec: RunSpec,
    registration_id: str,
    directory: Path,
    stage: str,
    workers: int,
) -> list[DoseBatch]:
    directory.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(spec, registration_id, stage)
    contract_path = directory / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != _json_ready(contract):
            raise ValueError("feedback-dose checkpoint contract changed")
    else:
        _atomic_json(contract_path, contract)
    batches: list[DoseBatch | None] = [None] * spec.matrices
    missing: list[int] = []
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:04d}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                batch = pickle.load(handle)
            if not isinstance(batch, DoseBatch) or batch.matrix_id != matrix_id:
                raise ValueError(f"invalid checkpoint {path}")
            if batch.scientific_digest != _batch_digest(batch):
                raise ValueError(f"checkpoint digest mismatch {path}")
            batches[matrix_id] = batch
        else:
            missing.append(matrix_id)
    completed = spec.matrices - len(missing)
    _write_status(DEFAULT_WORK, stage, completed, spec.matrices, reused=completed)
    arguments = [
        (matrix_id, spec, str(DEFAULT_REGISTRATION / "frozen_full_predictor.npz"))
        for matrix_id in missing
    ]
    executor: ProcessPoolExecutor | None = None
    generated: Iterable[DoseBatch]
    if workers <= 1:
        generated = map(_run_matrix, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        generated = executor.map(_run_matrix, arguments, chunksize=1)
    try:
        for matrix_id, batch in zip(missing, generated, strict=True):
            if batch.matrix_id != matrix_id or batch.scientific_digest != _batch_digest(batch):
                raise AssertionError("feedback-dose worker returned invalid batch")
            batches[matrix_id] = batch
            _atomic_pickle(directory / f"matrix_{matrix_id:04d}.pkl", batch)
            completed += 1
            _write_status(
                DEFAULT_WORK,
                stage,
                completed,
                spec.matrices,
                reused=spec.matrices - len(missing),
            )
            print(f"[{stage}] {completed}/{spec.matrices} matrices", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    if any(batch is None for batch in batches):
        raise AssertionError("feedback-dose checkpoint stage incomplete")
    return [batch for batch in batches if batch is not None]


def _seeded_rng(domain: str, *keys: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(SEED_DOMAINS[domain], LABEL, *keys))


def _summary(
    values: NDArray,
    repetitions: int,
    key: str,
    arrays: dict[str, NDArray],
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if vector.size:
        bootstrap_rng = _seeded_rng("bootstrap", key)
        indices = bootstrap_rng.integers(0, vector.size, size=(repetitions, vector.size))
        bootstrap = vector[indices].mean(axis=1)
        random_rng = _seeded_rng("randomization", key)
        signs = random_rng.choice((-1.0, 1.0), size=(repetitions, vector.size))
        randomized = (signs * vector).mean(axis=1)
        observed = float(vector.mean())
        positive_p = (1 + np.count_nonzero(randomized >= observed)) / (repetitions + 1)
        negative_p = (1 + np.count_nonzero(randomized <= observed)) / (repetitions + 1)
        ci95 = np.quantile(bootstrap, (0.025, 0.975))
    else:
        bootstrap = randomized = np.full(repetitions, np.nan)
        positive_p = negative_p = float("nan")
        ci95 = np.asarray((np.nan, np.nan))
    safe = key.replace("/", "__")
    arrays[f"{safe}__matrix_values"] = vector
    arrays[f"{safe}__bootstrap"] = np.asarray(bootstrap, dtype=np.float64)
    arrays[f"{safe}__sign_randomization"] = np.asarray(randomized, dtype=np.float64)
    return {
        "effect": float(vector.mean()) if vector.size else float("nan"),
        "ci95": [float(ci95[0]), float(ci95[1])],
        "positive_sign_randomization_p": float(positive_p),
        "negative_sign_randomization_p": float(negative_p),
        "matrices": int(vector.size),
        "matrices_positive": int(np.count_nonzero(vector > 0)),
        "matrices_negative": int(np.count_nonzero(vector < 0)),
        "maximum_absolute_matrix_effect": (
            float(np.max(np.abs(vector))) if vector.size else float("nan")
        ),
    }


def _holm(items: Sequence[dict[str, Any]], source: str, destination: str) -> None:
    finite = [item for item in items if np.isfinite(item.get(source, np.nan))]
    if not finite:
        return
    adjusted = holm_adjust([float(item[source]) for item in finite])
    for item, value in zip(finite, adjusted, strict=True):
        item[destination] = float(value)


def _metric_frame(lineages: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    keys = ["matrix_id", "candidate", "replicate", "arm"]
    output = lineages.copy()
    value_columns = [
        f"{preprocessing}_{suffix}"
        for preprocessing in PREPROCESSINGS
        for suffix in _score_suffixes()
        if f"{preprocessing}_{suffix}" in rolling.columns
    ]
    means = rolling.groupby(keys, sort=True)[value_columns].mean().reset_index()
    means = means.rename(columns={name: f"rolling30_{name}" for name in value_columns})
    return output.merge(means, on=keys, how="left", validate="one_to_one")


def _dose_series(
    frame: pd.DataFrame,
    metric: str,
    dose: float,
    candidate: str,
    replicate: int,
) -> pd.Series:
    selected = frame[
        (frame["candidate"].astype(str).str.zfill(2) == candidate)
        & (frame["replicate"] == replicate)
    ]
    table = selected.groupby(["matrix_id", "arm"], sort=True)[metric].mean().unstack("arm")
    high = _directed_arm("STABILIZE", dose)
    low = _directed_arm("DESTABILIZE", dose)
    if not {high, low}.issubset(table.columns):
        return pd.Series(dtype=float)
    return (table[high] - table[low]).dropna()


def _control_series(
    frame: pd.DataFrame,
    metric: str,
    high: str,
    low: str,
    candidate: str,
    replicate: int,
) -> pd.Series:
    selected = frame[
        (frame["candidate"].astype(str).str.zfill(2) == candidate)
        & (frame["replicate"] == replicate)
    ]
    table = selected.groupby(["matrix_id", "arm"], sort=True)[metric].mean().unstack("arm")
    if not {high, low}.issubset(table.columns):
        return pd.Series(dtype=float)
    return (table[high] - table[low]).dropna()


def _pass_positive(item: dict[str, Any]) -> bool:
    return bool(
        item["effect"] > 0
        and item["ci95"][0] > 0
        and item.get("holm_positive_p", 1.0) < 0.05
    )


def _pass_negative(item: dict[str, Any]) -> bool:
    return bool(
        item["effect"] < 0
        and item["ci95"][1] < 0
        and item.get("holm_negative_p", 1.0) < 0.05
    )


def analyze_batches(
    batches: Sequence[DoseBatch], spec: RunSpec
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    lineages = pd.DataFrame([row for batch in batches for row in batch.lineage_rows])
    rolling = pd.DataFrame([row for batch in batches for row in batch.rolling_rows])
    edits = pd.DataFrame([row for batch in batches for row in batch.selected_edit_rows])
    metrics_frame = _metric_frame(lineages, rolling)
    arrays: dict[str, NDArray] = {}
    matrix_rows: list[dict[str, Any]] = []
    dose_cells: list[dict[str, Any]] = []
    metrics = ["inherited_31_60"]
    for window in ("pooled30", "rolling30"):
        for preprocessing in PREPROCESSINGS:
            for suffix in _score_suffixes():
                name = f"{window}_{preprocessing}_{suffix}"
                if (
                    name in metrics_frame.columns
                    and np.isfinite(
                        pd.to_numeric(
                            metrics_frame[name], errors="coerce"
                        ).to_numpy(float)
                    ).any()
                ):
                    metrics.append(name)
    for metric in metrics:
        for dose in DOSES:
            local: list[dict[str, Any]] = []
            for candidate in CANDIDATES:
                for replicate in range(spec.replicates):
                    series = _dose_series(
                        metrics_frame, metric, dose, candidate, replicate
                    )
                    for matrix_id, value in series.items():
                        matrix_rows.append(
                            {
                                "family": "dose_contrast",
                                "metric": metric,
                                "dose": dose,
                                "candidate": candidate,
                                "replicate": replicate,
                                "matrix_id": int(matrix_id),
                                "value": float(value),
                            }
                        )
                    item = _summary(
                        series.to_numpy(float),
                        spec.bootstrap_repetitions,
                        f"{spec.label}/dose/{metric}/a{dose}/c{candidate}/r{replicate}",
                        arrays,
                    )
                    item.update(
                        {
                            "family": "dose_contrast",
                            "metric": metric,
                            "dose": dose,
                            "candidate": candidate,
                            "replicate": replicate,
                        }
                    )
                    local.append(item)
                    dose_cells.append(item)
            _holm(local, "positive_sign_randomization_p", "holm_positive_p")
            _holm(local, "negative_sign_randomization_p", "holm_negative_p")
    slope_cells: list[dict[str, Any]] = []
    for metric in ("inherited_31_60", "pooled30_clr_revised"):
        local = []
        for candidate in CANDIDATES:
            for replicate in range(spec.replicates):
                series_by_dose = {
                    dose: _dose_series(metrics_frame, metric, dose, candidate, replicate)
                    for dose in DOSES
                }
                common = set.intersection(*(set(value.index) for value in series_by_dose.values()))
                slopes: dict[int, float] = {}
                x = np.asarray(DOSES, dtype=np.float64)
                centered = x - x.mean()
                denominator = float(np.sum(centered**2))
                for matrix_id in sorted(common):
                    y = np.asarray(
                        [series_by_dose[dose].loc[matrix_id] for dose in DOSES],
                        dtype=np.float64,
                    )
                    slopes[int(matrix_id)] = float(np.sum(centered * y) / denominator)
                series = pd.Series(slopes, dtype=float)
                for matrix_id, value in series.items():
                    matrix_rows.append(
                        {
                            "family": "dose_slope",
                            "metric": metric,
                            "dose": float("nan"),
                            "candidate": candidate,
                            "replicate": replicate,
                            "matrix_id": int(matrix_id),
                            "value": float(value),
                        }
                    )
                item = _summary(
                    series.to_numpy(float),
                    spec.bootstrap_repetitions,
                    f"{spec.label}/slope/{metric}/c{candidate}/r{replicate}",
                    arrays,
                )
                item.update(
                    {
                        "family": "dose_slope",
                        "metric": metric,
                        "candidate": candidate,
                        "replicate": replicate,
                    }
                )
                local.append(item)
                slope_cells.append(item)
        _holm(local, "positive_sign_randomization_p", "holm_positive_p")
        _holm(local, "negative_sign_randomization_p", "holm_negative_p")
    half_high_cells: list[dict[str, Any]] = []
    local = []
    for candidate in CANDIDATES:
        for replicate in range(spec.replicates):
            half = _dose_series(
                metrics_frame, "pooled30_clr_revised", 0.5, candidate, replicate
            )
            high = _dose_series(
                metrics_frame, "pooled30_clr_revised", 1.0, candidate, replicate
            )
            common = half.index.intersection(high.index)
            series = half.loc[common] - high.loc[common]
            for matrix_id, value in series.items():
                matrix_rows.append(
                    {
                        "family": "half_minus_high",
                        "metric": "pooled30_clr_revised",
                        "dose": float("nan"),
                        "candidate": candidate,
                        "replicate": replicate,
                        "matrix_id": int(matrix_id),
                        "value": float(value),
                    }
                )
            item = _summary(
                series.to_numpy(float),
                spec.bootstrap_repetitions,
                f"{spec.label}/half_minus_high/c{candidate}/r{replicate}",
                arrays,
            )
            item.update(
                {
                    "family": "half_minus_high",
                    "metric": "pooled30_clr_revised",
                    "candidate": candidate,
                    "replicate": replicate,
                }
            )
            local.append(item)
            half_high_cells.append(item)
    _holm(local, "positive_sign_randomization_p", "holm_positive_p")
    _holm(local, "negative_sign_randomization_p", "holm_negative_p")
    control_cells: list[dict[str, Any]] = []
    for contrast, high_arm in (("random_minus_noop", "RANDOM"), ("neutral_minus_noop", "NEUTRAL")):
        for metric in ("inherited_31_60", "pooled30_clr_revised"):
            local = []
            for candidate in CANDIDATES:
                for replicate in range(spec.replicates):
                    series = _control_series(
                        metrics_frame, metric, high_arm, "NOOP", candidate, replicate
                    )
                    item = _summary(
                        series.to_numpy(float),
                        spec.bootstrap_repetitions,
                        f"{spec.label}/control/{contrast}/{metric}/c{candidate}/r{replicate}",
                        arrays,
                    )
                    item.update(
                        {
                            "family": contrast,
                            "metric": metric,
                            "candidate": candidate,
                            "replicate": replicate,
                        }
                    )
                    local.append(item)
                    control_cells.append(item)
            _holm(local, "positive_sign_randomization_p", "holm_positive_p")
            _holm(local, "negative_sign_randomization_p", "holm_negative_p")

    def dose_items(metric: str, dose: float) -> list[dict[str, Any]]:
        return [
            item
            for item in dose_cells
            if item["metric"] == metric and item["dose"] == dose
        ]

    high_heredity = dose_items("inherited_31_60", 1.0)
    heredity_slopes = [
        item for item in slope_cells if item["metric"] == "inherited_31_60"
    ]
    half_phi = dose_items("pooled30_clr_revised", 0.5)
    high_phi = dose_items("pooled30_clr_revised", 1.0)
    dose_validity = bool(
        len(high_heredity) == len(heredity_slopes) == 4
        and all(_pass_positive(item) for item in high_heredity)
        and all(_pass_positive(item) for item in heredity_slopes)
    )
    high_reproduced = bool(
        len(high_phi) == 4 and all(_pass_negative(item) for item in high_phi)
    )
    half_positive = bool(
        len(half_phi) == 4 and all(_pass_positive(item) for item in half_phi)
    )
    moderation = bool(
        len(half_high_cells) == 4
        and all(_pass_positive(item) for item in half_high_cells)
    )
    strict = bool(dose_validity and half_positive and high_reproduced and moderation)
    partial = bool(dose_validity and moderation and not strict)
    gates = {
        "controller_dose_validity": dose_validity,
        "high_dose_codex_response_reproduced": high_reproduced,
        "half_dose_positive": half_positive,
        "half_minus_high_positive": moderation,
        "full_strength_explanation": strict,
        "partial_dose_moderation": partial,
        "no_registered_evidence_for_scale": bool(not strict and not partial),
    }
    completion = [
        {
            "candidate": str(candidate).zfill(2),
            "replicate": int(replicate),
            "arm": arm,
            "lineages": int(len(group)),
            "completed_horizon": int(group["completed_horizon"].sum()),
            "information_eligible": int(group["information_eligible"].sum()),
        }
        for (candidate, replicate, arm), group in lineages.groupby(
            ["candidate", "replicate", "arm"], sort=True
        )
    ]
    selection: list[dict[str, Any]] = []
    for (candidate, replicate, arm), group in edits.groupby(
        ["candidate", "replicate", "arm"], sort=True
    ):
        requested = np.asarray(group["requested_alpha"], dtype=float)
        achieved = np.asarray(group["achieved_alpha"], dtype=float)
        errors = np.abs(requested - achieved)
        errors = errors[np.isfinite(errors)]
        selection.append(
            {
                "candidate": str(candidate).zfill(2),
                "replicate": int(replicate),
                "arm": arm,
                "actions": int(len(group)),
                "edits_applied": int(group["edit_applied"].sum()),
                "mean_requested_alpha": float(group["requested_alpha"].mean()),
                "mean_achieved_alpha": float(group["achieved_alpha"].mean()),
                "median_absolute_alpha_error": float(
                    np.median(errors) if errors.size else float("nan")
                ),
            }
        )
    metrics_payload = {
        "format": "codex-ch5-phir-feedback-dose-metrics-v1",
        "phase": spec.label,
        "matrices": spec.matrices,
        "dose_cells": dose_cells,
        "slope_cells": slope_cells,
        "half_minus_high_cells": half_high_cells,
        "control_cells": control_cells,
        "selection_diagnostics": selection,
        "completion": completion,
        "gates": gates,
        "decision_status": "fresh_24_feedback_dose_complete_awaiting_user_review",
    }
    frames = {
        "lineages": lineages,
        "rolling_windows": rolling,
        "lineage_metrics": metrics_frame,
        "selected_edits": edits,
        "matrix_effects": pd.DataFrame(matrix_rows),
        "selection_diagnostics": pd.DataFrame(selection),
    }
    return metrics_payload, frames, arrays


def _replay_audit(generated: Sequence[DoseBatch], replayed: Sequence[DoseBatch]) -> dict[str, Any]:
    rows = [
        {
            "matrix_id": left.matrix_id,
            "generated_digest": left.scientific_digest,
            "replay_digest": right.scientific_digest,
            "exact": left.scientific_digest == right.scientific_digest,
        }
        for left, right in zip(generated, replayed, strict=True)
    ]
    return {
        "format": "codex-ch5-phir-feedback-dose-replay-v1",
        "matrices": rows,
        "complete_exact_replay": bool(
            len(rows) == MATRICES and all(row["exact"] for row in rows)
        ),
    }


def _effect_text(item: dict[str, Any]) -> str:
    return f"{item['effect']:+.4f} [{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}]"


def _reports(metrics: dict[str, Any], registration_id: str) -> tuple[str, str]:
    rows = []
    for metric in ("inherited_31_60", "pooled30_clr_revised"):
        for item in metrics["dose_cells"]:
            if item["metric"] != metric:
                continue
            rows.append(
                "| "
                + " | ".join(
                    (
                        metric,
                        f"{item['dose']:.2f}",
                        str(item["candidate"]).zfill(2),
                        str(item["replicate"]),
                        _effect_text(item),
                        f"{item.get('holm_positive_p', float('nan')):.4g}",
                        f"{item.get('holm_negative_p', float('nan')):.4g}",
                    )
                )
                + " |"
            )
    slope_rows = [
        f"| {item['metric']} | {str(item['candidate']).zfill(2)} | {item['replicate']} | {_effect_text(item)} | {item.get('holm_positive_p', float('nan')):.4g} |"
        for item in metrics["slope_cells"]
    ]
    moderation_rows = [
        f"| {str(item['candidate']).zfill(2)} | {item['replicate']} | {_effect_text(item)} | {item.get('holm_positive_p', float('nan')):.4g} |"
        for item in metrics["half_minus_high_cells"]
    ]
    gate_lines = [f"- {name}: **{value}**" for name, value in metrics["gates"].items()]
    technical = "\n".join(
        (
            "# Chapter 5 D24 feedback-strength dose reconciliation",
            "",
            f"Registration: `{registration_id}`.",
            "",
            "## Stabilizing minus destabilizing dose effects",
            "",
            "| Metric | Alpha | Candidate | Replicate | Effect [95% matrix CI] | Holm p(+) | Holm p(-) |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Within-matrix dose slopes",
            "",
            "| Metric | Candidate | Replicate | Slope [95% matrix CI] | Holm p(+) |",
            "| --- | --- | ---: | ---: | ---: |",
            *slope_rows,
            "",
            "## Alpha-0.5 minus alpha-1 revised Phi-r moderation",
            "",
            "| Candidate | Replicate | Difference [95% matrix CI] | Holm p(+) |",
            "| --- | ---: | ---: | ---: |",
            *moderation_rows,
            "",
            "## Registered classification",
            "",
            *gate_lines,
            "",
            "## Boundaries",
            "",
            "This fresh 24-matrix campaign tests repeated-feedback strength inside the two Codex GARD contracts. It does not overwrite prior results, authorize the locked confirmation, select a uniquely correct Phi-r, or support consciousness, agency, life, biological memory, or metaphysical claims.",
            "",
        )
    )
    gates = metrics["gates"]
    if gates["full_strength_explanation"]:
        result = "The strict gate passed: the predefined half-strength controller produced a positive revised Phi-r contrast while the full-strength controller produced a negative contrast in every cell."
    elif gates["partial_dose_moderation"]:
        result = "The strict sign-crossing gate failed, but the half-strength reading was consistently above the full-strength reading. Control strength moderated the gauge without fully explaining the external sign disagreement."
    else:
        result = "The registered scale explanation did not pass. The predefined strength ladder did not consistently turn the revised Phi-r response from positive at half strength to negative at full strength."
    lay = "\n".join(
        (
            "# Lay summary — feedback-strength dose reconciliation",
            "",
            "We treated the controller like a volume dial. Every edited lineage still received one one-molecule change after every fission, but the chosen edit ranged from nearly neutral to the strongest prediction-guided edit.",
            "",
            result,
            "",
            "The heredity dial and all information gauges were measured independently. A dose effect would show that the gauge depends on how hard heredity is pushed; it would not make the gauge the cause of heredity or establish a universal information principle.",
            "",
        )
    )
    return technical, lay


def _write_result(
    registration: dict[str, Any],
    spec: RunSpec,
    batches: Sequence[DoseBatch],
    replay: dict[str, Any],
    metrics: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    arrays: dict[str, NDArray],
) -> None:
    technical, lay = _reports(metrics, registration["registration_id"])
    with _atomic_destination(DEFAULT_OUTPUT) as destination:
        _atomic_json(destination / "primary_metrics.json", metrics)
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical, encoding="utf-8")
        (destination / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        _atomic_json(destination / "replay_audit.json", replay)
        _atomic_json(
            destination / "claim_boundaries.json",
            {
                "supported_claims": [],
                "previous_results_modified": False,
                "original_confirmation_remains_locked": True,
                "prohibited_interpretations": protocol()["claim_boundary"],
            },
        )
        row_counts: dict[str, int] = {}
        for name, frame in frames.items():
            table = _jsonify_table(frame)
            table.to_csv(destination / f"{name}.csv.gz", index=False, compression="gzip")
            row_counts[name] = int(len(table))
        np.savez_compressed(destination / "inference_arrays.npz", **arrays)
        np.savez_compressed(
            destination / "matrix_inputs.npz",
            matrix_id=np.asarray([batch.matrix_id for batch in batches], dtype=np.int16),
            beta=np.stack([batch.beta for batch in batches]),
            initial_composition=np.stack([batch.initial_composition for batch in batches]),
            scientific_digest=np.asarray([batch.scientific_digest for batch in batches]),
        )
        readback = {
            "table_row_counts": {
                name: int(len(pd.read_csv(destination / f"{name}.csv.gz")))
                for name in frames
            },
            "expected_table_row_counts": row_counts,
            "replay_exact": replay["complete_exact_replay"],
        }
        readback["complete_readback_exact"] = bool(
            readback["table_row_counts"] == row_counts and readback["replay_exact"]
        )
        if not readback["complete_readback_exact"]:
            raise AssertionError("feedback-dose readback failed")
        _atomic_json(destination / "readback_audit.json", readback)
        _atomic_json(
            destination / "manifest.json",
            {
                "format": RESULT_FORMAT,
                "registration_id": registration["registration_id"],
                "matrices": spec.matrices,
                "candidates": list(CANDIDATES),
                "replicates": spec.replicates,
                "arms": list(ARMS),
                "complete_exact_replay": replay["complete_exact_replay"],
                "complete_readback_exact": True,
                "previous_results_modified": False,
                "original_confirmation_authorized": False,
                "original_confirmation_launched": False,
                "raw_molecular_trajectories_persisted": False,
                "row_counts": row_counts,
                "runtime": _runtime_versions(),
            },
        )
        write_checksums(destination)
    verify_checksums(DEFAULT_OUTPUT)


def validation_checks() -> dict[str, bool]:
    pilot = verify_ch5_result(CH5_PILOT)
    bridge = verify_window_bridge_result(WINDOW_BRIDGE_OUTPUT)
    fixture_scores = tuple(
        ScoredEdit(MolecularEdit(index, index + 1), probability, probability - 0.52)
        for index, probability in enumerate((0.1, 0.3, 0.5, 0.7, 0.9))
    )
    choices = select_dose_choices(0.52, fixture_scores)
    composition = np.asarray([2, 1, 0, 0, 0, 0], dtype=np.int64)
    legal = enumerate_legal_edits(composition)
    edited = apply_molecular_edit(composition, legal[0])
    fixture = np.asarray(
        [[2 + ((time + molecule) % 4) for molecule in range(100)] for time in range(40)],
        dtype=np.int64,
    )
    ours = score_counts(fixture, "clr", include_full_typeset=True)
    sealed = score_phi_window(fixture, include_typeset=True)
    snapshot = Snapshot(np.asarray([1] * 40 + [0] * 60, dtype=np.int64), 7, (0.8,), (0.8,), 12, 50)
    changed = edited_snapshot(snapshot, MolecularEdit(0, 41))
    batch = DoseBatch(
        matrix_id=0,
        beta=np.eye(2),
        initial_composition=np.asarray([1, 0], dtype=np.int16),
        lineage_rows=({"x": float("nan")},),
        rolling_rows=({"x": 1.0},),
        selected_edit_rows=(),
        scientific_digest="",
    )
    transported = pickle.loads(pickle.dumps(batch, protocol=5))
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(9301))
    initial = generate_initial_composition(config, np.random.default_rng(9302))
    trace_rng = np.random.default_rng(9303)
    plain_rng = np.random.default_rng(9303)
    traced = advance_fission_traced(initial, beta, config, CANDIDATES["02"], trace_rng)
    plain = advance_fission(initial, beta, config, CANDIDATES["02"], plain_rng)
    parity_spec = smoke_spec()
    parity_launch = _natural_launch(0, beta, initial, "02", 0, parity_spec)
    parity_predictor = FrozenFullPredictor.load(
        CH5_REGISTRATION / "frozen_full_predictor.npz"
    )
    noop_lineage, _, _ = _run_arm(
        0, parity_launch, beta, parity_predictor, parity_spec, "NOOP"
    )
    parity_rng = np.random.default_rng(_future_seed(parity_spec, "02", 0, 0))
    parity_snapshot = parity_launch.snapshot
    parity_records: list[FissionRecord] = []
    for _ in range(parity_spec.control_horizon):
        record = advance_fission(
            parity_snapshot.composition,
            beta,
            config,
            CANDIDATES["02"],
            parity_rng,
        )
        parity_records.append(record)
        parity_snapshot = _snapshot_after_record(parity_snapshot, record)
    stabilize = [choices[_directed_arm("STABILIZE", dose)].selected_probability for dose in DOSES]
    destabilize = [choices[_directed_arm("DESTABILIZE", dose)].selected_probability for dose in DOSES]
    checks = {
        "01_completed_ch5_pilot_verified": pilot["complete_exact_replay"],
        "02_completed_window_bridge_verified": bridge["complete_exact_replay"],
        "03_original_confirmation_absent": not CH5_CONFIRMATION.exists(),
        "04_original_confirmation_work_absent": not CH5_CONFIRMATION_WORK.exists(),
        "05_original_confirmation_unauthorized": not CH5_CONFIRMATION_AUTHORIZATION.exists(),
        "06_fresh_24_matrices": scientific_spec().matrices == 24,
        "07_two_replicates": scientific_spec().replicates == 2,
        "08_horizon_60": scientific_spec().control_horizon == 60,
        "09_final30_fixed": scientific_spec().pooled30_start == 31,
        "10_doses_fixed": DOSES == (0.25, 0.5, 0.75, 1.0),
        "11_eleven_arms": len(ARMS) == 11,
        "12_neutral_closest": choices["NEUTRAL"].selected_probability == 0.5,
        "13_half_stabilizing_target": choices["STABILIZE_50"].selected_probability == 0.3,
        "14_half_destabilizing_target": choices["DESTABILIZE_50"].selected_probability == 0.7,
        "15_stabilizing_monotone": all(a >= b for a, b in zip(stabilize, stabilize[1:])),
        "16_destabilizing_monotone": all(a <= b for a, b in zip(destabilize, destabilize[1:])),
        "17_alpha1_minimum": choices["STABILIZE_100"].selected_probability == 0.1,
        "18_alpha1_maximum": choices["DESTABILIZE_100"].selected_probability == 0.9,
        "19_legal_enumeration_exact": len(legal) == 2 * 5,
        "20_edit_preserves_mass": int(edited.sum()) == int(composition.sum()),
        "21_edit_nonnegative_integer": np.issubdtype(edited.dtype, np.integer) and np.all(edited >= 0),
        "22_history_unchanged": (
            changed.generation == snapshot.generation
            and changed.inheritance == snapshot.inheritance
            and changed.boundary_h == snapshot.boundary_h
            and changed.previous_growth_steps == snapshot.previous_growth_steps
            and changed.cumulative_growth_steps == snapshot.cumulative_growth_steps
        ),
        "23_future_seed_arm_free": _future_seed(scientific_spec(), "02", 3, 1) == _future_seed(scientific_spec(), "02", 3, 1),
        "24_random_action_seed_separate": _action_seed(scientific_spec(), "02", 3, 1) != _future_seed(scientific_spec(), "02", 3, 1),
        "25_trace_matches_plain_record": records_equal(traced.record, plain),
        "26_trace_matches_plain_rng": rng_states_equal(trace_rng.bit_generator.state, plain_rng.bit_generator.state),
        "27_clr_revised_identity": abs(ours.revised - sealed.revised_phi_r) < 1e-12,
        "28_atom_identity": np.allclose(ours.atoms, sealed.atoms, atol=1e-12, rtol=0),
        "29_typeset_identity": abs(ours.full_typeset - sealed.typeset_phi_r) < 1e-12,
        "30_batch_pickle_stable": _batch_digest(batch) == _batch_digest(transported),
        "31_frozen_model_hash": sha256_file(CH5_REGISTRATION / "frozen_full_predictor.npz") == EXPECTED_MODEL_SHA256,
        "32_matrix_inference_unit": protocol()["inference"]["unit"] == "whole catalytic matrix",
        "33_complete_replay_registered": protocol()["replay"].startswith("complete deterministic"),
        "34_no_raw_trajectory_persistence": not protocol()["raw_molecular_trajectories_persisted"],
        "35_no_48_continuation": protocol()["no_48_matrix_continuation"],
        "36_all_source_files_exist": all((ROOT / name).is_file() for name in SOURCE_FILES),
        "37_noop_records_match_plain": (
            noop_lineage["controlled_record_digest"]
            == _records_digest(parity_records)
        ),
        "38_noop_rng_matches_plain": (
            noop_lineage["final_rng_state_digest"]
            == _canonical_digest(_json_ready(parity_rng.bit_generator.state))
        ),
    }
    return checks


def run_validation() -> dict[str, Any]:
    checks = validation_checks()
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])
    payload = {
        "format": "codex-ch5-phir-feedback-dose-validation-v1",
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": True,
        "source_hashes": _source_hashes(),
        "scientific_matrices_generated": 0,
    }
    with _atomic_destination(DEFAULT_VALIDATION) as destination:
        _atomic_json(destination / "validation.json", payload)
        write_checksums(destination)
    verify_checksums(DEFAULT_VALIDATION)
    print(f"Feedback-dose validation passed: {len(checks)}/{len(checks)}", flush=True)
    return payload


def register_program() -> dict[str, Any]:
    verify_checksums(DEFAULT_VALIDATION)
    validation = json.loads((DEFAULT_VALIDATION / "validation.json").read_text(encoding="utf-8"))
    if validation["source_hashes"] != _source_hashes():
        raise ValueError("source changed after feedback-dose validation")
    for forbidden in (DEFAULT_REGISTRATION, DEFAULT_SMOKE, DEFAULT_OUTPUT, DEFAULT_WORK):
        if forbidden.exists():
            raise FileExistsError(f"pre-scientific artifact already exists: {forbidden}")
    pilot = verify_ch5_result(CH5_PILOT)
    bridge = verify_window_bridge_result(WINDOW_BRIDGE_OUTPUT)
    body: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol": protocol(),
        "protocol_id": protocol()["protocol_id"],
        "source_hashes": _source_hashes(),
        "source_tree_sha256": _canonical_digest(_source_hashes()),
        "seed_registry": SEED_DOMAINS,
        "frozen_model_sha256": EXPECTED_MODEL_SHA256,
        "completed_ch5_pilot_manifest_sha256": sha256_file(CH5_PILOT / "manifest.json"),
        "completed_window_bridge_manifest_sha256": sha256_file(WINDOW_BRIDGE_OUTPUT / "manifest.json"),
        "completed_ch5_pilot_registration_id": pilot["registration_id"],
        "completed_window_bridge_registration_id": bridge["registration_id"],
        "scientific_matrices_at_registration": 0,
        "external_code_data_seeds_models_imported": False,
        "numeric_environment": _runtime_versions(),
    }
    body["registration_id"] = _canonical_digest(_json_ready(body))
    with _atomic_destination(DEFAULT_REGISTRATION) as destination:
        shutil.copy2(ROOT / DOCUMENT, destination / "preregistration.md")
        shutil.copy2(DEFAULT_VALIDATION / "validation.json", destination / "validation.json")
        shutil.copy2(CH5_REGISTRATION / "frozen_full_predictor.npz", destination / "frozen_full_predictor.npz")
        _atomic_json(destination / "protocol.json", protocol())
        _atomic_json(destination / "seed_registry.json", SEED_DOMAINS)
        _atomic_json(destination / "registration.json", body)
        write_checksums(destination)
    verify_checksums(DEFAULT_REGISTRATION)
    _append_ledger(
        f"<!-- phir-feedback-dose-registration-{body['registration_id']} -->",
        (
            "## Chapter 5 D24 feedback-strength reconciliation registered",
            "",
            f"- Registration: `{body['registration_id']}`.",
            "- A fresh 24-matrix dose ladder was sealed after the completed window bridge.",
            "- Prior results remain unchanged and the 48-matrix confirmation remains locked.",
            "- No D24 scientific matrix existed at registration.",
        ),
    )
    print(f"Feedback-dose program registered: {body['registration_id']}", flush=True)
    return body


def verify_registration() -> dict[str, Any]:
    verify_checksums(DEFAULT_REGISTRATION)
    registration = json.loads((DEFAULT_REGISTRATION / "registration.json").read_text(encoding="utf-8"))
    body = dict(registration)
    observed = body.pop("registration_id")
    if registration["format"] != REGISTRATION_FORMAT or _canonical_digest(_json_ready(body)) != observed:
        raise ValueError("feedback-dose registration identity failed")
    if registration["source_hashes"] != _source_hashes():
        raise ValueError("feedback-dose source tree changed")
    if registration["protocol"] != _json_ready(protocol()):
        raise ValueError("feedback-dose protocol changed")
    if sha256_file(DEFAULT_REGISTRATION / "frozen_full_predictor.npz") != EXPECTED_MODEL_SHA256:
        raise ValueError("feedback-dose frozen model changed")
    return registration


def run_smoke() -> dict[str, Any]:
    registration = verify_registration()
    spec = smoke_spec()
    model = str(DEFAULT_REGISTRATION / "frozen_full_predictor.npz")
    first = _run_matrix((0, spec, model))
    second = _run_matrix((0, spec, model))
    metrics, frames, arrays = analyze_batches([first], spec)
    payload = {
        "format": "codex-ch5-phir-feedback-dose-smoke-v1",
        "registration_id": registration["registration_id"],
        "artificial_non_scientific_fixture": True,
        "exact_replay": first.scientific_digest == second.scientific_digest,
        "all_paths_exercised": bool(metrics and frames and arrays),
        "scientific_effect_sizes_or_arm_order_disclosed": False,
        "scientific_matrices_generated": 0,
    }
    if not payload["exact_replay"] or not payload["all_paths_exercised"]:
        raise AssertionError("feedback-dose smoke failed")
    with _atomic_destination(DEFAULT_SMOKE) as destination:
        _atomic_json(destination / "smoke.json", payload)
        write_checksums(destination)
    verify_checksums(DEFAULT_SMOKE)
    print("Feedback-dose non-scientific smoke passed", flush=True)
    return payload


def _prepare_work(registration_id: str, spec: RunSpec) -> None:
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"completed output exists: {DEFAULT_OUTPUT}")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("feedback-dose campaign requires at least 1.5 GB free")
    DEFAULT_WORK.mkdir(parents=True, exist_ok=True)
    expected = {
        "format": "codex-ch5-phir-feedback-dose-work-v1",
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
        "spec": asdict(spec),
    }
    path = DEFAULT_WORK / "campaign_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != _json_ready(expected):
            raise ValueError("feedback-dose work contract changed")
    else:
        _atomic_json(path, expected)


def run_scientific(workers: int = min(os.cpu_count() or 1, 12)) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if CH5_CONFIRMATION.exists() or CH5_CONFIRMATION_WORK.exists() or CH5_CONFIRMATION_AUTHORIZATION.exists():
        raise RuntimeError("original confirmation state changed after D24 registration")
    spec = scientific_spec()
    _prepare_work(registration["registration_id"], spec)
    try:
        generated = _run_checkpointed(
            spec, registration["registration_id"], DEFAULT_WORK / "generated", "generated", workers
        )
        replayed = _run_checkpointed(
            spec, registration["registration_id"], DEFAULT_WORK / "replay", "replay", workers
        )
        replay = _replay_audit(generated, replayed)
        if not replay["complete_exact_replay"]:
            raise AssertionError("feedback-dose complete replay failed")
        _write_status(DEFAULT_WORK, "analysis", 0, 1)
        metrics, frames, arrays = analyze_batches(generated, spec)
        _write_result(registration, spec, generated, replay, metrics, frames, arrays)
        _write_status(
            DEFAULT_WORK,
            "awaiting_user_review",
            1,
            1,
            output=str(DEFAULT_OUTPUT),
        )
    except BaseException as error:
        _write_status(
            DEFAULT_WORK,
            "failed",
            0,
            1,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    _append_ledger(
        f"<!-- phir-feedback-dose-result-{sha256_file(DEFAULT_OUTPUT / 'manifest.json')} -->",
        (
            "## Chapter 5 D24 feedback-strength reconciliation completed",
            "",
            f"- Result: `{DEFAULT_OUTPUT.relative_to(ROOT)}`.",
            "- Complete exact replay and readback passed.",
            f"- Registered gates: `{json.dumps(metrics['gates'], sort_keys=True)}`.",
            "- Prior Chapter 5 results remain unchanged; the confirmation remains locked.",
        ),
    )
    return metrics


def verify_result() -> dict[str, Any]:
    verify_checksums(DEFAULT_OUTPUT)
    manifest = json.loads((DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    registration = verify_registration()
    if manifest["format"] != RESULT_FORMAT:
        raise ValueError("unsupported feedback-dose result")
    if manifest["registration_id"] != registration["registration_id"]:
        raise ValueError("feedback-dose result registration mismatch")
    if not manifest["complete_exact_replay"] or not manifest["complete_readback_exact"]:
        raise ValueError("feedback-dose result integrity failed")
    return manifest


def status_payload() -> dict[str, Any]:
    output: dict[str, Any] = {
        "validation": DEFAULT_VALIDATION.exists(),
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "complete": DEFAULT_OUTPUT.exists(),
        "service": SERVICE_NAME,
        "original_confirmation_authorized": CH5_CONFIRMATION_AUTHORIZATION.exists(),
        "original_confirmation_complete": CH5_CONFIRMATION.exists(),
    }
    status = DEFAULT_WORK / "campaign_status.json"
    if status.exists():
        output["campaign"] = json.loads(status.read_text(encoding="utf-8"))
    return output


def launch_detached(workers: int = min(os.cpu_count() or 1, 12)) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"completed output exists: {DEFAULT_OUTPUT}")
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
        "plastic_heredity.phir_feedback_dose",
        "run",
        "--workers",
        str(workers),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = {
        "format": "codex-ch5-phir-feedback-dose-detached-launch-v1",
        "registration_id": registration["registration_id"],
        "service": SERVICE_NAME,
        "workers": workers,
        "launched_at_unix": time.time(),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    _atomic_json(DEFAULT_WORK / "detached_launch.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("register")
    subparsers.add_parser("smoke")
    run = subparsers.add_parser("run")
    run.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    launch = subparsers.add_parser("launch")
    launch.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    subparsers.add_parser("status")
    subparsers.add_parser("verify")
    arguments = parser.parse_args(argv)
    if arguments.command == "validate":
        run_validation()
    elif arguments.command == "register":
        register_program()
    elif arguments.command == "smoke":
        run_smoke()
    elif arguments.command == "run":
        run_scientific(arguments.workers)
    elif arguments.command == "launch":
        print(json.dumps(launch_detached(arguments.workers), sort_keys=True, indent=2))
    elif arguments.command == "status":
        print(json.dumps(status_payload(), sort_keys=True, indent=2))
    elif arguments.command == "verify":
        print(json.dumps(verify_result(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
