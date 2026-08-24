"""Chapter 5 clean-room Phi-r / plastic-heredity program.

The 24-matrix pilot and 48-matrix confirmation share one sealed protocol but
use disjoint Codex seeds.  Confirmation is additionally guarded by a manual
authorization artifact and can never follow the pilot automatically.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, ExperimentConfig, GardConfig
from .features import history_features
from .intervention_core import (
    FrozenFullPredictor,
    MolecularEdit,
    _records_digest,
    apply_molecular_edit,
    edited_snapshot,
    enumerate_legal_edits,
    score_legal_edits,
)
from .intervention_outgoing_rule import select_outgoing_rule_edits
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .mechanistic_metrics import holm_adjust
from .metrics import centered_spearman, spearman
from .phir_instruments import (
    ATOM_NAMES,
    PhiWindowScore,
    advance_fission_traced,
    records_equal,
    rng_states_equal,
    score_phi_window,
    trailing_run,
)
from .processes import evaluate_process
from .seeds import derive_seed
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    advance_fission,
    generate_beta,
    generate_initial_composition,
    simulate_future_absorbing,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCUMENT = "CODEX_CH5_PHIR_PREREGISTRATION.md"
AMENDMENT_DOCUMENT = "CODEX_CH5_PHIR_PROCEDURAL_AMENDMENT_001.md"
LEDGER = "PHIR_RESULTS_LEDGER.md"
DEFAULT_VALIDATION = RESULTS / "phir_ch5_validation"
DEFAULT_REGISTRATION = RESULTS / "phir_ch5_registration"
DEFAULT_SMOKE = RESULTS / "phir_ch5_smoke"
DEFAULT_PILOT = RESULTS / "phir_ch5_pilot"
DEFAULT_CONFIRMATION = RESULTS / "phir_ch5_confirmation"
DEFAULT_PILOT_WORK = RESULTS / ".phir_ch5_pilot_work"
DEFAULT_CONFIRMATION_WORK = RESULTS / ".phir_ch5_confirmation_work"
AUTHORIZATION = RESULTS / "phir_ch5_confirmation_authorization.json"
FROZEN_MODEL = RESULTS / "scaled5" / "frozen_models.npz"
EXPECTED_MODEL_SHA256 = "9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af"

PROGRAM_FORMAT = "codex-ch5-phir-program-v1"
REGISTRATION_FORMAT = "codex-ch5-phir-registration-v1"
RESULT_FORMAT = "codex-ch5-phir-result-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-checkpoint-v1"
STATUS_FORMAT = "codex-ch5-phir-status-v1"
LABEL = "CODEX_CH5_PHIR_V1"

PILOT_MATRICES = 24
CONFIRMATION_MATRICES = 48
REPLICATES = 2
NATURAL_GENERATIONS = 100
LANDMARKS = (20, 35, 50, 65, 80)
LAUNCH_GENERATION = 60
F12_BRANCHES = 64
F12_HORIZON = 12
BRIDGE_HORIZON = 60
DOSE_HORIZON = 24
DOSE_QUANTILES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
PROBE_CANDIDATES = 64
PROBE_ROLLOUTS = 4
PROBE_HORIZON = 6
PROBE_CONFIRM_HORIZON = 24
MOLECULAR_WINDOW = 512
GENERATIONAL_WINDOW = 20
BOOTSTRAP_REPETITIONS = 4096
RANDOMIZATION_REPETITIONS = 4096
PROBABILITY_EQUIVALENCE_MARGIN = 0.025
STANDARDIZED_PHI_EQUIVALENCE_MARGIN = 0.10
FORESIGHT_CORRELATION_MARGIN = 0.10
FORESIGHT_LOGLOSS_MARGIN = 0.005
MINIMUM_FREE_DISK_BYTES = 3_000_000_000

BRIDGE_ARMS = (
    "MODEL_STABILIZE",
    "MODEL_DESTABILIZE",
    "RULE_STABILIZE",
    "RULE_DESTABILIZE",
    "RANDOM",
    "NOOP",
)
PROBE_ARMS = ("PHI_UP", "PHI_DOWN", "RANDOM", "NOOP")

SOURCE_FILES = (
    DOCUMENT,
    AMENDMENT_DOCUMENT,
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_ch5.py",
    "tests/test_phir_ch5.py",
    "plastic_heredity/config.py",
    "plastic_heredity/features.py",
    "plastic_heredity/metrics.py",
    "plastic_heredity/mechanistic.py",
    "plastic_heredity/mechanistic_metrics.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_outgoing_rule.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/processes.py",
)


def _seed_value(name: str) -> str:
    return hashlib.sha256(f"{LABEL}::{name}".encode("utf-8")).hexdigest()


SEED_DOMAINS = {
    name: _seed_value(name)
    for name in (
        "pilot_matrix",
        "confirmation_matrix",
        "initial",
        "main_path",
        "f12_future",
        "bridge_future",
        "controller_action",
        "dose_future",
        "probe_selection",
        "probe_screen",
        "probe_confirmation",
        "bootstrap",
        "randomization",
        "replay",
        "validation",
        "smoke",
    )
}


def _runtime_versions() -> dict[str, str]:
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "threadpoolctl": importlib.metadata.version("threadpoolctl"),
    }


@dataclass(frozen=True)
class RunSpec:
    label: str
    matrices: int
    replicates: int
    natural_generations: int
    landmarks: tuple[int, ...]
    launch_generation: int
    branches: int
    bridge_horizon: int
    bridge_arms: tuple[str, ...]
    dose_horizon: int
    dose_quantiles: tuple[float, ...]
    probe_candidates: int
    probe_rollouts: int
    probe_horizon: int
    probe_confirm_horizon: int
    probe_arms: tuple[str, ...]
    bootstrap_repetitions: int
    randomization_repetitions: int


@dataclass(frozen=True)
class BufferState:
    observations: NDArray[np.int16]
    transition_kinds: NDArray[np.int8]
    daughters: NDArray[np.int16]


@dataclass(frozen=True)
class NaturalRun:
    candidate: str
    replicate: int
    snapshots: dict[int, Snapshot]
    launch_buffer: BufferState
    natural_rows: tuple[dict[str, Any], ...]
    record_digest: str
    path_attempt: int


@dataclass(frozen=True)
class CampaignBatch:
    matrix_id: int
    beta: NDArray[np.float64]
    initial_composition: NDArray[np.int16]
    natural_rows: tuple[dict[str, Any], ...]
    branch_rows: tuple[dict[str, Any], ...]
    bridge_rows: tuple[dict[str, Any], ...]
    dose_rows: tuple[dict[str, Any], ...]
    probe_rows: tuple[dict[str, Any], ...]
    selected_edit_rows: tuple[dict[str, Any], ...]
    probe_screen_rows: tuple[dict[str, Any], ...]
    no_op_plain_exact: bool
    scientific_digest: str


def scientific_spec(phase: str) -> RunSpec:
    if phase not in {"pilot", "confirmation"}:
        raise ValueError("phase must be pilot or confirmation")
    return RunSpec(
        label=phase,
        matrices=PILOT_MATRICES if phase == "pilot" else CONFIRMATION_MATRICES,
        replicates=REPLICATES,
        natural_generations=NATURAL_GENERATIONS,
        landmarks=LANDMARKS,
        launch_generation=LAUNCH_GENERATION,
        branches=F12_BRANCHES,
        bridge_horizon=BRIDGE_HORIZON,
        bridge_arms=BRIDGE_ARMS,
        dose_horizon=DOSE_HORIZON,
        dose_quantiles=DOSE_QUANTILES,
        probe_candidates=PROBE_CANDIDATES,
        probe_rollouts=PROBE_ROLLOUTS,
        probe_horizon=PROBE_HORIZON,
        probe_confirm_horizon=PROBE_CONFIRM_HORIZON,
        probe_arms=PROBE_ARMS,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        randomization_repetitions=RANDOMIZATION_REPETITIONS,
    )


def smoke_spec() -> RunSpec:
    return RunSpec(
        label="smoke",
        matrices=2,
        replicates=1,
        natural_generations=20,
        landmarks=(20,),
        launch_generation=20,
        branches=2,
        bridge_horizon=2,
        bridge_arms=("RANDOM", "NOOP"),
        dose_horizon=2,
        dose_quantiles=(0.0, 1.0),
        probe_candidates=6,
        probe_rollouts=1,
        probe_horizon=1,
        probe_confirm_horizon=2,
        probe_arms=PROBE_ARMS,
        bootstrap_repetitions=32,
        randomization_repetitions=32,
    )


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "program": "independent Codex Phi-r / plastic-heredity Chapter 5",
        "clean_room": {
            "fable_code_data_seeds_models_states_forbidden": True,
            "public_phirl_commit": "a6d1d0d18c7551302724b7158c6ccdc4d3a33373",
            "public_preprint": "arXiv:2607.28250v1",
            "external_values_are_post_seal_benchmarks": True,
        },
        "instruments": {
            "typeset": "unnormalized multivariate whole-minus-Fiedler-parts",
            "revised": "public-PhiRL nine-atom PhiID sum on macro Fiedler halves",
            "text_artifact": "typeset numerator divided by whole MI; negative control",
            "atoms": list(ATOM_NAMES),
            "pseudocount": 0.5,
            "clr_drop_last": True,
            "active_std_threshold": 1e-8,
            "graph_floor": 1e-6,
            "covariance_ridge": "max(1e-8, 1e-6*trace/d) for multivariate MI",
            "units": "nats",
        },
        "clocks": {
            "molecular_observations": MOLECULAR_WINDOW,
            "generational_observations": GENERATIONAL_WINDOW,
            "primary_includes_fission_and_intervention_transitions": True,
            "growth_only_sensitivity": True,
            "all_fits_past_only": True,
        },
        "pilot": asdict(scientific_spec("pilot")),
        "confirmation": asdict(scientific_spec("confirmation")),
        "manual_confirmation_barrier": {
            "pilot_result_required": True,
            "user_authorization_artifact_required": True,
            "automatic_launch_forbidden": True,
            "pilot_confirmation_pooling_forbidden": True,
        },
        "simulator": {
            "candidates": {name: asdict(contract) for name, contract in CANDIDATES.items()},
            "gard": asdict(GardConfig()),
            "sealed_simulator_unmodified": True,
            "trace_wrapper_bitwise_parity_required": True,
        },
        "frozen_predictor": {
            "path": str(FROZEN_MODEL.relative_to(ROOT)),
            "sha256": EXPECTED_MODEL_SHA256,
            "no_refit_recalibration_or_threshold_change": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "candidate_pooling": False,
            "bootstrap": BOOTSTRAP_REPETITIONS,
            "sign_randomization": RANDOMIZATION_REPETITIONS,
            "holm_within_family": True,
            "probability_equivalence_margin": PROBABILITY_EQUIVALENCE_MARGIN,
            "standardized_phi_equivalence_margin": STANDARDIZED_PHI_EQUIVALENCE_MARGIN,
            "foresight_correlation_margin": FORESIGHT_CORRELATION_MARGIN,
            "foresight_logloss_margin": FORESIGHT_LOGLOSS_MARGIN,
        },
        "seed_domains": SEED_DOMAINS,
        "replay": "complete deterministic replay of every scientific matrix unit",
        "storage": "rolling molecular buffers only; raw molecular traces forbidden",
        "numeric_environment": _runtime_versions(),
        "claim_boundary": [
            "no consciousness, life, agency, or biological-memory claim",
            "no universal origin-of-life mechanism",
            "no Ruliad or Platonic-space portal",
            "no validation of the unavailable private GARD-paper pipeline",
            "gauge response is not hereditary control",
        ],
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
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


def _batch_digest(batch: CampaignBatch) -> str:
    """Return a process- and serialization-stable digest of batch values.

    Raw pickle bytes are not a canonical representation: equivalent object
    graphs can acquire different memo/reference layouts after crossing a
    multiprocessing boundary.  JSON-normalizing the dataclass values makes
    the integrity digest depend on scientific content rather than Python
    object identity.  NaN tokens are deterministic under the stdlib encoder.
    """
    value = CampaignBatch(
        matrix_id=batch.matrix_id,
        beta=batch.beta,
        initial_composition=batch.initial_composition,
        natural_rows=batch.natural_rows,
        branch_rows=batch.branch_rows,
        bridge_rows=batch.bridge_rows,
        dose_rows=batch.dose_rows,
        probe_rows=batch.probe_rows,
        selected_edit_rows=batch.selected_edit_rows,
        probe_screen_rows=batch.probe_screen_rows,
        no_op_plain_exact=batch.no_op_plain_exact,
        scientific_digest="",
    )
    return _canonical_digest(_json_ready(asdict(value)))


def _append_observation(
    observations: list[NDArray[np.int64]],
    kinds: list[int],
    composition: NDArray,
    kind: int,
) -> None:
    observations.append(np.asarray(composition, dtype=np.int64).copy())
    kinds.append(int(kind))
    if len(kinds) != len(observations) - 1:
        raise AssertionError("transition-kind buffer lost alignment")


def _buffer_state(
    observations: Sequence[NDArray], kinds: Sequence[int], daughters: Sequence[NDArray]
) -> BufferState:
    observation_count = min(MOLECULAR_WINDOW, len(observations))
    selected_observations = np.asarray(
        observations[-observation_count:], dtype=np.int16
    )
    selected_kinds = np.asarray(
        kinds[-max(0, observation_count - 1) :], dtype=np.int8
    )
    selected_daughters = np.asarray(
        daughters[-GENERATIONAL_WINDOW:], dtype=np.int16
    )
    return BufferState(selected_observations, selected_kinds, selected_daughters)


def _restore_buffer(buffer: BufferState) -> tuple[list[NDArray[np.int64]], list[int], list[NDArray[np.int64]]]:
    return (
        [np.asarray(row, dtype=np.int64).copy() for row in buffer.observations],
        [int(value) for value in buffer.transition_kinds],
        [np.asarray(row, dtype=np.int64).copy() for row in buffer.daughters],
    )


def _nan_score() -> PhiWindowScore:
    return PhiWindowScore(
        revised_phi_r=float("nan"),
        typeset_phi_r=float("nan"),
        text_normalized_phi_r=float("nan"),
        causation=float("nan"),
        emergence=float("nan"),
        synergy_persistence=float("nan"),
        atoms=np.full(len(ATOM_NAMES), np.nan, dtype=np.float64),
        active_dimensions=0,
        partition_a=(),
        partition_b=(),
        observations=0,
        transitions=0,
        digest="missing",
    )


def _score_molecular(
    observations: Sequence[NDArray],
    kinds: Sequence[int],
    *,
    include_typeset: bool,
    growth_only: bool = False,
) -> PhiWindowScore:
    if len(observations) < MOLECULAR_WINDOW:
        return _nan_score()
    counts = np.asarray(observations[-MOLECULAR_WINDOW:], dtype=np.float64)
    recent_kinds = np.asarray(kinds[-(MOLECULAR_WINDOW - 1) :], dtype=np.int8)
    mask = recent_kinds == 0 if growth_only else None
    try:
        return score_phi_window(counts, mask, include_typeset=include_typeset)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return _nan_score()


def _score_generational(
    daughters: Sequence[NDArray], *, include_typeset: bool
) -> PhiWindowScore:
    if len(daughters) < GENERATIONAL_WINDOW:
        return _nan_score()
    try:
        return score_phi_window(
            np.asarray(daughters[-GENERATIONAL_WINDOW:], dtype=np.float64),
            include_typeset=include_typeset,
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return _nan_score()


def _score_fields(prefix: str, score: PhiWindowScore) -> dict[str, Any]:
    output: dict[str, Any] = {
        f"{prefix}_revised": score.revised_phi_r,
        f"{prefix}_typeset": score.typeset_phi_r,
        f"{prefix}_text": score.text_normalized_phi_r,
        f"{prefix}_causation": score.causation,
        f"{prefix}_emergence": score.emergence,
        f"{prefix}_synergy": score.synergy_persistence,
        f"{prefix}_active_dimensions": score.active_dimensions,
        f"{prefix}_partition_a_size": len(score.partition_a),
        f"{prefix}_partition_digest": hashlib.sha256(
            repr((score.partition_a, score.partition_b)).encode("ascii")
        ).hexdigest(),
        f"{prefix}_observations": score.observations,
        f"{prefix}_transitions": score.transitions,
        f"{prefix}_digest": score.digest,
    }
    output.update(
        {
            f"{prefix}_atom_{name}": float(value)
            for name, value in zip(ATOM_NAMES, score.atoms, strict=True)
        }
    )
    return output


def _snapshot_after_record(snapshot: Snapshot, record: FissionRecord) -> Snapshot:
    return Snapshot(
        composition=np.asarray(record.daughter, dtype=np.int64).copy(),
        generation=snapshot.generation + 1,
        inheritance=snapshot.inheritance + (record.h > GardConfig().inheritance_threshold,),
        boundary_h=snapshot.boundary_h + (float(record.h),),
        previous_growth_steps=record.growth_steps,
        cumulative_growth_steps=snapshot.cumulative_growth_steps + record.growth_steps,
    )


def _phase_seed_domain(phase: str) -> str:
    if phase == "pilot":
        return SEED_DOMAINS["pilot_matrix"]
    if phase == "confirmation":
        return SEED_DOMAINS["confirmation_matrix"]
    return SEED_DOMAINS["smoke"]


def _run_natural(
    matrix_id: int,
    beta: NDArray,
    initial: NDArray,
    candidate: str,
    replicate: int,
    spec: RunSpec,
) -> NaturalRun:
    config = GardConfig()
    required = set(spec.landmarks) | {spec.launch_generation}
    for attempt in range(100):
        rng = np.random.default_rng(
            derive_seed(
                _phase_seed_domain(spec.label),
                f"{LABEL}.{spec.label}.main_path",
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
        snapshots: dict[int, Snapshot] = {}
        rows: list[dict[str, Any]] = []
        launch_buffer: BufferState | None = None
        try:
            for generation in range(1, spec.natural_generations + 1):
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
                if generation in required:
                    snapshots[generation] = snapshot
                if generation == spec.launch_generation:
                    launch_buffer = _buffer_state(observations, kinds, daughters)
                if generation >= GENERATIONAL_WINDOW:
                    include_typeset = generation in required
                    molecular = _score_molecular(
                        observations, kinds, include_typeset=include_typeset
                    )
                    growth = _score_molecular(
                        observations,
                        kinds,
                        include_typeset=False,
                        growth_only=True,
                    )
                    generational = _score_generational(
                        daughters, include_typeset=include_typeset
                    )
                    row: dict[str, Any] = {
                        "phase": spec.label,
                        "matrix_id": matrix_id,
                        "candidate": candidate,
                        "replicate": replicate,
                        "generation": generation,
                        "landmark": generation if generation in spec.landmarks else -1,
                        "h": float(traced.record.h),
                        "inherited": int(traced.record.h > config.inheritance_threshold),
                        "trailing_run": trailing_run(snapshot.inheritance),
                        "sr_run5": int(trailing_run(snapshot.inheritance) >= 5),
                        "growth_updates": traced.record.growth_steps,
                        "composition": snapshot.composition.astype(int).tolist(),
                    }
                    row.update(_score_fields("molecular", molecular))
                    row.update(_score_fields("growth_only", growth))
                    row.update(_score_fields("generational", generational))
                    rows.append(row)
            if set(snapshots) != required or launch_buffer is None:
                raise AssertionError("natural lineage omitted a required state")
            lineage_digest = _records_digest(records)
            rng_digest = _canonical_digest(_json_ready(rng.bit_generator.state))
            for row in rows:
                row["lineage_record_digest"] = lineage_digest
                row["final_rng_state_digest"] = rng_digest
            return NaturalRun(
                candidate=candidate,
                replicate=replicate,
                snapshots=snapshots,
                launch_buffer=launch_buffer,
                natural_rows=tuple(rows),
                record_digest=lineage_digest,
                path_attempt=attempt,
            )
        except SimulationError:
            continue
    raise SimulationError(
        f"failed complete natural lineage for {spec.label} c{candidate} m{matrix_id} r{replicate}"
    )


def _branch_rows(
    matrix_id: int,
    run: NaturalRun,
    beta: NDArray,
    predictor: FrozenFullPredictor,
    spec: RunSpec,
) -> list[dict[str, Any]]:
    config = GardConfig()
    natural_lookup = {
        int(row["generation"]): row
        for row in run.natural_rows
        if int(row["generation"]) in spec.landmarks
    }
    rows: list[dict[str, Any]] = []
    for landmark in spec.landmarks:
        snapshot = run.snapshots[landmark]
        targets = np.empty(spec.branches, dtype=np.int8)
        completed = np.empty(spec.branches, dtype=np.int8)
        future_digests: list[str] = []
        future_rng_digests: list[str] = []
        for branch in range(spec.branches):
            rng = np.random.default_rng(
                derive_seed(
                    _phase_seed_domain(spec.label),
                    f"{LABEL}.{spec.label}.f12",
                    run.candidate,
                    matrix_id,
                    run.replicate,
                    landmark,
                    branch,
                )
            )
            records, complete = simulate_future_absorbing(
                snapshot,
                beta,
                config,
                CANDIDATES[run.candidate],
                F12_HORIZON,
                rng,
            )
            targets[branch] = int(
                evaluate_process(records, config.inheritance_threshold).joint_break_run3
            )
            completed[branch] = int(complete)
            future_digests.append(_records_digest(records))
            future_rng_digests.append(
                _canonical_digest(_json_ready(rng.bit_generator.state))
            )
        source = natural_lookup[landmark]
        history = history_features(snapshot, config)
        row = {
            "phase": spec.label,
            "matrix_id": matrix_id,
            "candidate": run.candidate,
            "replicate": run.replicate,
            "landmark": landmark,
            "state_id": f"{spec.label}-c{run.candidate}-m{matrix_id:03d}-r{run.replicate}-g{landmark:03d}",
            "frozen_prediction": predictor.predict_snapshot(
                run.candidate, snapshot, beta, config
            ),
            "history": history.tolist(),
            "targets": targets.tolist(),
            "completed": completed.tolist(),
            "future_record_digests": future_digests,
            "future_rng_state_digests": future_rng_digests,
            "composition": snapshot.composition.astype(int).tolist(),
        }
        for key, value in source.items():
            if key.startswith(("molecular_", "growth_only_", "generational_")):
                row[key] = value
        rows.append(row)
    return rows


def _select_controller_edit(
    arm: str,
    predictor: FrozenFullPredictor,
    candidate: str,
    snapshot: Snapshot,
    beta: NDArray,
    action_rng: np.random.Generator,
) -> tuple[MolecularEdit | None, float, float]:
    config = GardConfig()
    before = predictor.predict_snapshot(candidate, snapshot, beta, config)
    if arm in {"MODEL_STABILIZE", "MODEL_DESTABILIZE"}:
        noop, scores = score_legal_edits(predictor, candidate, snapshot, beta, config)
        probabilities = np.asarray(
            [item.predicted_probability for item in scores], dtype=np.float64
        )
        extreme = probabilities.min() if arm == "MODEL_STABILIZE" else probabilities.max()
        index = int(np.flatnonzero(probabilities == extreme)[0])
        return scores[index].edit, float(noop), float(probabilities[index])
    if arm in {"RULE_STABILIZE", "RULE_DESTABILIZE"}:
        rules = select_outgoing_rule_edits(snapshot.composition, beta)
        name = "RULE_DOWN" if arm == "RULE_STABILIZE" else "RULE_UP"
        edit = rules[name]
        after = predictor.predict_snapshot(
            candidate, edited_snapshot(snapshot, edit), beta, config
        )
        return edit, before, after
    if arm == "RANDOM":
        legal = enumerate_legal_edits(snapshot.composition)
        edit = legal[int(action_rng.integers(0, len(legal)))]
        after = predictor.predict_snapshot(
            candidate, edited_snapshot(snapshot, edit), beta, config
        )
        return edit, before, after
    if arm == "NOOP":
        return None, before, before
    raise ValueError(f"unknown controller arm {arm}")


def _run_bridge_arm(
    matrix_id: int,
    run: NaturalRun,
    beta: NDArray,
    predictor: FrozenFullPredictor,
    spec: RunSpec,
    arm: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[FissionRecord], Snapshot, dict]:
    config = GardConfig()
    observations, kinds, daughters = _restore_buffer(run.launch_buffer)
    snapshot = run.snapshots[spec.launch_generation]
    records: list[FissionRecord] = []
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    rng = np.random.default_rng(
        derive_seed(
            _phase_seed_domain(spec.label),
            f"{LABEL}.{spec.label}.bridge.future",
            run.candidate,
            matrix_id,
            run.replicate,
        )
    )
    action_rng = np.random.default_rng(
        derive_seed(
            SEED_DOMAINS["controller_action"],
            f"{LABEL}.{spec.label}.bridge.action",
            run.candidate,
            matrix_id,
            run.replicate,
        )
    )
    for step in range(1, spec.bridge_horizon + 1):
        try:
            traced = advance_fission_traced(
                snapshot.composition, beta, config, CANDIDATES[run.candidate], rng
            )
        except SimulationError:
            missing = _nan_score()
            for adverse_step in range(step, spec.bridge_horizon + 1):
                adverse: dict[str, Any] = {
                    "phase": spec.label,
                    "campaign": "bridge",
                    "matrix_id": matrix_id,
                    "candidate": run.candidate,
                    "replicate": run.replicate,
                    "arm": arm,
                    "step": adverse_step,
                    "generation": snapshot.generation,
                    "h": float("nan"),
                    "inherited": 0,
                    "trailing_run": 0,
                    "sr_run5": 0,
                    "risk_before": float("nan"),
                    "risk_after": float("nan"),
                    "extinct": 1,
                    "completed_horizon": 0,
                    "composition": snapshot.composition.astype(int).tolist(),
                }
                adverse.update(_score_fields("molecular", missing))
                adverse.update(_score_fields("growth_only", missing))
                adverse.update(_score_fields("generational", missing))
                rows.append(adverse)
            break
        for composition in traced.growth_observations:
            _append_observation(observations, kinds, composition, 0)
        _append_observation(observations, kinds, traced.record.daughter, 1)
        records.append(traced.record)
        snapshot = _snapshot_after_record(snapshot, traced.record)
        daughters.append(snapshot.composition.copy())
        edit, risk_before, risk_after = _select_controller_edit(
            arm, predictor, run.candidate, snapshot, beta, action_rng
        )
        if edit is not None:
            snapshot = edited_snapshot(snapshot, edit)
            _append_observation(observations, kinds, snapshot.composition, 2)
            daughters[-1] = snapshot.composition.copy()
            actions.append(
                {
                    "phase": spec.label,
                    "campaign": "bridge",
                    "matrix_id": matrix_id,
                    "candidate": run.candidate,
                    "replicate": run.replicate,
                    "arm": arm,
                    "step": step,
                    "remove_type": edit.remove_type,
                    "add_type": edit.add_type,
                    "risk_before": risk_before,
                    "risk_after": risk_after,
                }
            )
        include_typeset = step in {20, 40, spec.bridge_horizon}
        molecular = _score_molecular(
            observations, kinds, include_typeset=include_typeset
        )
        growth = _score_molecular(
            observations, kinds, include_typeset=False, growth_only=True
        )
        generational = _score_generational(daughters, include_typeset=include_typeset)
        row: dict[str, Any] = {
            "phase": spec.label,
            "campaign": "bridge",
            "matrix_id": matrix_id,
            "candidate": run.candidate,
            "replicate": run.replicate,
            "arm": arm,
            "step": step,
            "generation": snapshot.generation,
            "h": float(traced.record.h),
            "inherited": int(traced.record.h > config.inheritance_threshold),
            "trailing_run": trailing_run(snapshot.inheritance),
            "sr_run5": int(trailing_run(snapshot.inheritance) >= 5),
            "risk_before": risk_before,
            "risk_after": risk_after,
            "extinct": 0,
            "completed_horizon": 1,
            "composition": snapshot.composition.astype(int).tolist(),
        }
        row.update(_score_fields("molecular", molecular))
        row.update(_score_fields("growth_only", growth))
        row.update(_score_fields("generational", generational))
        rows.append(row)
    record_digest = _records_digest(records)
    rng_state = _json_ready(rng.bit_generator.state)
    rng_digest = _canonical_digest(rng_state)
    for row in rows:
        row["lineage_record_digest"] = record_digest
        row["final_rng_state_digest"] = rng_digest
    return rows, actions, records, snapshot, rng_state


def _plain_noop_exact(
    matrix_id: int,
    run: NaturalRun,
    beta: NDArray,
    spec: RunSpec,
    traced_records: Sequence[FissionRecord],
    traced_snapshot: Snapshot,
    traced_rng_state: dict,
) -> bool:
    rng = np.random.default_rng(
        derive_seed(
            _phase_seed_domain(spec.label),
            f"{LABEL}.{spec.label}.bridge.future",
            run.candidate,
            matrix_id,
            run.replicate,
        )
    )
    snapshot = run.snapshots[spec.launch_generation]
    records: list[FissionRecord] = []
    for _ in range(spec.bridge_horizon):
        try:
            record = advance_fission(
                snapshot.composition,
                beta,
                GardConfig(),
                CANDIDATES[run.candidate],
                rng,
            )
        except SimulationError:
            break
        records.append(record)
        snapshot = _snapshot_after_record(snapshot, record)
    return bool(
        _records_digest(records) == _records_digest(list(traced_records))
        and np.array_equal(snapshot.composition, traced_snapshot.composition)
        and snapshot.inheritance == traced_snapshot.inheritance
        and rng_states_equal(rng.bit_generator.state, traced_rng_state)
    )


def _run_static_edit_lineage(
    matrix_id: int,
    run: NaturalRun,
    beta: NDArray,
    spec: RunSpec,
    campaign: str,
    arm: str,
    edit: MolecularEdit | None,
    horizon: int,
    seed_domain: str,
    score_steps: set[int],
) -> tuple[list[dict[str, Any]], list[FissionRecord]]:
    config = GardConfig()
    observations, kinds, daughters = _restore_buffer(run.launch_buffer)
    snapshot = run.snapshots[spec.launch_generation]
    if edit is not None:
        snapshot = edited_snapshot(snapshot, edit)
        _append_observation(observations, kinds, snapshot.composition, 2)
        daughters[-1] = snapshot.composition.copy()
    rng = np.random.default_rng(
        derive_seed(
            _phase_seed_domain(spec.label),
            seed_domain,
            run.candidate,
            matrix_id,
            run.replicate,
        )
    )
    records: list[FissionRecord] = []
    rows: list[dict[str, Any]] = []
    for step in range(1, horizon + 1):
        try:
            traced = advance_fission_traced(
                snapshot.composition, beta, config, CANDIDATES[run.candidate], rng
            )
        except SimulationError:
            outcome = evaluate_process(records, config.inheritance_threshold)
            missing = _nan_score()
            inherited_count = sum(
                record.h > config.inheritance_threshold for record in records
            )
            for adverse_step in sorted(value for value in score_steps if value >= step):
                adverse: dict[str, Any] = {
                    "phase": spec.label,
                    "campaign": campaign,
                    "matrix_id": matrix_id,
                    "candidate": run.candidate,
                    "replicate": run.replicate,
                    "arm": arm,
                    "step": adverse_step,
                    "inherited_fraction": inherited_count / adverse_step,
                    "joint_break_run3": int(outcome.joint_break_run3),
                    "episode_3": float(outcome.episode_3),
                    "extinct": 1,
                    "completed_horizon": 0,
                    "composition": snapshot.composition.astype(int).tolist(),
                }
                adverse.update(_score_fields("molecular", missing))
                adverse.update(_score_fields("growth_only", missing))
                adverse.update(_score_fields("generational", missing))
                rows.append(adverse)
            break
        for composition in traced.growth_observations:
            _append_observation(observations, kinds, composition, 0)
        _append_observation(observations, kinds, traced.record.daughter, 1)
        snapshot = _snapshot_after_record(snapshot, traced.record)
        daughters.append(snapshot.composition.copy())
        records.append(traced.record)
        if step not in score_steps:
            continue
        molecular = _score_molecular(observations, kinds, include_typeset=True)
        growth = _score_molecular(
            observations, kinds, include_typeset=False, growth_only=True
        )
        generational = _score_generational(daughters, include_typeset=True)
        outcome = evaluate_process(records, config.inheritance_threshold)
        row: dict[str, Any] = {
            "phase": spec.label,
            "campaign": campaign,
            "matrix_id": matrix_id,
            "candidate": run.candidate,
            "replicate": run.replicate,
            "arm": arm,
            "step": step,
            "inherited_fraction": float(
                np.mean([record.h > config.inheritance_threshold for record in records])
            ),
            "joint_break_run3": int(outcome.joint_break_run3),
            "episode_3": float(outcome.episode_3),
            "extinct": 0,
            "completed_horizon": 1,
            "composition": snapshot.composition.astype(int).tolist(),
        }
        row.update(_score_fields("molecular", molecular))
        row.update(_score_fields("growth_only", growth))
        row.update(_score_fields("generational", generational))
        rows.append(row)
    record_digest = _records_digest(records)
    rng_digest = _canonical_digest(_json_ready(rng.bit_generator.state))
    for row in rows:
        row["lineage_record_digest"] = record_digest
        row["final_rng_state_digest"] = rng_digest
    return rows, records


def _dose_phase(
    matrix_id: int,
    run: NaturalRun,
    beta: NDArray,
    predictor: FrozenFullPredictor,
    spec: RunSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snapshot = run.snapshots[spec.launch_generation]
    noop, scores = score_legal_edits(
        predictor, run.candidate, snapshot, beta, GardConfig()
    )
    ordered = sorted(
        scores,
        key=lambda item: (
            item.predicted_probability,
            item.edit.remove_type,
            item.edit.add_type,
        ),
    )
    rows: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    score_steps = {max(1, spec.dose_horizon // 2), spec.dose_horizon}
    for quantile in spec.dose_quantiles:
        index = int(round(float(quantile) * (len(ordered) - 1)))
        selected = ordered[index]
        arm = f"Q{int(round(quantile * 100)):03d}"
        local, _ = _run_static_edit_lineage(
            matrix_id,
            run,
            beta,
            spec,
            "dose",
            arm,
            selected.edit,
            spec.dose_horizon,
            f"{LABEL}.{spec.label}.dose.future",
            score_steps,
        )
        for row in local:
            row["dose_quantile"] = quantile
            row["predicted_probability"] = selected.predicted_probability
            row["predicted_shift"] = selected.predicted_probability - noop
        rows.extend(local)
        edits.append(
            {
                "phase": spec.label,
                "campaign": "dose",
                "matrix_id": matrix_id,
                "candidate": run.candidate,
                "replicate": run.replicate,
                "arm": arm,
                "step": 0,
                "remove_type": selected.edit.remove_type,
                "add_type": selected.edit.add_type,
                "risk_before": noop,
                "risk_after": selected.predicted_probability,
            }
        )
    return rows, edits


def _probe_candidate_set(
    matrix_id: int,
    run: NaturalRun,
    beta: NDArray,
    predictor: FrozenFullPredictor,
    spec: RunSpec,
) -> tuple[tuple[MolecularEdit, ...], tuple, float]:
    snapshot = run.snapshots[spec.launch_generation]
    noop, scores = score_legal_edits(
        predictor, run.candidate, snapshot, beta, GardConfig()
    )
    ordered_scores = sorted(
        scores,
        key=lambda item: (
            item.predicted_probability,
            item.edit.remove_type,
            item.edit.add_type,
        ),
    )
    rules = select_outgoing_rule_edits(snapshot.composition, beta)
    special = [
        ordered_scores[0].edit,
        ordered_scores[-1].edit,
        rules["RULE_DOWN"],
        rules["RULE_UP"],
    ]
    legal = enumerate_legal_edits(snapshot.composition)
    selection_rng = np.random.default_rng(
        derive_seed(
            SEED_DOMAINS["probe_selection"],
            f"{LABEL}.{spec.label}.probe.candidates",
            run.candidate,
            matrix_id,
            run.replicate,
        )
    )
    selected: list[MolecularEdit] = []
    seen: set[MolecularEdit] = set()
    for edit in special:
        if edit not in seen:
            selected.append(edit)
            seen.add(edit)
    for index in selection_rng.permutation(len(legal)):
        edit = legal[int(index)]
        if edit not in seen:
            selected.append(edit)
            seen.add(edit)
        if len(selected) == min(spec.probe_candidates, len(legal)):
            break
    return tuple(sorted(selected)), tuple(ordered_scores), float(noop)


def _probe_end_score(
    matrix_id: int,
    run: NaturalRun,
    beta: NDArray,
    spec: RunSpec,
    edit: MolecularEdit,
    rollout: int,
) -> tuple[float, str, str, bool]:
    observations, kinds, daughters = _restore_buffer(run.launch_buffer)
    snapshot = edited_snapshot(run.snapshots[spec.launch_generation], edit)
    _append_observation(observations, kinds, snapshot.composition, 2)
    daughters[-1] = snapshot.composition.copy()
    rng = np.random.default_rng(
        derive_seed(
            SEED_DOMAINS["probe_screen"],
            f"{LABEL}.{spec.label}.probe.screen",
            run.candidate,
            matrix_id,
            run.replicate,
            rollout,
        )
    )
    records: list[FissionRecord] = []
    completed = True
    for _ in range(spec.probe_horizon):
        try:
            traced = advance_fission_traced(
                snapshot.composition, beta, GardConfig(), CANDIDATES[run.candidate], rng
            )
        except SimulationError:
            completed = False
            break
        for composition in traced.growth_observations:
            _append_observation(observations, kinds, composition, 0)
        _append_observation(observations, kinds, traced.record.daughter, 1)
        snapshot = _snapshot_after_record(snapshot, traced.record)
        daughters.append(snapshot.composition.copy())
        records.append(traced.record)
    score = _score_molecular(observations, kinds, include_typeset=False).revised_phi_r
    return (
        score,
        _records_digest(records),
        _canonical_digest(_json_ready(rng.bit_generator.state)),
        completed,
    )


def _probe_phase(
    matrix_id: int,
    run: NaturalRun,
    beta: NDArray,
    predictor: FrozenFullPredictor,
    spec: RunSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, ordered_scores, noop = _probe_candidate_set(
        matrix_id, run, beta, predictor, spec
    )
    screen_rows: list[dict[str, Any]] = []
    means: list[float] = []
    for edit in candidates:
        rollouts = [
            _probe_end_score(matrix_id, run, beta, spec, edit, rollout)
            for rollout in range(spec.probe_rollouts)
        ]
        values = np.asarray([item[0] for item in rollouts], dtype=np.float64)
        mean = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        means.append(mean)
        screen_rows.append(
            {
                "phase": spec.label,
                "matrix_id": matrix_id,
                "candidate": run.candidate,
                "replicate": run.replicate,
                "remove_type": edit.remove_type,
                "add_type": edit.add_type,
                "probe_phi_mean": mean,
                "probe_phi_values": json.dumps(values.tolist()),
                "probe_record_digests": json.dumps([item[1] for item in rollouts]),
                "probe_rng_state_digests": json.dumps([item[2] for item in rollouts]),
                "probe_rollouts_complete": json.dumps([item[3] for item in rollouts]),
            }
        )
    finite = np.isfinite(means)
    if not finite.any():
        raise ValueError("all bounded Phi probe candidates were undefined")
    maximum = float(np.nanmax(means))
    minimum = float(np.nanmin(means))
    up = candidates[int(np.flatnonzero(np.asarray(means) == maximum)[0])]
    down = candidates[int(np.flatnonzero(np.asarray(means) == minimum)[0])]
    legal = enumerate_legal_edits(run.snapshots[spec.launch_generation].composition)
    random_rng = np.random.default_rng(
        derive_seed(
            SEED_DOMAINS["probe_selection"],
            f"{LABEL}.{spec.label}.probe.random",
            run.candidate,
            matrix_id,
            run.replicate,
        )
    )
    random_edit = legal[int(random_rng.integers(0, len(legal)))]
    arm_edits = {"PHI_UP": up, "PHI_DOWN": down, "RANDOM": random_edit, "NOOP": None}
    probability_lookup = {
        item.edit: item.predicted_probability for item in ordered_scores
    }
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    score_steps = {max(1, spec.probe_confirm_horizon // 2), spec.probe_confirm_horizon}
    for arm in spec.probe_arms:
        edit = arm_edits[arm]
        local, _ = _run_static_edit_lineage(
            matrix_id,
            run,
            beta,
            spec,
            "probe_confirmation",
            arm,
            edit,
            spec.probe_confirm_horizon,
            f"{LABEL}.{spec.label}.probe.confirmation",
            score_steps,
        )
        rows.extend(local)
        if edit is not None:
            selected_rows.append(
                {
                    "phase": spec.label,
                    "campaign": "probe_confirmation",
                    "matrix_id": matrix_id,
                    "candidate": run.candidate,
                    "replicate": run.replicate,
                    "arm": arm,
                    "step": 0,
                    "remove_type": edit.remove_type,
                    "add_type": edit.add_type,
                    "risk_before": noop,
                    "risk_after": probability_lookup[edit],
                }
            )
    return rows, selected_rows, screen_rows


def _run_matrix(args: tuple[int, RunSpec, str]) -> CampaignBatch:
    matrix_id, spec, model_path = args
    with threadpool_limits(limits=1):
        config = GardConfig()
        matrix_rng = np.random.default_rng(
            derive_seed(
                _phase_seed_domain(spec.label),
                f"{LABEL}.{spec.label}.beta",
                matrix_id,
            )
        )
        initial_rng = np.random.default_rng(
            derive_seed(
                SEED_DOMAINS["initial"],
                f"{LABEL}.{spec.label}.initial",
                matrix_id,
            )
        )
        beta = generate_beta(config, matrix_rng)
        initial = generate_initial_composition(config, initial_rng)
        predictor = FrozenFullPredictor.load(model_path)
        natural_rows: list[dict[str, Any]] = []
        branch_rows: list[dict[str, Any]] = []
        bridge_rows: list[dict[str, Any]] = []
        dose_rows: list[dict[str, Any]] = []
        probe_rows: list[dict[str, Any]] = []
        edit_rows: list[dict[str, Any]] = []
        screen_rows: list[dict[str, Any]] = []
        no_op_exact: list[bool] = []
        digest = hashlib.sha256()
        for candidate in CANDIDATES:
            for replicate in range(spec.replicates):
                natural = _run_natural(
                    matrix_id, beta, initial, candidate, replicate, spec
                )
                natural_rows.extend(natural.natural_rows)
                branch_rows.extend(
                    _branch_rows(matrix_id, natural, beta, predictor, spec)
                )
                for arm in spec.bridge_arms:
                    local_rows, local_edits, records, final_snapshot, rng_state = _run_bridge_arm(
                        matrix_id, natural, beta, predictor, spec, arm
                    )
                    bridge_rows.extend(local_rows)
                    edit_rows.extend(local_edits)
                    digest.update(_records_digest(records).encode("ascii"))
                    if arm == "NOOP":
                        no_op_exact.append(
                            _plain_noop_exact(
                                matrix_id,
                                natural,
                                beta,
                                spec,
                                records,
                                final_snapshot,
                                rng_state,
                            )
                        )
                local_dose, dose_edits = _dose_phase(
                    matrix_id, natural, beta, predictor, spec
                )
                dose_rows.extend(local_dose)
                edit_rows.extend(dose_edits)
                local_probe, probe_edits, local_screen = _probe_phase(
                    matrix_id, natural, beta, predictor, spec
                )
                probe_rows.extend(local_probe)
                edit_rows.extend(probe_edits)
                screen_rows.extend(local_screen)
                digest.update(natural.record_digest.encode("ascii"))
        provisional = CampaignBatch(
            matrix_id=matrix_id,
            beta=np.asarray(beta, dtype=np.float64),
            initial_composition=np.asarray(initial, dtype=np.int16),
            natural_rows=tuple(natural_rows),
            branch_rows=tuple(branch_rows),
            bridge_rows=tuple(bridge_rows),
            dose_rows=tuple(dose_rows),
            probe_rows=tuple(probe_rows),
            selected_edit_rows=tuple(edit_rows),
            probe_screen_rows=tuple(screen_rows),
            no_op_plain_exact=bool(no_op_exact and all(no_op_exact)),
            scientific_digest=digest.hexdigest(),
        )
        return CampaignBatch(**{**asdict(provisional), "scientific_digest": _batch_digest(provisional)})


def _write_status(
    work: Path,
    stage: str,
    completed: int,
    total: int,
    **extra: Any,
) -> None:
    safe_stage = stage.replace("/", "_").replace(" ", "_")
    started_path = work / f"started_at_{safe_stage}.txt"
    if not started_path.exists():
        started_path.parent.mkdir(parents=True, exist_ok=True)
        started_path.write_text(str(time.time()), encoding="ascii")
    started = float(started_path.read_text(encoding="ascii"))
    elapsed = max(0.0, time.time() - started)
    rate = completed / elapsed if completed > 0 and elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0.0 else None
    value = {
        "format": STATUS_FORMAT,
        "stage": stage,
        "completed": completed,
        "total": total,
        "fraction": completed / total if total else 1.0,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "pid": os.getpid(),
        **extra,
    }
    _atomic_json(work / "campaign_status.json", value)


def _checkpoint_contract(spec: RunSpec, registration_id: str, stage: str) -> dict[str, Any]:
    value = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "stage": stage,
        "spec": asdict(spec),
        "source_hashes": _source_hashes(),
    }
    value["contract_id"] = _canonical_digest(value)
    return value


def _run_checkpointed(
    spec: RunSpec,
    registration_id: str,
    directory: Path,
    work: Path,
    stage: str,
    workers: int,
) -> list[CampaignBatch]:
    directory.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(spec, registration_id, stage)
    contract_path = directory / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(_json_ready(contract))
        ):
            raise ValueError("Chapter 5 checkpoint contract changed")
    else:
        _atomic_json(contract_path, contract)
    batches: list[CampaignBatch | None] = [None] * spec.matrices
    missing: list[int] = []
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:04d}.pkl"
        if path.is_file():
            with path.open("rb") as handle:
                batch = pickle.load(handle)
            if not isinstance(batch, CampaignBatch) or batch.matrix_id != matrix_id:
                raise ValueError(f"invalid Chapter 5 checkpoint {path}")
            if batch.scientific_digest != _batch_digest(batch):
                raise ValueError(f"Chapter 5 checkpoint digest mismatch {path}")
            batches[matrix_id] = batch
        else:
            missing.append(matrix_id)
    completed = spec.matrices - len(missing)
    _write_status(work, stage, completed, spec.matrices, reused=completed)
    arguments = [(matrix_id, spec, str(DEFAULT_REGISTRATION / "frozen_full_predictor.npz")) for matrix_id in missing]
    generated: Iterable[CampaignBatch]
    if workers <= 1:
        generated = map(_run_matrix, arguments)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        generated = executor.map(_run_matrix, arguments, chunksize=1)
    try:
        for matrix_id, batch in zip(missing, generated, strict=True):
            if batch.matrix_id != matrix_id:
                raise AssertionError(
                    "Chapter 5 worker returned matrix "
                    f"{batch.matrix_id}, expected {matrix_id}"
                )
            observed_digest = _batch_digest(batch)
            if batch.scientific_digest != observed_digest:
                raise AssertionError(
                    "Chapter 5 worker batch content digest mismatch: "
                    f"stored={batch.scientific_digest}, observed={observed_digest}"
                )
            batches[matrix_id] = batch
            _atomic_pickle(directory / f"matrix_{matrix_id:04d}.pkl", batch)
            completed += 1
            _write_status(
                work,
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
        raise AssertionError("Chapter 5 checkpoint stage is incomplete")
    return [batch for batch in batches if batch is not None]


def _rows_frame(batches: Sequence[CampaignBatch], field: str) -> pd.DataFrame:
    rows = [row for batch in batches for row in getattr(batch, field)]
    return pd.DataFrame(rows)


def _seeded_rng(domain: str, *keys: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(SEED_DOMAINS[domain], LABEL, *keys))


def _mean_bootstrap(
    values: NDArray,
    repetitions: int,
    key: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.full(repetitions, np.nan, dtype=np.float64)
    rng = _seeded_rng("bootstrap", key)
    indices = rng.integers(0, array.size, size=(repetitions, array.size))
    return np.asarray(array[indices].mean(axis=1), dtype=np.float64)


def _sign_randomization_p(
    values: NDArray,
    repetitions: int,
    key: str,
) -> tuple[float, NDArray[np.float64]]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), np.full(repetitions, np.nan, dtype=np.float64)
    observed = float(array.mean())
    rng = _seeded_rng("randomization", key)
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(repetitions, array.size))
    null = np.asarray((signs * array).mean(axis=1), dtype=np.float64)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (repetitions + 1))
    return p_value, null


def _paired_summary(
    values: NDArray,
    repetitions: int,
    key: str,
    arrays: dict[str, NDArray],
    *,
    equivalence_margin: float | None = None,
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    bootstrap = _mean_bootstrap(vector, repetitions, key)
    p_value, randomized = _sign_randomization_p(vector, repetitions, key)
    finite_bootstrap = bootstrap[np.isfinite(bootstrap)]
    if finite_bootstrap.size:
        lower, upper = np.quantile(finite_bootstrap, (0.025, 0.975))
        lower90, upper90 = np.quantile(finite_bootstrap, (0.05, 0.95))
    else:
        lower = upper = lower90 = upper90 = float("nan")
    safe = key.replace("/", "__").replace(" ", "_")
    arrays[f"{safe}__matrix_values"] = vector
    arrays[f"{safe}__bootstrap"] = bootstrap
    arrays[f"{safe}__sign_randomization"] = randomized
    result: dict[str, Any] = {
        "effect": float(vector.mean()) if vector.size else float("nan"),
        "ci95": [float(lower), float(upper)],
        "ci90": [float(lower90), float(upper90)],
        "one_sided_sign_randomization_p": p_value,
        "matrices": int(vector.size),
        "matrices_positive": int(np.count_nonzero(vector > 0.0)),
        "maximum_absolute_matrix_effect": (
            float(np.max(np.abs(vector))) if vector.size else float("nan")
        ),
    }
    if equivalence_margin is not None:
        result["equivalence_margin"] = float(equivalence_margin)
        result["tost_via_90ci"] = bool(
            np.isfinite(lower90)
            and lower90 > -equivalence_margin
            and upper90 < equivalence_margin
        )
    return result


def _arm_effects(
    frame: pd.DataFrame,
    value: str,
    high: str,
    low: str,
    *,
    candidate: str,
    replicate: int | None,
) -> NDArray[np.float64]:
    selected = frame[frame["candidate"] == candidate]
    if replicate is not None:
        selected = selected[selected["replicate"] == replicate]
    group_columns = ["matrix_id", "arm"]
    means = selected.groupby(group_columns, sort=True)[value].mean().unstack("arm")
    if high not in means or low not in means:
        return np.empty(0, dtype=np.float64)
    return np.asarray((means[high] - means[low]).dropna(), dtype=np.float64)


def _apply_holm(items: list[dict[str, Any]], field: str = "one_sided_sign_randomization_p") -> None:
    finite_locations = [
        index for index, item in enumerate(items) if np.isfinite(item.get(field, np.nan))
    ]
    if not finite_locations:
        return
    adjusted = holm_adjust([float(items[index][field]) for index in finite_locations])
    for index, value in zip(finite_locations, adjusted, strict=True):
        items[index]["holm_adjusted_p"] = float(value)


def _bridge_analysis(
    bridge: pd.DataFrame,
    spec: RunSpec,
    arrays: dict[str, NDArray],
) -> dict[str, Any]:
    late = bridge[bridge["step"] > spec.bridge_horizon // 2].copy()
    metrics = (
        "inherited",
        "molecular_revised",
        "molecular_typeset",
        "molecular_text",
        "molecular_causation",
        "molecular_emergence",
        "molecular_synergy",
        "growth_only_revised",
        "generational_revised",
        "generational_typeset",
    )
    contrasts = {
        "model": ("MODEL_STABILIZE", "MODEL_DESTABILIZE"),
        "rule": ("RULE_STABILIZE", "RULE_DESTABILIZE"),
        "random_specificity": ("RANDOM", "NOOP"),
    }
    cells: list[dict[str, Any]] = []
    for contrast, (high, low) in contrasts.items():
        if high not in set(late["arm"]) or low not in set(late["arm"]):
            continue
        for metric in metrics:
            if metric not in late:
                continue
            for candidate in CANDIDATES:
                for replicate in range(spec.replicates):
                    values = _arm_effects(
                        late,
                        metric,
                        high,
                        low,
                        candidate=candidate,
                        replicate=replicate,
                    )
                    margin = None
                    if contrast == "random_specificity" and metric == "inherited":
                        margin = PROBABILITY_EQUIVALENCE_MARGIN
                    summary = _paired_summary(
                        values,
                        spec.bootstrap_repetitions,
                        f"{spec.label}/bridge/{contrast}/{metric}/c{candidate}/r{replicate}",
                        arrays,
                        equivalence_margin=margin,
                    )
                    summary.update(
                        {
                            "contrast": contrast,
                            "high_arm": high,
                            "low_arm": low,
                            "metric": metric,
                            "candidate": candidate,
                            "replicate": replicate,
                        }
                    )
                    cells.append(summary)
    for contrast in contrasts:
        for metric in metrics:
            _apply_holm(
                [
                    item
                    for item in cells
                    if item["contrast"] == contrast and item["metric"] == metric
                ]
            )
    primary = [
        item
        for item in cells
        if item["contrast"] == "model"
        and item["metric"] == "molecular_revised"
    ]
    validity = [
        item
        for item in cells
        if item["contrast"] == "model" and item["metric"] == "inherited"
    ]
    return {
        "late_window": [spec.bridge_horizon // 2 + 1, spec.bridge_horizon],
        "cells": cells,
        "model_revised_response_gate": bool(
            len(primary) == 2 * spec.replicates
            and all(
                item["effect"] > 0.0
                and item["ci95"][0] > 0.0
                and item.get("holm_adjusted_p", 1.0) < 0.05
                for item in primary
            )
        ),
        "model_heredity_validity_gate": bool(
            len(validity) == 2 * spec.replicates
            and all(
                item["effect"] > 0.0
                and item["ci95"][0] > 0.0
                and item.get("holm_adjusted_p", 1.0) < 0.05
                for item in validity
            )
        ),
    }


def _state_reading_analysis(
    natural: pd.DataFrame,
    bridge: pd.DataFrame,
    spec: RunSpec,
    arrays: dict[str, NDArray],
) -> dict[str, Any]:
    sources: list[tuple[str, pd.DataFrame, list[str]]] = [
        ("natural", natural.copy(), ["matrix_id", "candidate", "replicate"]),
        (
            "bridge",
            bridge.copy(),
            ["matrix_id", "candidate", "replicate", "arm"],
        ),
    ]
    metrics = (
        "molecular_revised",
        "molecular_typeset",
        "molecular_causation",
        "molecular_emergence",
        "molecular_synergy",
        "generational_revised",
        "generational_typeset",
    )
    results: list[dict[str, Any]] = []
    for source, frame, lineage_columns in sources:
        for candidate in CANDIDATES:
            candidate_frame = frame[frame["candidate"] == candidate]
            for metric in metrics:
                lineage_values: list[dict[str, Any]] = []
                for keys, group in candidate_frame.groupby(lineage_columns, sort=True):
                    values_sr = np.asarray(group.loc[group["sr_run5"] == 1, metric], dtype=float)
                    values_other = np.asarray(group.loc[group["sr_run5"] == 0, metric], dtype=float)
                    values_sr = values_sr[np.isfinite(values_sr)]
                    values_other = values_other[np.isfinite(values_other)]
                    if values_sr.size and values_other.size:
                        matrix_id = int(keys[0] if isinstance(keys, tuple) else keys)
                        lineage_values.append(
                            {
                                "matrix_id": matrix_id,
                                "difference": float(values_sr.mean() - values_other.mean()),
                            }
                        )
                if lineage_values:
                    local = pd.DataFrame(lineage_values)
                    vector = np.asarray(
                        local.groupby("matrix_id")["difference"].mean(), dtype=np.float64
                    )
                else:
                    vector = np.empty(0, dtype=np.float64)
                summary = _paired_summary(
                    vector,
                    spec.bootstrap_repetitions,
                    f"{spec.label}/state_reading/{source}/{metric}/c{candidate}",
                    arrays,
                )
                summary.update(
                    {
                        "source": source,
                        "metric": metric,
                        "candidate": candidate,
                        "eligible_lineages": len(lineage_values),
                    }
                )
                results.append(summary)
    for source in ("natural", "bridge"):
        for metric in metrics:
            _apply_holm(
                [
                    item
                    for item in results
                    if item["source"] == source and item["metric"] == metric
                ]
            )
    primary = [
        item
        for item in results
        if item["source"] == "bridge" and item["metric"] == "molecular_revised"
    ]
    return {
        "definition": "within-lineage mean Phi during trailing inheritance run >=5 minus other states",
        "results": results,
        "bridge_revised_reads_state_gate": bool(
            len(primary) == 2
            and all(
                item["effect"] > 0.0
                and item["ci95"][0] > 0.0
                and item.get("holm_adjusted_p", 1.0) < 0.05
                for item in primary
            )
        ),
    }


def _branch_state_frame(branch: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in branch.to_dict(orient="records"):
        targets = np.asarray(item.pop("targets"), dtype=np.float64)
        completed = np.asarray(item.pop("completed"), dtype=np.float64)
        split = targets.size // 2
        if split == 0:
            continue
        for half, selection in (("A", slice(0, split)), ("B", slice(split, targets.size))):
            values = targets[selection]
            completion = completed[selection]
            row = dict(item)
            row.update(
                {
                    "half": half,
                    "q": float(values.mean()),
                    "branches": int(values.size),
                    "complete_fraction": float(completion.mean()),
                    "target_vector": values.astype(np.int8).tolist(),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_centered_correlation(
    frame: pd.DataFrame,
    left: str,
    right: str,
    repetitions: int,
    key: str,
) -> NDArray[np.float64]:
    matrices = np.asarray(sorted(frame["matrix_id"].unique()), dtype=np.int64)
    locations = {
        matrix_id: np.flatnonzero(np.asarray(frame["matrix_id"]) == matrix_id)
        for matrix_id in matrices
    }
    left_values = np.asarray(frame[left], dtype=np.float64)
    right_values = np.asarray(frame[right], dtype=np.float64)
    rng = _seeded_rng("bootstrap", key)
    output = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.choice(matrices, size=matrices.size, replace=True)
        indices = np.concatenate([locations[int(value)] for value in sampled])
        groups = np.concatenate(
            [np.full(locations[int(value)].size, index) for index, value in enumerate(sampled)]
        )
        output[repetition] = centered_spearman(
            left_values[indices], right_values[indices], groups
        )
    return output


def _correlation_summary(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    repetitions: int,
    key: str,
    arrays: dict[str, NDArray],
) -> dict[str, Any]:
    valid = np.isfinite(np.asarray(frame[predictor], dtype=float)) & np.isfinite(
        np.asarray(frame[outcome], dtype=float)
    )
    local = frame.loc[valid].copy()
    if local.empty or local["matrix_id"].nunique() < 2:
        observed = float("nan")
        bootstrap = np.full(repetitions, np.nan)
    else:
        observed = centered_spearman(
            np.asarray(local[predictor], dtype=float),
            np.asarray(local[outcome], dtype=float),
            np.asarray(local["matrix_id"]),
        )
        bootstrap = _bootstrap_centered_correlation(
            local, predictor, outcome, repetitions, key
        )
    finite = bootstrap[np.isfinite(bootstrap)]
    ci = (
        [float(value) for value in np.quantile(finite, (0.025, 0.975))]
        if finite.size
        else [float("nan"), float("nan")]
    )
    ci90 = (
        [float(value) for value in np.quantile(finite, (0.05, 0.95))]
        if finite.size
        else [float("nan"), float("nan")]
    )
    safe = key.replace("/", "__")
    arrays[f"{safe}__bootstrap"] = bootstrap
    return {
        "predictor": predictor,
        "outcome": outcome,
        "centered_spearman": float(observed),
        "ci95": ci,
        "ci90": ci90,
        "states": int(len(local)),
        "matrices": int(local["matrix_id"].nunique()),
        "equivalent_within_0.10": bool(
            np.isfinite(ci90[0])
            and ci90[0] > -FORESIGHT_CORRELATION_MARGIN
            and ci90[1] < FORESIGHT_CORRELATION_MARGIN
        ),
    }


def _logit(values: NDArray) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped))


def _fit_predict_binary(
    train_x: NDArray,
    train_y: NDArray,
    test_x: NDArray,
) -> NDArray[np.float64]:
    train = np.asarray(train_x, dtype=np.float64)
    test = np.asarray(test_x, dtype=np.float64)
    outcomes = np.asarray(train_y, dtype=np.int8)
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(train), train, medians)
    test = np.where(np.isfinite(test), test, medians)
    if np.unique(outcomes).size < 2:
        return np.full(test.shape[0], float(np.mean(outcomes)))
    scaler = StandardScaler().fit(train)
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2_000,
        random_state=0,
    ).fit(scaler.transform(train), outcomes)
    return np.asarray(
        classifier.predict_proba(scaler.transform(test))[:, 1], dtype=np.float64
    )


def _crossfit_incremental_logloss(
    frame: pd.DataFrame,
    repetitions: int,
    key: str,
    arrays: dict[str, NDArray],
) -> dict[str, Any]:
    rows = frame.to_dict(orient="records")
    base_state: list[list[float]] = []
    enhanced_state: list[list[float]] = []
    targets: list[NDArray[np.int8]] = []
    matrices: list[int] = []
    phi_columns = (
        "molecular_revised",
        "molecular_causation",
        "molecular_emergence",
        "molecular_synergy",
        "molecular_typeset",
        "generational_revised",
        "generational_typeset",
    )
    for row in rows:
        history = [float(value) for value in row["history"]]
        base = [_logit(np.asarray([row["frozen_prediction"]]))[0], *history]
        phi = [float(row.get(column, np.nan)) for column in phi_columns]
        base_state.append(base)
        enhanced_state.append([*base, *phi])
        targets.append(np.asarray(row["target_vector"], dtype=np.int8))
        matrices.append(int(row["matrix_id"]))
    base_array = np.asarray(base_state, dtype=np.float64)
    enhanced_array = np.asarray(enhanced_state, dtype=np.float64)
    matrix_array = np.asarray(matrices, dtype=np.int64)
    base_prediction = np.full(len(rows), np.nan, dtype=np.float64)
    enhanced_prediction = np.full(len(rows), np.nan, dtype=np.float64)
    for fold in (0, 1):
        test_state = (matrix_array % 2) == fold
        train_state = ~test_state
        train_targets = np.concatenate(
            [targets[index] for index in np.flatnonzero(train_state)]
        )
        repeats = np.asarray(
            [targets[index].size for index in np.flatnonzero(train_state)], dtype=int
        )
        base_train = np.repeat(base_array[train_state], repeats, axis=0)
        enhanced_train = np.repeat(enhanced_array[train_state], repeats, axis=0)
        base_prediction[test_state] = _fit_predict_binary(
            base_train, train_targets, base_array[test_state]
        )
        enhanced_prediction[test_state] = _fit_predict_binary(
            enhanced_train, train_targets, enhanced_array[test_state]
        )
    losses: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        baseline = float(np.clip(base_prediction[index], 1e-12, 1.0 - 1e-12))
        enhanced = float(np.clip(enhanced_prediction[index], 1e-12, 1.0 - 1e-12))
        base_loss = float(
            -np.mean(target * np.log(baseline) + (1 - target) * np.log(1 - baseline))
        )
        enhanced_loss = float(
            -np.mean(target * np.log(enhanced) + (1 - target) * np.log(1 - enhanced))
        )
        losses.append(
            {
                "matrix_id": matrix_array[index],
                "gain": base_loss - enhanced_loss,
            }
        )
    loss_frame = pd.DataFrame(losses)
    vector = np.asarray(loss_frame.groupby("matrix_id")["gain"].mean(), dtype=float)
    summary = _paired_summary(vector, repetitions, key, arrays)
    summary["positive_gain_means_phi_improves_log_loss"] = True
    summary["equivalent_within_0.005"] = bool(
        summary["ci90"][0] > -FORESIGHT_LOGLOSS_MARGIN
        and summary["ci90"][1] < FORESIGHT_LOGLOSS_MARGIN
    )
    arrays[f"{key.replace('/', '__')}__base_prediction"] = base_prediction
    arrays[f"{key.replace('/', '__')}__enhanced_prediction"] = enhanced_prediction
    return summary


def _foresight_analysis(
    branch: pd.DataFrame,
    spec: RunSpec,
    arrays: dict[str, NDArray],
) -> tuple[dict[str, Any], pd.DataFrame]:
    states = _branch_state_frame(branch)
    predictors = (
        "frozen_prediction",
        "molecular_revised",
        "molecular_typeset",
        "molecular_text",
        "molecular_causation",
        "molecular_emergence",
        "molecular_synergy",
        "growth_only_revised",
        "generational_revised",
        "generational_typeset",
    )
    correlations: list[dict[str, Any]] = []
    incremental: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for half in ("A", "B"):
            local = states[
                (states["candidate"] == candidate) & (states["half"] == half)
            ].copy()
            for predictor in predictors:
                summary = _correlation_summary(
                    local,
                    predictor,
                    "q",
                    spec.bootstrap_repetitions,
                    f"{spec.label}/foresight/{predictor}/c{candidate}/h{half}",
                    arrays,
                )
                summary.update({"candidate": candidate, "half": half})
                correlations.append(summary)
            gain = _crossfit_incremental_logloss(
                local,
                spec.bootstrap_repetitions,
                f"{spec.label}/foresight/incremental/c{candidate}/h{half}",
                arrays,
            )
            gain.update({"candidate": candidate, "half": half})
            incremental.append(gain)
    validity = [item for item in correlations if item["predictor"] == "frozen_prediction"]
    revised = [item for item in correlations if item["predictor"] == "molecular_revised"]
    return (
        {
            "correlations": correlations,
            "incremental_log_loss": incremental,
            "frozen_predictor_validity_gate": bool(
                len(validity) == 4
                and all(item["centered_spearman"] > 0.0 and item["ci95"][0] > 0.0 for item in validity)
            ),
            "revised_phi_centered_equivalence_gate": bool(
                len(revised) == 4 and all(item["equivalent_within_0.10"] for item in revised)
            ),
            "incremental_phi_logloss_equivalence_gate": bool(
                len(incremental) == 4
                and all(item["equivalent_within_0.005"] for item in incremental)
            ),
        },
        states,
    )


def _dose_analysis(
    dose: pd.DataFrame,
    spec: RunSpec,
    arrays: dict[str, NDArray],
) -> dict[str, Any]:
    final = dose[dose["step"] == spec.dose_horizon]
    metrics = (
        "molecular_revised",
        "molecular_causation",
        "molecular_emergence",
        "molecular_synergy",
        "generational_revised",
        "inherited_fraction",
    )
    results: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for replicate in range(spec.replicates):
            local = final[
                (final["candidate"] == candidate) & (final["replicate"] == replicate)
            ]
            for metric in metrics:
                values: list[float] = []
                for _, group in local.groupby("matrix_id", sort=True):
                    values.append(
                        spearman(
                            -np.asarray(group["predicted_shift"], dtype=float),
                            np.asarray(group[metric], dtype=float),
                        )
                    )
                summary = _paired_summary(
                    np.asarray(values),
                    spec.bootstrap_repetitions,
                    f"{spec.label}/dose/{metric}/c{candidate}/r{replicate}",
                    arrays,
                )
                summary.update(
                    {
                        "candidate": candidate,
                        "replicate": replicate,
                        "metric": metric,
                        "direction": "larger stabilizing predicted shift mapped to larger metric",
                    }
                )
                results.append(summary)
    for metric in metrics:
        _apply_holm([item for item in results if item["metric"] == metric])
    primary = [item for item in results if item["metric"] == "molecular_revised"]
    return {
        "results": results,
        "revised_phi_dose_gate": bool(
            len(primary) == 2 * spec.replicates
            and all(
                item["effect"] > 0.0
                and item["ci95"][0] > 0.0
                and item.get("holm_adjusted_p", 1.0) < 0.05
                for item in primary
            )
        ),
    }


def _probe_analysis(
    probe: pd.DataFrame,
    spec: RunSpec,
    arrays: dict[str, NDArray],
) -> dict[str, Any]:
    final = probe[probe["step"] == spec.probe_confirm_horizon]
    metrics = (
        "molecular_revised",
        "molecular_causation",
        "molecular_emergence",
        "molecular_synergy",
        "generational_revised",
        "inherited_fraction",
        "joint_break_run3",
    )
    results: list[dict[str, Any]] = []
    contrasts = {
        "phi_extremes": ("PHI_UP", "PHI_DOWN"),
        "random_specificity": ("RANDOM", "NOOP"),
    }
    for contrast, (high, low) in contrasts.items():
        for metric in metrics:
            for candidate in CANDIDATES:
                for replicate in range(spec.replicates):
                    vector = _arm_effects(
                        final,
                        metric,
                        high,
                        low,
                        candidate=candidate,
                        replicate=replicate,
                    )
                    margin = None
                    if contrast == "phi_extremes" and metric in {
                        "inherited_fraction",
                        "joint_break_run3",
                    }:
                        margin = PROBABILITY_EQUIVALENCE_MARGIN
                    if contrast == "random_specificity" and metric in {
                        "inherited_fraction",
                        "joint_break_run3",
                    }:
                        margin = PROBABILITY_EQUIVALENCE_MARGIN
                    summary = _paired_summary(
                        vector,
                        spec.bootstrap_repetitions,
                        f"{spec.label}/probe/{contrast}/{metric}/c{candidate}/r{replicate}",
                        arrays,
                        equivalence_margin=margin,
                    )
                    summary.update(
                        {
                            "contrast": contrast,
                            "metric": metric,
                            "candidate": candidate,
                            "replicate": replicate,
                            "high_arm": high,
                            "low_arm": low,
                        }
                    )
                    results.append(summary)
    for contrast in contrasts:
        for metric in metrics:
            _apply_holm(
                [
                    item
                    for item in results
                    if item["contrast"] == contrast and item["metric"] == metric
                ]
            )
    gauge = [
        item
        for item in results
        if item["contrast"] == "phi_extremes"
        and item["metric"] == "molecular_revised"
    ]
    heredity = [
        item
        for item in results
        if item["contrast"] == "phi_extremes"
        and item["metric"] == "inherited_fraction"
    ]
    return {
        "results": results,
        "probe_moves_revised_phi_gate": bool(
            len(gauge) == 2 * spec.replicates
            and all(
                item["effect"] > 0.0
                and item["ci95"][0] > 0.0
                and item.get("holm_adjusted_p", 1.0) < 0.05
                for item in gauge
            )
        ),
        "probe_heredity_equivalence_gate": bool(
            len(heredity) == 2 * spec.replicates
            and all(item.get("tost_via_90ci", False) for item in heredity)
        ),
    }


def analyze_batches(
    batches: Sequence[CampaignBatch], spec: RunSpec
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    frames = {
        "natural": _rows_frame(batches, "natural_rows"),
        "branch_states": _rows_frame(batches, "branch_rows"),
        "bridge": _rows_frame(batches, "bridge_rows"),
        "dose": _rows_frame(batches, "dose_rows"),
        "probe": _rows_frame(batches, "probe_rows"),
        "selected_edits": _rows_frame(batches, "selected_edit_rows"),
        "probe_screens": _rows_frame(batches, "probe_screen_rows"),
    }
    arrays: dict[str, NDArray] = {}
    bridge = _bridge_analysis(frames["bridge"], spec, arrays)
    state_reading = _state_reading_analysis(
        frames["natural"], frames["bridge"], spec, arrays
    )
    foresight, half_states = _foresight_analysis(
        frames["branch_states"], spec, arrays
    )
    frames["branch_halves"] = half_states
    dose = _dose_analysis(frames["dose"], spec, arrays)
    probe = _probe_analysis(frames["probe"], spec, arrays)
    metrics = {
        "format": f"{RESULT_FORMAT}-metrics",
        "phase": spec.label,
        "decision_status": (
            "pilot_estimation_only_awaiting_user_decision"
            if spec.label == "pilot"
            else "prospective_confirmation"
        ),
        "matrices": spec.matrices,
        "candidates_never_pooled": True,
        "bridge": bridge,
        "state_reading": state_reading,
        "foresight": foresight,
        "dose": dose,
        "probe": probe,
        "integrity": {
            "all_noop_plain_exact": bool(all(batch.no_op_plain_exact for batch in batches)),
            "batch_digests_unique": len({batch.scientific_digest for batch in batches})
            == len(batches),
            "raw_molecular_traces_persisted": False,
        },
        "confirmation_was_automatically_launched": False,
    }
    return metrics, frames, arrays


def _public_parity_fixture() -> NDArray[np.float64]:
    """Synthetic fixture independently compared with pinned public PhiRL."""

    rng = np.random.default_rng(20_260_818)
    observations = 600
    values = np.zeros((6, observations), dtype=np.float64)
    transition = np.asarray(
        [
            [0.62, 0.22, 0.05, 0.03, 0.02, 0.01],
            [0.18, 0.58, 0.06, 0.03, 0.02, 0.01],
            [0.06, 0.05, 0.63, 0.16, 0.03, 0.02],
            [0.02, 0.03, 0.18, 0.59, 0.07, 0.03],
            [0.03, 0.02, 0.04, 0.06, 0.61, 0.18],
            [0.02, 0.02, 0.03, 0.05, 0.19, 0.60],
        ],
        dtype=np.float64,
    )
    values[:, 0] = rng.normal(size=6)
    for index in range(1, observations):
        values[:, index] = (
            transition @ values[:, index - 1] + rng.normal(scale=0.45, size=6)
        )
    return (values - values.mean(axis=1, keepdims=True)) / values.std(
        axis=1, keepdims=True
    )


def validation_checks() -> dict[str, bool]:
    from .phir_instruments import (
        ALL_ATOMS,
        PHIR_ATOMS,
        close_clr_drop_last,
        fiedler_bipartition,
        lagged_gaussian_mi_graph,
        revised_phi_from_partition,
        typeset_whole_minus_parts,
    )

    checks: dict[str, bool] = {}
    counts = np.asarray(
        [
            [2, 0, 5, 1, 0, 3, 4],
            [3, 1, 4, 0, 2, 2, 5],
            [5, 2, 3, 1, 1, 0, 4],
            [4, 1, 2, 3, 0, 2, 6],
        ],
        dtype=np.int64,
    )
    clr = close_clr_drop_last(counts)
    checks["01_clr_shape_drop_last"] = clr.shape == (6, 4)
    replaced = counts.astype(np.float64) + 0.5
    logged = np.log(replaced / replaced.sum(axis=1, keepdims=True))
    expected_clr = (logged - logged.mean(axis=1, keepdims=True))[:, :-1].T
    checks["02_clr_zero_replacement_exact"] = bool(np.array_equal(clr, expected_clr))
    checks["03_all_sixteen_atoms_registered"] = len(ALL_ATOMS) == 16 and len(ATOM_NAMES) == 16
    checks["04_revised_phi_uses_nine_atoms"] = len(PHIR_ATOMS) == 9

    fixture = _public_parity_fixture()
    graph = lagged_gaussian_mi_graph(fixture)
    partition_a, partition_b = fiedler_bipartition(graph)
    revised, causation, emergence, synergy, atoms = revised_phi_from_partition(
        fixture, partition_a, partition_b
    )
    checks["05_public_fixture_partition"] = bool(
        set(map(int, partition_a)) == {4, 5}
        and set(map(int, partition_b)) == {0, 1, 2, 3}
    )
    checks["06_public_phirl_revised_parity"] = bool(
        abs(revised - 0.6944357302999425) < 1e-12
    )
    checks["07_public_fixture_atom_count"] = atoms.shape == (16,) and np.isfinite(atoms).all()
    checks["08_emergence_identity"] = bool(abs(emergence - (causation + synergy)) < 1e-12)
    typeset, whole = typeset_whole_minus_parts(fixture, partition_a, partition_b)
    checks["09_typeset_formula_is_unnormalized"] = bool(
        np.isfinite(typeset) and np.isfinite(whole) and abs(whole) > 1e-12
    )
    checks["10_text_artifact_is_separate_ratio"] = bool(
        abs((typeset / whole) - typeset) > 1e-8
    )
    prefix_first = revised_phi_from_partition(
        fixture[:, :400],
        *fiedler_bipartition(lagged_gaussian_mi_graph(fixture[:, :400])),
    )[0]
    prefix_second = revised_phi_from_partition(
        fixture[:, :400].copy(),
        *fiedler_bipartition(lagged_gaussian_mi_graph(fixture[:, :400].copy())),
    )[0]
    checks["11_prefix_fit_is_deterministic"] = bool(prefix_first == prefix_second)
    mask = np.zeros(fixture.shape[1] - 1, dtype=bool)
    mask[::2] = True
    masked_graph = lagged_gaussian_mi_graph(fixture, mask)
    checks["12_growth_pair_mask_is_honored"] = bool(
        masked_graph.shape == graph.shape and not np.array_equal(masked_graph, graph)
    )

    trace_records: list[bool] = []
    trace_rngs: list[bool] = []
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(44_001))
    initial = generate_initial_composition(config, np.random.default_rng(44_002))
    for candidate_index, candidate in enumerate(CANDIDATES):
        for seed in (71, 72, 73):
            traced_rng = np.random.default_rng(seed + candidate_index * 1_000)
            plain_rng = np.random.default_rng(seed + candidate_index * 1_000)
            traced = advance_fission_traced(
                initial, beta, config, CANDIDATES[candidate], traced_rng
            )
            plain = advance_fission(
                initial, beta, config, CANDIDATES[candidate], plain_rng
            )
            trace_records.append(records_equal(traced.record, plain))
            trace_rngs.append(
                rng_states_equal(traced_rng.bit_generator.state, plain_rng.bit_generator.state)
            )
    checks["13_candidate02_trace_record_parity"] = all(trace_records[:3])
    checks["14_candidate03_trace_record_parity"] = all(trace_records[3:])
    checks["15_trace_rng_state_parity"] = all(trace_rngs)
    checks["16_growth_trace_has_no_extra_rng_draws"] = all(trace_rngs) and all(trace_records)

    composition = np.asarray([2, 0, 1, 0], dtype=np.int64)
    legal = enumerate_legal_edits(composition)
    checks["17_legal_edit_enumeration_exact"] = len(legal) == 6 and all(
        composition[item.remove_type] > 0 and item.remove_type != item.add_type
        for item in legal
    )
    checks["18_legal_edit_preserves_mass"] = all(
        int(apply_molecular_edit(composition, item).sum()) == int(composition.sum())
        for item in legal
    )
    snapshot = Snapshot(composition, 7, (True, False, True), (0.95, 0.8, 0.96), 11, 123)
    edited = edited_snapshot(snapshot, legal[0])
    checks["19_instant_edit_preserves_history"] = bool(
        edited.generation == snapshot.generation
        and edited.inheritance == snapshot.inheritance
        and edited.boundary_h == snapshot.boundary_h
        and edited.previous_growth_steps == snapshot.previous_growth_steps
        and edited.cumulative_growth_steps == snapshot.cumulative_growth_steps
    )
    checks["20_frozen_model_hash"] = sha256_file(FROZEN_MODEL) == EXPECTED_MODEL_SHA256
    predictor_a = FrozenFullPredictor.load(FROZEN_MODEL)
    temporary_model = Path("/tmp/codex_ch5_model_roundtrip.npz")
    shutil.copy2(FROZEN_MODEL, temporary_model)
    predictor_b = FrozenFullPredictor.load(temporary_model)
    model_snapshot = Snapshot(initial, 20, tuple([True] * 20), tuple([0.95] * 20), 10, 200)
    prediction_a = predictor_a.predict_snapshot("02", model_snapshot, beta, config)
    prediction_b = predictor_b.predict_snapshot("02", model_snapshot, beta, config)
    checks["21_frozen_predictor_serialization_exact"] = bool(prediction_a == prediction_b)
    checks["22_seed_domains_unique"] = len(set(SEED_DOMAINS.values())) == len(SEED_DOMAINS)
    checks["23_pilot_confirmation_seed_firewall"] = (
        SEED_DOMAINS["pilot_matrix"] != SEED_DOMAINS["confirmation_matrix"]
        and _phase_seed_domain("pilot") != _phase_seed_domain("confirmation")
    )
    checks["24_pilot_has_24_matrices"] = scientific_spec("pilot").matrices == 24
    checks["25_confirmation_has_48_matrices"] = scientific_spec("confirmation").matrices == 48
    checks["26_both_scientific_specs_have_two_replicates"] = all(
        scientific_spec(phase).replicates == 2 for phase in ("pilot", "confirmation")
    )
    checks["27_fixed_five_landmarks"] = scientific_spec("pilot").landmarks == LANDMARKS
    checks["28_fixed_64_f12_branches"] = scientific_spec("pilot").branches == 64
    checks["29_manual_confirmation_barrier_registered"] = bool(
        protocol()["manual_confirmation_barrier"]["automatic_launch_forbidden"]
    )
    checks["30_raw_molecular_trace_persistence_forbidden"] = (
        protocol()["storage"].startswith("rolling molecular buffers")
    )
    checks["31_matrix_is_inference_unit"] = protocol()["inference"]["unit"] == "whole catalytic matrix"
    checks["32_bootstrap_draws_fixed"] = BOOTSTRAP_REPETITIONS == 4096
    checks["33_randomization_draws_fixed"] = RANDOMIZATION_REPETITIONS == 4096
    checks["34_all_sealed_source_files_exist"] = all((ROOT / name).is_file() for name in SOURCE_FILES)
    if len(checks) != 34:
        raise AssertionError(f"validation suite has {len(checks)} checks, expected 34")
    return checks


def run_validation(output: Path = DEFAULT_VALIDATION) -> dict[str, Any]:
    checks = validation_checks()
    payload = {
        "format": "codex-ch5-phir-validation-v1",
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": bool(all(checks.values())),
        "source_hashes": _source_hashes(),
        "frozen_model_sha256": sha256_file(FROZEN_MODEL),
        "numeric_environment": _runtime_versions(),
        "public_parity": {
            "source": "pigozzif/PhiRL",
            "commit": "a6d1d0d18c7551302724b7158c6ccdc4d3a33373",
            "synthetic_fixture_expected_revised_phi_r": 0.6944357302999425,
            "maximum_tolerance": 1e-12,
            "external_source_not_vendored": True,
        },
        "scientific_matrices_generated": 0,
    }
    if not payload["all_checks_passed"]:
        failures = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Chapter 5 validation failed: {failures}")
    with _atomic_destination(output) as destination:
        _atomic_json(destination / "validation.json", payload)
        write_checksums(destination)
    verify_checksums(output)
    print(f"Chapter 5 validation passed: {len(checks)}/34", flush=True)
    return payload


def _append_ledger(marker: str, lines: Sequence[str]) -> None:
    path = ROOT / LEDGER
    current = path.read_text(encoding="utf-8").rstrip() + "\n"
    if marker in current:
        return
    path.write_text(current + "\n" + marker + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def register_program(
    validation_directory: Path = DEFAULT_VALIDATION,
    output: Path = DEFAULT_REGISTRATION,
) -> dict[str, Any]:
    verify_checksums(validation_directory)
    validation = json.loads((validation_directory / "validation.json").read_text(encoding="utf-8"))
    if not validation["all_checks_passed"]:
        raise ValueError("Chapter 5 validation has not passed")
    if validation["source_hashes"] != _source_hashes():
        raise ValueError("Chapter 5 source changed after validation")
    if sha256_file(FROZEN_MODEL) != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen 5x predictor hash differs from its registered value")
    for forbidden in (
        DEFAULT_REGISTRATION,
        DEFAULT_SMOKE,
        DEFAULT_PILOT,
        DEFAULT_CONFIRMATION,
        DEFAULT_PILOT_WORK,
        DEFAULT_CONFIRMATION_WORK,
        AUTHORIZATION,
    ):
        if forbidden.exists():
            raise FileExistsError(f"pre-registration artifact already exists: {forbidden}")
    body: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol": protocol(),
        "protocol_id": protocol()["protocol_id"],
        "procedural_amendment": {
            "document": AMENDMENT_DOCUMENT,
            "document_sha256": sha256_file(ROOT / AMENDMENT_DOCUMENT),
            "predecessor_registration_id": "fedbe1184b1b202411e511725efb0a086f305c5d5575614c61d4cc790b803899",
            "scientific_outputs_before_repair": 0,
            "scientific_protocol_changed": False,
            "seeds_changed": False,
        },
        "source_hashes": _source_hashes(),
        "source_tree_sha256": _canonical_digest(_source_hashes()),
        "seed_registry": SEED_DOMAINS,
        "frozen_model_sha256": EXPECTED_MODEL_SHA256,
        "numeric_environment": _runtime_versions(),
        "validation_checksum_manifest_sha256": sha256_file(
            validation_directory / "SHA256SUMS"
        ),
        "pilot_matrices_at_registration": 0,
        "confirmation_matrices_at_registration": 0,
        "fable_code_or_artifacts_imported": False,
    }
    body["registration_id"] = _canonical_digest(_json_ready(body))
    with _atomic_destination(output) as destination:
        shutil.copy2(ROOT / DOCUMENT, destination / "preregistration.md")
        shutil.copy2(ROOT / AMENDMENT_DOCUMENT, destination / "procedural_amendment_001.md")
        shutil.copy2(validation_directory / "validation.json", destination / "validation.json")
        shutil.copy2(FROZEN_MODEL, destination / "frozen_full_predictor.npz")
        _atomic_json(destination / "protocol.json", protocol())
        _atomic_json(destination / "seed_registry.json", SEED_DOMAINS)
        _atomic_json(destination / "registration.json", body)
        _atomic_json(
            destination / "public_sources.json",
            {
                "PhiRL": {
                    "repository": "https://github.com/pigozzif/PhiRL",
                    "commit": "a6d1d0d18c7551302724b7158c6ccdc4d3a33373",
                    "used_for": "independent semantic audit and synthetic parity fixture",
                },
                "preprint": {
                    "identifier": "arXiv:2607.28250v1",
                    "used_for": "visual verification of the typeset whole-minus-parts equation",
                },
                "Fable": "No code, data, models, states, seeds, or result objects imported.",
            },
        )
        write_checksums(destination)
    verify_checksums(output)
    if output.resolve() == DEFAULT_REGISTRATION.resolve():
        _append_ledger(
            f"<!-- ch5-registration-{body['registration_id']} -->",
            (
                "## Chapter 5 Φ-r program registered",
                "",
                f"- Registration: `{body['registration_id']}`.",
                "- A 24-matrix pilot and disjoint 48-matrix prospective confirmation were sealed together.",
                "- The confirmation has a manual authorization barrier and cannot auto-launch after the pilot.",
                "- No scientific Chapter 5 matrix existed at registration.",
            ),
        )
    print(f"Chapter 5 registered: {body['registration_id']}", flush=True)
    return body


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    verify_checksums(directory)
    registration = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    if registration["format"] != REGISTRATION_FORMAT:
        raise ValueError("unsupported Chapter 5 registration format")
    body = dict(registration)
    observed = body.pop("registration_id")
    if _canonical_digest(_json_ready(body)) != observed:
        raise ValueError("Chapter 5 registration ID changed")
    if registration["source_hashes"] != _source_hashes():
        raise ValueError("Chapter 5 registered source tree changed")
    if registration["protocol"] != _json_ready(protocol()) or registration["seed_registry"] != SEED_DOMAINS:
        raise ValueError("Chapter 5 protocol or seed registry changed")
    if sha256_file(directory / "frozen_full_predictor.npz") != EXPECTED_MODEL_SHA256:
        raise ValueError("Chapter 5 frozen predictor copy changed")
    return registration


def run_smoke(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> dict[str, Any]:
    registration = verify_registration(registration_directory)
    spec = smoke_spec()
    first = [
        _run_matrix((matrix_id, spec, str(registration_directory / "frozen_full_predictor.npz")))
        for matrix_id in range(spec.matrices)
    ]
    second = [
        _run_matrix((matrix_id, spec, str(registration_directory / "frozen_full_predictor.npz")))
        for matrix_id in range(spec.matrices)
    ]
    smoke_metrics, smoke_frames, smoke_arrays = analyze_batches(first, spec)
    payload = {
        "format": "codex-ch5-phir-smoke-v1",
        "registration_id": registration["registration_id"],
        "artificial_non_scientific_fixture": True,
        "exact_replay": [item.scientific_digest for item in first]
        == [item.scientific_digest for item in second],
        "noop_plain_bitwise_exact": all(
            item.no_op_plain_exact for item in (*first, *second)
        ),
        "all_io_campaigns_exercised": bool(
            all(
                item.natural_rows
                and item.branch_rows
                and item.bridge_rows
                and item.dose_rows
                and item.probe_rows
                and item.probe_screen_rows
                for item in first
            )
        ),
        "analysis_and_inference_paths_exercised": bool(
            smoke_metrics and smoke_frames and smoke_arrays
        ),
        "effect_sizes_arm_order_event_rates_and_candidate_differences_disclosed": False,
        "scientific_matrices_generated": 0,
    }
    if not all(
        payload[key]
        for key in (
            "exact_replay",
            "noop_plain_bitwise_exact",
            "all_io_campaigns_exercised",
            "analysis_and_inference_paths_exercised",
        )
    ):
        raise AssertionError(f"Chapter 5 smoke failed: {payload}")
    with _atomic_destination(output) as destination:
        _atomic_json(destination / "smoke.json", payload)
        write_checksums(destination)
    verify_checksums(output)
    print("Chapter 5 non-scientific smoke passed", flush=True)
    return payload


def _format_effect(item: dict[str, Any]) -> str:
    return (
        f"{item['effect']:+.4f} "
        f"[{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}]"
    )


def _matching_cells(
    cells: Sequence[dict[str, Any]], **criteria: object
) -> list[dict[str, Any]]:
    return [
        item
        for item in cells
        if all(item.get(name) == value for name, value in criteria.items())
    ]


def _reports(metrics: dict[str, Any], registration_id: str) -> tuple[str, str]:
    phase = metrics["phase"]
    stage_note = (
        "This is the pre-specified 24-matrix pilot. It estimates directions and feasibility; "
        "it is not the independent 48-matrix confirmation."
        if phase == "pilot"
        else "This is the disjoint, prospectively sealed 48-matrix confirmation."
    )
    bridge_cells = metrics["bridge"]["cells"]
    bridge_lines: list[str] = []
    for contrast, metric in (
        ("model", "inherited"),
        ("model", "molecular_revised"),
        ("model", "molecular_typeset"),
        ("model", "generational_revised"),
        ("rule", "molecular_revised"),
    ):
        for item in _matching_cells(bridge_cells, contrast=contrast, metric=metric):
            bridge_lines.append(
                f"| {contrast} | {metric} | {item['candidate']} | {item['replicate']} | "
                f"{_format_effect(item)} | {item.get('holm_adjusted_p', float('nan')):.4g} |"
            )
    foresight_lines: list[str] = []
    for predictor in ("frozen_prediction", "molecular_revised", "generational_revised"):
        for item in _matching_cells(
            metrics["foresight"]["correlations"], predictor=predictor
        ):
            foresight_lines.append(
                f"| {predictor} | {item['candidate']} | {item['half']} | "
                f"{item['centered_spearman']:+.3f} "
                f"[{item['ci95'][0]:+.3f}, {item['ci95'][1]:+.3f}] |"
            )
    probe_lines: list[str] = []
    for metric in ("molecular_revised", "inherited_fraction", "joint_break_run3"):
        for item in _matching_cells(
            metrics["probe"]["results"], contrast="phi_extremes", metric=metric
        ):
            probe_lines.append(
                f"| {metric} | {item['candidate']} | {item['replicate']} | "
                f"{_format_effect(item)} |"
            )
    technical = "\n".join(
        [
            f"# Chapter 5 Φ-r / plastic-heredity {phase} report",
            "",
            stage_note,
            "",
            f"Registration: `{registration_id}`. Candidates were analyzed separately and the catalytic matrix was the inference unit.",
            "",
            "## Six-arm causal bridge",
            "",
            "Contrasts are stabilizing minus destabilizing for model/rule arms. The table uses the final 30 fissions.",
            "",
            "| Contrast | Reading | Candidate | Replicate | Mean [95% matrix CI] | Holm p |",
            "| --- | --- | --- | ---: | ---: | ---: |",
            *bridge_lines,
            "",
            f"- Frozen-control heredity validity gate: **{metrics['bridge']['model_heredity_validity_gate']}**.",
            f"- Revised Φ-r response gate: **{metrics['bridge']['model_revised_response_gate']}**.",
            "",
            "## Hereditary-state reading",
            "",
            "The state contrast is within-lineage: readings during a trailing inherited run of at least five fissions minus all other states.",
            "",
            f"- Revised Φ-r state-reading gate: **{metrics['state_reading']['bridge_revised_reads_state_gate']}**.",
            "",
            "## Prospective foresight",
            "",
            "All correlations are centered within catalytic matrix and kept separate by candidate and fixed branch half.",
            "",
            "| Predictor/reading | Candidate | Half | Centered Spearman [95% matrix CI] |",
            "| --- | --- | --- | ---: |",
            *foresight_lines,
            "",
            f"- Frozen-predictor validity gate: **{metrics['foresight']['frozen_predictor_validity_gate']}**.",
            f"- Revised Φ-r ±0.10 correlation-equivalence gate: **{metrics['foresight']['revised_phi_centered_equivalence_gate']}**.",
            f"- Incremental Φ log-loss ±0.005 equivalence gate: **{metrics['foresight']['incremental_phi_logloss_equivalence_gate']}**.",
            "",
            "## Bounded Φ-directed probe",
            "",
            "The selector screened a fixed 64-edit set with four short common-random-stream probes, then confirmed the selected extremes on fresh streams.",
            "",
            "| Outcome | Candidate | Replicate | Φ-up minus Φ-down [95% matrix CI] |",
            "| --- | --- | ---: | ---: |",
            *probe_lines,
            "",
            f"- Probe moves revised Φ-r gate: **{metrics['probe']['probe_moves_revised_phi_gate']}**.",
            f"- Probe heredity-equivalence gate: **{metrics['probe']['probe_heredity_equivalence_gate']}**.",
            "",
            "## Dose response and instrument separation",
            "",
            f"- Revised Φ-r dose gate: **{metrics['dose']['revised_phi_dose_gate']}**.",
            "- The unnormalized typeset equation, the text-extraction ratio, revised Φ-r, all 16 atoms, causation, emergence, synergy-persistence, molecular and generational clocks, and a growth-only sensitivity were retained as distinct readings.",
            "",
            "## Integrity and claim boundary",
            "",
            f"- No-op traced callback exactly matched the plain simulator: **{metrics['integrity']['all_noop_plain_exact']}**.",
            "- Complete generation and replay results are compared matrix by matrix in `replay_audit.json`.",
            "- No raw molecular trace was saved; only rolling-window scores and compact state/outcome records were retained.",
            "- A Φ-r response is an information-statistical gauge response. It is not evidence of consciousness, life, agency, biological memory, a universal origin-of-life mechanism, or a portal to a Platonic space.",
            "- The public PhiRL code belongs to a companion RL paper; this program does not validate the unavailable private GARD-paper pipeline.",
            "",
            "## Phase decision",
            "",
            (
                "The mandatory next step is human review of this pilot. The software will not create or launch the 48-matrix confirmation without a separate authorization artifact."
                if phase == "pilot"
                else "This confirmation stands alone; it was not pooled with the pilot."
            ),
            "",
        ]
    )
    revised = _matching_cells(
        bridge_cells, contrast="model", metric="molecular_revised"
    )
    heredity = _matching_cells(bridge_cells, contrast="model", metric="inherited")
    revised_text = ", ".join(
        f"candidate {item['candidate']} replicate {item['replicate']}: {_format_effect(item)}"
        for item in revised
    )
    heredity_text = ", ".join(
        f"candidate {item['candidate']} replicate {item['replicate']}: {_format_effect(item)}"
        for item in heredity
    )
    lay = "\n".join(
        [
            f"# Lay summary — Chapter 5 {phase}",
            "",
            stage_note,
            "",
            "We asked whether two mathematical ‘thermometers’ of information integration track the plastic-heredity behavior already established in the Codex GARD reconstruction. One thermometer is the formula printed on the paper page; the other is the revised Φ-r calculation independently reconstructed from the public PhiRL code. They are not assumed to be interchangeable.",
            "",
            f"As a reality check, the already frozen controller changed ordinary hereditary stability as follows: {heredity_text or 'not estimable'}. The revised Φ-r reading changed as follows: {revised_text or 'not estimable'}. The full report separately tests whether that reading marks a currently stable hereditary episode, predicts a future break-and-renewal event, changes smoothly with intervention strength, or can itself be pushed without changing heredity.",
            "",
            "The key distinction is between a gauge and a cause. A dashboard needle can move when the engine changes, yet moving the needle need not control the engine. This experiment was designed to detect exactly that distinction.",
            "",
            (
                "Nothing will automatically continue. We must inspect these 24-matrix estimates before deciding whether to authorize the sealed 48-matrix confirmation."
                if phase == "pilot"
                else "These 48 matrices were generated from a disjoint seed domain after explicit pilot review, so they provide the prospective confirmation rather than a larger reanalysis of the pilot."
            ),
            "",
        ]
    )
    return technical, lay


def _jsonify_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if output[column].map(lambda value: isinstance(value, (list, tuple, dict))).any():
            output[column] = output[column].map(
                lambda value: json.dumps(_json_ready(value), separators=(",", ":"))
                if isinstance(value, (list, tuple, dict))
                else value
            )
    return output


def _replay_audit(
    generated: Sequence[CampaignBatch], replayed: Sequence[CampaignBatch]
) -> dict[str, Any]:
    if len(generated) != len(replayed):
        raise ValueError("Chapter 5 replay matrix count differs")
    rows = []
    for left, right in zip(generated, replayed, strict=True):
        rows.append(
            {
                "matrix_id": left.matrix_id,
                "generated_digest": left.scientific_digest,
                "replay_digest": right.scientific_digest,
                "exact": left.scientific_digest == right.scientific_digest,
                "noop_plain_exact": left.no_op_plain_exact and right.no_op_plain_exact,
            }
        )
    return {
        "format": "codex-ch5-phir-replay-audit-v1",
        "matrices": rows,
        "complete_exact_replay": bool(all(row["exact"] for row in rows)),
        "complete_noop_plain_exact": bool(all(row["noop_plain_exact"] for row in rows)),
    }


def _claim_boundaries(metrics: dict[str, Any]) -> dict[str, Any]:
    phase = metrics["phase"]
    confirmed = phase == "confirmation"
    supported: list[str] = []
    if confirmed and metrics["bridge"]["model_revised_response_gate"]:
        supported.append(
            "the revised Phi-r gauge responds prospectively to frozen interventions that change hereditary stability in both Codex candidates"
        )
    if confirmed and metrics["state_reading"]["bridge_revised_reads_state_gate"]:
        supported.append(
            "revised Phi-r distinguishes short-run hereditary states within controlled Codex lineages"
        )
    if confirmed and metrics["foresight"]["revised_phi_centered_equivalence_gate"]:
        supported.append(
            "revised Phi-r is equivalent to a small centered foresight association under the registered margin"
        )
    failed: list[str] = []
    if confirmed:
        gate_paths = {
            "revised Phi-r bridge response": metrics["bridge"]["model_revised_response_gate"],
            "revised Phi-r hereditary-state reading": metrics["state_reading"]["bridge_revised_reads_state_gate"],
            "revised Phi-r dose response": metrics["dose"]["revised_phi_dose_gate"],
            "bounded probe gauge movement": metrics["probe"]["probe_moves_revised_phi_gate"],
        }
        failed = [name for name, passed in gate_paths.items() if not passed]
    return {
        "phase": phase,
        "pilot_is_not_confirmation": phase == "pilot",
        "supported_claims": supported,
        "failed_registered_predictions": failed,
        "unresolved_questions": [
            "whether either Phi-r formulation generalizes beyond these simulator contracts",
            "whether a gauge relationship has any substrate-independent meaning",
            "whether the typeset and revised formulations can be reconciled theoretically",
        ],
        "prohibited_interpretations": protocol()["claim_boundary"],
    }


def _write_result(
    output: Path,
    registration: dict[str, Any],
    spec: RunSpec,
    batches: Sequence[CampaignBatch],
    replay: dict[str, Any],
    metrics: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    arrays: dict[str, NDArray],
) -> None:
    technical, lay = _reports(metrics, registration["registration_id"])
    claims = _claim_boundaries(metrics)
    with _atomic_destination(output) as destination:
        metrics_text = json.dumps(_json_ready(metrics), sort_keys=True, indent=2, allow_nan=True) + "\n"
        (destination / "primary_metrics.json").write_text(metrics_text, encoding="utf-8")
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical, encoding="utf-8")
        (destination / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        _atomic_json(destination / "claim_boundaries.json", claims)
        _atomic_json(destination / "replay_audit.json", replay)
        row_counts: dict[str, int] = {}
        for name, frame in frames.items():
            table = _jsonify_table(frame)
            path = destination / f"{name}.csv.gz"
            table.to_csv(path, index=False, compression="gzip")
            row_counts[name] = int(len(table))
        np.savez_compressed(destination / "inference_arrays.npz", **arrays)
        np.savez_compressed(
            destination / "batch_digests.npz",
            matrix_id=np.asarray([batch.matrix_id for batch in batches], dtype=np.int16),
            scientific_digest=np.asarray([batch.scientific_digest for batch in batches]),
        )
        np.savez_compressed(
            destination / "matrix_inputs.npz",
            beta=np.stack([batch.beta for batch in batches]),
            initial_composition=np.stack(
                [batch.initial_composition for batch in batches]
            ),
            matrix_id=np.asarray([batch.matrix_id for batch in batches], dtype=np.int16),
        )
        readback_counts = {
            name: int(len(pd.read_csv(destination / f"{name}.csv.gz")))
            for name in frames
        }
        with np.load(destination / "batch_digests.npz", allow_pickle=False) as archive:
            digest_rows = int(archive["matrix_id"].size)
        with np.load(destination / "matrix_inputs.npz", allow_pickle=False) as archive:
            input_shapes_exact = bool(
                archive["beta"].shape
                == (spec.matrices, GardConfig().n_types, GardConfig().n_types)
                and archive["initial_composition"].shape
                == (spec.matrices, GardConfig().n_types)
            )
        readback = {
            "primary_metrics_text_exact": (destination / "primary_metrics.json").read_text(encoding="utf-8") == metrics_text,
            "all_table_row_counts_exact": readback_counts == row_counts,
            "batch_digest_count_exact": digest_rows == spec.matrices,
            "matrix_input_shapes_exact": input_shapes_exact,
            "replay_exact": replay["complete_exact_replay"],
            "noop_plain_exact": replay["complete_noop_plain_exact"],
        }
        readback["complete_readback_exact"] = bool(all(readback.values()))
        if not readback["complete_readback_exact"]:
            raise AssertionError(f"Chapter 5 readback failed: {readback}")
        _atomic_json(destination / "readback_audit.json", readback)
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "phase": spec.label,
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "replicates": spec.replicates,
            "row_counts": row_counts,
            "complete_exact_replay": replay["complete_exact_replay"],
            "noop_plain_bitwise_exact": replay["complete_noop_plain_exact"],
            "complete_readback_exact": True,
            "pilot_and_confirmation_pooled": False,
            "confirmation_automatically_launched": False,
            "mandatory_stop_after_pilot": spec.label == "pilot",
            "confirmation_authorization_required": True,
            "raw_molecular_traces_persisted": False,
            "runtime": {
                **_runtime_versions(),
                "cpu_count": os.cpu_count(),
            },
        }
        _atomic_json(destination / "manifest.json", manifest)
        write_checksums(destination)
    verify_checksums(output)


def _prepare_work(
    work: Path,
    output: Path,
    registration_id: str,
    spec: RunSpec,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed result {output}")
    free = shutil.disk_usage(ROOT).free
    if free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError(
            f"Chapter 5 requires at least {MINIMUM_FREE_DISK_BYTES:,} free bytes; found {free:,}"
        )
    work.mkdir(parents=True, exist_ok=True)
    expected = {
        "format": "codex-ch5-phir-work-v1",
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
        "spec": asdict(spec),
    }
    path = work / "campaign_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(_json_ready(expected))
        ):
            raise ValueError("Chapter 5 work directory belongs to another contract")
    else:
        _atomic_json(path, expected)


def verify_result(output: Path) -> dict[str, Any]:
    verify_checksums(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != RESULT_FORMAT:
        raise ValueError("unsupported Chapter 5 result format")
    registration = verify_registration()
    if manifest["registration_id"] != registration["registration_id"]:
        raise ValueError("Chapter 5 result belongs to another registration")
    if not all(
        manifest[key]
        for key in (
            "complete_exact_replay",
            "noop_plain_bitwise_exact",
            "complete_readback_exact",
        )
    ):
        raise ValueError("Chapter 5 result integrity gate failed")
    return manifest


def run_scientific_phase(
    phase: str,
    *,
    workers: int = min(os.cpu_count() or 1, 12),
) -> dict[str, Any]:
    if phase not in {"pilot", "confirmation"}:
        raise ValueError("scientific phase must be pilot or confirmation")
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    smoke_payload = json.loads((DEFAULT_SMOKE / "smoke.json").read_text(encoding="utf-8"))
    if not smoke_payload["exact_replay"] or not smoke_payload["noop_plain_bitwise_exact"]:
        raise ValueError("Chapter 5 smoke did not pass")
    if phase == "confirmation":
        verify_result(DEFAULT_PILOT)
        if not AUTHORIZATION.is_file():
            raise PermissionError(
                "48-matrix confirmation is locked until explicit pilot-review authorization"
            )
        authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
        if authorization.get("registration_id") != registration["registration_id"]:
            raise ValueError("confirmation authorization belongs to another registration")
        if authorization.get("acknowledgement") != "PILOT_REVIEWED_AUTHORIZE_48":
            raise ValueError("confirmation authorization acknowledgement is invalid")
    spec = scientific_spec(phase)
    output = DEFAULT_PILOT if phase == "pilot" else DEFAULT_CONFIRMATION
    work = DEFAULT_PILOT_WORK if phase == "pilot" else DEFAULT_CONFIRMATION_WORK
    _prepare_work(work, output, registration["registration_id"], spec)
    try:
        generated = _run_checkpointed(
            spec,
            registration["registration_id"],
            work / "generated",
            work,
            "generated",
            workers,
        )
        replayed = _run_checkpointed(
            spec,
            registration["registration_id"],
            work / "replay",
            work,
            "replay",
            workers,
        )
        replay = _replay_audit(generated, replayed)
        if not replay["complete_exact_replay"] or not replay["complete_noop_plain_exact"]:
            raise AssertionError("Chapter 5 complete replay gate failed")
        _write_status(work, "analysis", 0, 1)
        metrics, frames, arrays = analyze_batches(generated, spec)
        _write_result(
            output,
            registration,
            spec,
            generated,
            replay,
            metrics,
            frames,
            arrays,
        )
        _write_status(
            work,
            "awaiting_user_review" if phase == "pilot" else "complete",
            1,
            1,
            output=str(output),
        )
    except BaseException as error:
        _write_status(
            work,
            "failed",
            0,
            1,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    _append_ledger(
        f"<!-- ch5-{phase}-{sha256_file(output / 'manifest.json')} -->",
        (
            f"## Chapter 5 {phase} completed",
            "",
            f"- Result: `{output.relative_to(ROOT)}`.",
            f"- Matrices: {spec.matrices}; candidates separate; two replicates; complete replay passed.",
            f"- Decision status: {metrics['decision_status']}.",
            f"- Model revised-Φ response gate: {metrics['bridge']['model_revised_response_gate']}.",
            f"- State-reading gate: {metrics['state_reading']['bridge_revised_reads_state_gate']}.",
            f"- Frozen-predictor foresight validity: {metrics['foresight']['frozen_predictor_validity_gate']}.",
            "- The pilot does not authorize or launch the confirmation." if phase == "pilot" else "- Pilot and confirmation were not pooled.",
        ),
    )
    return metrics


def authorize_confirmation(acknowledgement: str) -> dict[str, Any]:
    if acknowledgement != "PILOT_REVIEWED_AUTHORIZE_48":
        raise ValueError("exact acknowledgement PILOT_REVIEWED_AUTHORIZE_48 is required")
    registration = verify_registration()
    pilot = verify_result(DEFAULT_PILOT)
    if pilot["phase"] != "pilot" or pilot["matrices"] != PILOT_MATRICES:
        raise ValueError("the completed result is not the registered 24-matrix pilot")
    if DEFAULT_CONFIRMATION.exists() or DEFAULT_CONFIRMATION_WORK.exists():
        raise FileExistsError("confirmation work or output already exists")
    payload = {
        "format": "codex-ch5-phir-confirmation-authorization-v1",
        "registration_id": registration["registration_id"],
        "pilot_manifest_sha256": sha256_file(DEFAULT_PILOT / "manifest.json"),
        "pilot_checksum_manifest_sha256": sha256_file(DEFAULT_PILOT / "SHA256SUMS"),
        "acknowledgement": acknowledgement,
        "created_at_unix": time.time(),
        "confirmation_not_launched_by_this_command": True,
    }
    if AUTHORIZATION.exists():
        raise FileExistsError(f"authorization already exists: {AUTHORIZATION}")
    _atomic_json(AUTHORIZATION, payload)
    print("48-matrix confirmation authorized but not launched", flush=True)
    return payload


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def launch_detached(phase: str, workers: int) -> dict[str, Any]:
    if phase not in {"pilot", "confirmation"}:
        raise ValueError("phase must be pilot or confirmation")
    verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if phase == "confirmation" and not AUTHORIZATION.is_file():
        raise PermissionError("confirmation has not been explicitly authorized")
    work = DEFAULT_PILOT_WORK if phase == "pilot" else DEFAULT_CONFIRMATION_WORK
    output = DEFAULT_PILOT if phase == "pilot" else DEFAULT_CONFIRMATION
    log_path = RESULTS / f"phir_ch5_{phase}.log"
    launch_path = work / "detached_launch.json"
    if output.exists():
        raise FileExistsError(f"completed output already exists: {output}")
    if launch_path.exists():
        existing = json.loads(launch_path.read_text(encoding="utf-8"))
        if _pid_alive(int(existing.get("pid", -1))):
            raise RuntimeError(f"{phase} is already running as PID {existing['pid']}")
    work.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "plastic_heredity.phir_ch5",
        f"run-{phase}",
        "--workers",
        str(workers),
    ]
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    payload = {
        "format": "codex-ch5-phir-detached-launch-v1",
        "phase": phase,
        "pid": process.pid,
        "workers": workers,
        "command": command,
        "log": str(log_path),
        "launched_at_unix": time.time(),
    }
    _atomic_json(launch_path, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def status_payload() -> dict[str, Any]:
    output: dict[str, Any] = {
        "format": "codex-ch5-phir-status-report-v1",
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "pilot_complete": DEFAULT_PILOT.exists(),
        "confirmation_authorized": AUTHORIZATION.exists(),
        "confirmation_complete": DEFAULT_CONFIRMATION.exists(),
        "automatic_confirmation_launch_forbidden": True,
    }
    for phase, work in (
        ("pilot", DEFAULT_PILOT_WORK),
        ("confirmation", DEFAULT_CONFIRMATION_WORK),
    ):
        phase_status: dict[str, Any] = {"work_exists": work.exists()}
        status_path = work / "campaign_status.json"
        launch_path = work / "detached_launch.json"
        if status_path.exists():
            phase_status["campaign"] = json.loads(status_path.read_text(encoding="utf-8"))
        if launch_path.exists():
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["process_alive"] = _pid_alive(int(launch.get("pid", -1)))
            phase_status["launch"] = launch
        output[phase] = phase_status
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("register")
    subparsers.add_parser("smoke")
    for name in ("run-pilot", "run-confirmation", "launch-pilot", "launch-confirmation"):
        local = subparsers.add_parser(name)
        local.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    authorize = subparsers.add_parser("authorize-confirmation")
    authorize.add_argument("--acknowledge", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("verify-pilot")
    subparsers.add_parser("verify-confirmation")
    arguments = parser.parse_args(argv)
    if arguments.command == "validate":
        run_validation()
    elif arguments.command == "register":
        register_program()
    elif arguments.command == "smoke":
        run_smoke()
    elif arguments.command == "run-pilot":
        run_scientific_phase("pilot", workers=arguments.workers)
    elif arguments.command == "run-confirmation":
        run_scientific_phase("confirmation", workers=arguments.workers)
    elif arguments.command == "launch-pilot":
        launch_detached("pilot", arguments.workers)
    elif arguments.command == "launch-confirmation":
        launch_detached("confirmation", arguments.workers)
    elif arguments.command == "authorize-confirmation":
        authorize_confirmation(arguments.acknowledge)
    elif arguments.command == "status":
        print(json.dumps(status_payload(), indent=2, sort_keys=True))
    elif arguments.command == "verify-pilot":
        print(json.dumps(verify_result(DEFAULT_PILOT), indent=2, sort_keys=True))
    elif arguments.command == "verify-confirmation":
        print(json.dumps(verify_result(DEFAULT_CONFIRMATION), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
