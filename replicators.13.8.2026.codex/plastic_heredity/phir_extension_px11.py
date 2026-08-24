"""PX11 causal information redistribution and sensor confirmation program.

PX11 is an additive, Codex-only successor to PX10.  It compares three causal
intervention families on identical naturally broken daughter states, freezes a
one-parameter molecular-dose channel, and stress-tests the already registered
PX10 temporal score under reduced observation.  The 24-matrix pilot and the
optional 48-matrix confirmation are disjoint and confirmation is manual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from . import intervention_cr2_dose_response as cr2
from . import intervention_cr5 as cr5
from . import intervention_cr6_transfer as cr6
from . import intervention_replication as intervention_base
from . import intervention_p3c as p3c
from . import phir_extension_px9 as px9
from . import phir_extension_px10 as px10
from .config import CANDIDATES, GardConfig
from .experiment import StateCase
from .intervention_core import BetaSurgery, MolecularEdit, apply_molecular_edit
from .intervention_outgoing_rule import (
    outgoing_catalytic_influence,
    select_outgoing_rule_edits,
)
from .mechanistic import sha256_file, verify_checksums, write_checksums
from .mechanistic_metrics import holm_adjust
from .phir_ch5 import _append_ledger
from .phir_instruments import ATOM_NAMES, advance_fission_traced
from .phir_rescue_instruments import beta_physical_partition
from .seeds import derive_seed
from .simulator import Snapshot, generate_beta, generate_initial_composition


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "phir_extension"
DOCUMENT = "CODEX_CH5_PHIR_PX11_PREREGISTRATION.md"

DEFAULT_VALIDATION = RESULT_ROOT / "px11_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "px11_registration"
DEFAULT_SMOKE = RESULT_ROOT / "px11_smoke"
DEFAULT_PILOT = RESULT_ROOT / "px11_causal_redistribution24"
DEFAULT_CONFIRMATION_REGISTRATION = RESULT_ROOT / "px11_confirmation_registration"
DEFAULT_CONFIRMATION = RESULT_ROOT / "px11_causal_redistribution48"
DEFAULT_TRANSPORT = RESULT_ROOT / "px11_cr6_sensor_transport"
DEFAULT_PILOT_LOG = RESULT_ROOT / "px11_causal_redistribution24.log"
DEFAULT_CONFIRMATION_LOG = RESULT_ROOT / "px11_causal_redistribution48.log"

EXTERNAL_BASE = Path(
    "/mnt/bioIce1/PlasticHeredityArchivedWorkfiles/"
    "replicators.13.8.2026.codex"
)
DEFAULT_PILOT_WORK = EXTERNAL_BASE / "px11_causal_redistribution24_work"
DEFAULT_CONFIRMATION_WORK = EXTERNAL_BASE / "px11_causal_redistribution48_work"
DEFAULT_TRANSPORT_WORK = EXTERNAL_BASE / "px11_cr6_sensor_transport_work"
DEFAULT_TRANSPORT_LOG = RESULT_ROOT / "px11_cr6_sensor_transport.log"
CR6_ROOT = ROOT / "results_intervention_replication" / "cr6_zero_shot_transfer"

MODEL_SOURCE = px9.MODEL_SOURCE
MODEL_CONTRACT_SOURCE = px9.MODEL_CONTRACT_SOURCE
PX10_REGISTRATION = px10.DEFAULT_REGISTRATION
PX10_CALIBRATION = px10.DEFAULT_CALIBRATION

PROGRAM_FORMAT = "codex-ch5-phir-px11-causal-redistribution-v1"
REGISTRATION_FORMAT = "codex-ch5-phir-px11-registration-v1"
CONFIRMATION_REGISTRATION_FORMAT = (
    "codex-ch5-phir-px11-confirmation-registration-v1"
)
RESULT_FORMAT = "codex-ch5-phir-px11-result-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-px11-checkpoint-v1"
STATUS_FORMAT = "codex-ch5-phir-px11-status-v1"
LABEL = "CODEX_CH5_PHIR_PX11_CAUSAL_REDISTRIBUTION_V1"

PILOT_MATRICES = 24
CONFIRMATION_MATRICES = 48
LANDMARKS = (20, 35, 50, 65, 80)
BRANCHES = 128
HORIZON = 8
ACQUISITION_LIMIT = 60
MINIMUM_ELIGIBLE_PILOT = 20
MINIMUM_ELIGIBLE_CONFIRMATION = 40
BOOTSTRAP_DRAWS = 4096
RANDOMIZATION_DRAWS = 4096
MAX_WORKERS = 8
MAX_TOTAL_CPU_HOURS = 30.0
DEFAULT_PILOT_CPU_HOURS = 12.0
DEFAULT_CONFIRMATION_CPU_HOURS = 18.0
MINIMUM_FREE_DISK_BYTES = 2_000_000_000
OUTCOME_EQUIVALENCE_MARGIN = 0.025
INFORMATION_EQUIVALENCE_MARGIN_BITS = 0.0005
SENSOR_RETENTION_FRACTION = 0.75

QUANTILES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
QUANTILE_ARMS = ("Q00", "Q20", "Q40", "Q60", "Q80", "Q100")
ARMS = (
    *QUANTILE_ARMS,
    "RANDOM_SWAP",
    "RULE_STABILIZE",
    "RULE_DESTABILIZE",
    "TIGHTEN",
    "LOOSEN",
    "BLOCK_RANDOM",
    "NOOP",
)
ATOM_ARMS = (
    "Q00",
    "Q100",
    "RULE_DESTABILIZE",
    "RULE_STABILIZE",
    "LOOSEN",
    "TIGHTEN",
)
CONTRASTS = {
    "model": ("Q100", "Q00"),
    "physical_rule": ("RULE_STABILIZE", "RULE_DESTABILIZE"),
    "beta_surgery": ("TIGHTEN", "LOOSEN"),
}
CONTROL_CONTRASTS = {
    "random_swap": ("RANDOM_SWAP", "NOOP"),
    "block_random": ("BLOCK_RANDOM", "NOOP"),
}
PRIMARY_LAGS = (1, 2, 4)
ALL_LAGS = (1, 2, 4, 8)
EARLY_LAGS = (1, 2)
LATE_LAGS = (4, 8)
TEMPORAL_SHIFTS = px10.TEMPORAL_SHIFTS

GROUP_COLUMNS = {
    "downward_routing": ("atom_s_to_u0", "atom_s_to_u1"),
    "upward_integration": ("atom_u0_to_s", "atom_u1_to_s"),
    "redundant_persistence": ("atom_r_to_r",),
    "synergy_persistence": ("atom_s_to_s",),
    "cross_part_transfer": ("atom_u0_to_u1", "atom_u1_to_u0"),
}

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_extension_px11.py",
    "tests/test_phir_extension_px11.py",
    "plastic_heredity/phir_extension_px10.py",
    "plastic_heredity/phir_extension_px9.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_rescue_instruments.py",
    "plastic_heredity/intervention_cr5.py",
    "plastic_heredity/intervention_cr6_transfer.py",
    "plastic_heredity/intervention_replication.py",
    "plastic_heredity/intervention_outgoing_rule.py",
    "plastic_heredity/intervention_p3c.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/config.py",
    "plastic_heredity/seeds.py",
    "pyproject.toml",
    "requirements-lock.txt",
)


def _seed_domain(name: str) -> str:
    return hashlib.sha256(f"{LABEL}::{name}".encode()).hexdigest()


SEED_DOMAINS = {
    name: _seed_domain(name)
    for name in (
        "matrix",
        "initial",
        "main_path",
        "acquisition",
        "random_action",
        "beta_control",
        "future",
        "observation_mask",
        "observation_noise",
        "bootstrap",
        "randomization",
        "channel_derangement",
        "replay",
        "validation",
        "smoke",
    )
}


def _json_ready(value: Any) -> Any:
    return px10._json_ready(value)


def _digest(value: Any) -> str:
    return px10._digest(value)


def _array_digest(*arrays: NDArray) -> str:
    return px10._array_digest(*arrays)


def _atomic_json(path: Path, value: Any) -> None:
    px10._atomic_json(path, value)


def _atomic_pickle(path: Path, value: Any) -> None:
    px10._atomic_pickle(path, value)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


@dataclass(frozen=True)
class PX11Spec:
    stage: str
    matrices: int
    landmarks: tuple[int, ...]
    branches: int
    horizon: int
    acquisition_limit: int
    bootstrap_draws: int
    randomization_draws: int
    sensor_profile: str = "PILOT_GRID"
    active_families: tuple[str, ...] = tuple(CONTRASTS)
    dose_channel_active: bool = True
    sensor_active: bool = True

    def as_px9(self) -> px9.PX9Spec:
        return px9.PX9Spec(
            self.stage,
            self.matrices,
            self.landmarks,
            self.branches,
            self.horizon,
            self.acquisition_limit,
            self.bootstrap_draws,
            self.randomization_draws,
        )


def pilot_spec() -> PX11Spec:
    return PX11Spec(
        "pilot",
        PILOT_MATRICES,
        LANDMARKS,
        BRANCHES,
        HORIZON,
        ACQUISITION_LIMIT,
        BOOTSTRAP_DRAWS,
        RANDOMIZATION_DRAWS,
    )


def confirmation_spec(contract: Mapping[str, Any]) -> PX11Spec:
    return PX11Spec(
        "confirmation",
        CONFIRMATION_MATRICES,
        LANDMARKS,
        BRANCHES,
        HORIZON,
        ACQUISITION_LIMIT,
        BOOTSTRAP_DRAWS,
        RANDOMIZATION_DRAWS,
        str(contract["selected_sensor_profile"]),
        tuple(str(value) for value in contract["advancing_families"]),
        bool(contract["dose_channel_active"]),
        bool(contract["sensor_active"]),
    )


def smoke_spec() -> PX11Spec:
    return PX11Spec("smoke", 1, (20,), 16, 4, 20, 32, 32)


def _halves(spec: PX11Spec) -> dict[str, tuple[int, ...]]:
    midpoint = spec.branches // 2
    return {
        "A": tuple(range(midpoint)),
        "B": tuple(range(midpoint, spec.branches)),
    }


def _seed(spec: PX11Spec, domain: str, *keys: object) -> int:
    selected = "smoke" if spec.stage == "smoke" else domain
    return derive_seed(SEED_DOMAINS[selected], LABEL, spec.stage, domain, *keys)


@contextmanager
def _px9_seed_context(spec: PX11Spec) -> Iterable[None]:
    """Use PX9's sealed simulator helpers with PX11-only seed domains."""

    old_label = px9.LABEL
    old_domains = px9.SEED_DOMAINS
    old_matrices = px9.MATRICES
    try:
        px9.LABEL = LABEL
        px9.SEED_DOMAINS = SEED_DOMAINS
        px9.MATRICES = spec.matrices
        yield
    finally:
        px9.LABEL = old_label
        px9.SEED_DOMAINS = old_domains
        px9.MATRICES = old_matrices


@dataclass(frozen=True)
class SensorProfile:
    name: str
    support: int
    coordinate_fraction: float = 1.0
    count_fraction: float = 1.0
    depth_stride: int = 1


SENSOR_PROFILES = (
    SensorProfile("FULL64", 64),
    SensorProfile("FULL32", 32),
    SensorProfile("FULL16", 16),
    SensorProfile("COORD75", 64, coordinate_fraction=0.75),
    SensorProfile("COORD50", 64, coordinate_fraction=0.50),
    SensorProfile("COORD25", 64, coordinate_fraction=0.25),
    SensorProfile("COUNT50", 64, count_fraction=0.50),
    SensorProfile("COUNT25", 64, count_fraction=0.25),
    SensorProfile("TIME2", 64, depth_stride=2),
    SensorProfile("TIME4", 64, depth_stride=4),
    SensorProfile(
        "COMPACT", 32, coordinate_fraction=0.50, count_fraction=0.50,
        depth_stride=2,
    ),
    SensorProfile(
        "MINIMAL", 16, coordinate_fraction=0.25, count_fraction=0.25,
        depth_stride=4,
    ),
)
SENSOR_BY_NAME = {item.name: item for item in SENSOR_PROFILES}
SENSOR_SELECTION_ORDER = (
    "MINIMAL",
    "COMPACT",
    "COORD25",
    "COUNT25",
    "TIME4",
    "COORD50",
    "COUNT50",
    "TIME2",
    "COORD75",
    "FULL16",
    "FULL32",
    "FULL64",
)


def protocol() -> dict[str, Any]:
    pilot = pilot_spec()
    confirmation = PX11Spec(
        "confirmation",
        CONFIRMATION_MATRICES,
        LANDMARKS,
        BRANCHES,
        HORIZON,
        ACQUISITION_LIMIT,
        BOOTSTRAP_DRAWS,
        RANDOMIZATION_DRAWS,
    )
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "question": "causal information redistribution and robust temporal sensing of post-break renewal",
        "predecessors_immutable": True,
        "public_phi_r_can_win": False,
        "cohorts": {
            "pilot": _json_ready(pilot.__dict__),
            "confirmation": _json_ready(confirmation.__dict__),
            "fresh_disjoint_seeded_matrices": True,
            "minimum_eligible": {
                "pilot": MINIMUM_ELIGIBLE_PILOT,
                "confirmation": MINIMUM_ELIGIBLE_CONFIRMATION,
            },
            "manual_stop_after_pilot": True,
            "automatic_confirmation": False,
        },
        "state": {
            "natural_landmarks": list(LANDMARKS),
            "acquire_first_break": "unrounded float64 H <= 0.9",
            "shared_exact_post_break_daughter": True,
            "renewal_endpoint": "run3 within F8 from identical broken state",
            "replacement": False,
        },
        "arms": {
            "ordered": list(ARMS),
            "quantiles": dict(zip(QUANTILE_ARMS, QUANTILES, strict=True)),
            "contrasts": {key: list(item) for key, item in CONTRASTS.items()},
            "controls": {
                key: list(item) for key, item in CONTROL_CONTRASTS.items()
            },
            "outgoing_rule": "x @ beta == beta.T @ x",
            "tighten_factor": 1.5,
            "loosen_factor": 1.0 / 1.5,
            "block_random": "exact 0.5-block-norm and launch-throughput neutral",
        },
        "information_redistribution": {
            "estimator": "frozen calibrated PX10 local-MMI PhiID",
            "partition": "fixed unedited-beta Fiedler",
            "lags": list(ALL_LAGS),
            "primary_lags": list(PRIMARY_LAGS),
            "temporal_derangements": list(TEMPORAL_SHIFTS),
            "groups": {key: list(item) for key, item in GROUP_COLUMNS.items()},
            "iri": "downward + upward - redundant - synergy - cross_transfer",
            "iri_is_phi_r": False,
            "public_nine_atom_negative_control": True,
        },
        "dose_channel": {
            "input": "state-centred frozen predicted molecular-edit shift",
            "model": "one signed candidate/source-half logistic coefficient",
            "offset": "Jeffreys-smoothed source-half NOOP rate",
            "confirmation_coefficients_frozen_from_pilot": True,
            "label_derangements": 16,
            "equivalence_bits": INFORMATION_EQUIVALENCE_MARGIN_BITS,
        },
        "sensor": {
            "score": "PX10 temporal pairing correction",
            "profiles": [_json_ready(item.__dict__) for item in SENSOR_PROFILES],
            "selection_order": list(SENSOR_SELECTION_ORDER),
            "minimum_full_effect_retention": SENSOR_RETENTION_FRACTION,
            "ensemble_not_single_state_sensor": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "randomization_draws": RANDOMIZATION_DRAWS,
            "holm_within_named_families": True,
            "probability_equivalence_margin": OUTCOME_EQUIVALENCE_MARGIN,
        },
        "randomness": {
            "domains": SEED_DOMAINS,
            "future_seed_excludes_arm": True,
            "observation_seed_excludes_arm": True,
            "common_random_streams": True,
        },
        "operational": {
            "workers_max": MAX_WORKERS,
            "total_cpu_hours_max": MAX_TOTAL_CPU_HOURS,
            "detached_science": True,
            "matrix_checkpointing": True,
            "complete_replay": True,
            "large_artifacts_ignored": True,
        },
        "claim_boundary": {
            "prohibited": [
                "rescue of public Phi-r",
                "Phi-r as cause",
                "consciousness",
                "agency",
                "life",
                "biological memory",
                "origin-of-life universality",
                "Platonic space",
                "Ruliad",
            ]
        },
    }
    value["protocol_id"] = _digest(value)
    return value


@dataclass(frozen=True)
class ArmPlan:
    arm: str
    family: str
    role: str
    edit: MolecularEdit | None
    beta: NDArray[np.float64]
    prediction: float
    predicted_shift: float
    empirical_quantile: float
    selected_rank: int
    legal_edits: int
    surgery: BetaSurgery | None = None


@dataclass(frozen=True)
class CasePayload:
    case: px9.ResilienceCase
    initial_by_arm: Mapping[str, NDArray[np.int64]]
    blocks_by_arm: Mapping[str, tuple[px9.PairBlock, ...]]


@dataclass(frozen=True)
class PX11Batch:
    matrix_id: int
    beta: NDArray[np.float64]
    initial: NDArray[np.int16]
    acquisition_rows: tuple[dict[str, Any], ...]
    intervention_rows: tuple[dict[str, Any], ...]
    branch_rows: tuple[dict[str, Any], ...]
    atom_rows: tuple[dict[str, Any], ...]
    sensor_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str


@dataclass(frozen=True)
class TransportBatch:
    regime: str
    matrix_id: int
    score_rows: tuple[dict[str, Any], ...]
    replay_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str


def _edit_key(edit: MolecularEdit) -> tuple[int, int]:
    return int(edit.remove_type), int(edit.add_type)


def _score_for_edit(
    scores: Sequence[Any], edit: MolecularEdit
) -> tuple[int, float, float]:
    key = _edit_key(edit)
    for index, scored in enumerate(scores):
        if _edit_key(scored.edit) == key:
            return (
                index,
                float(scored.predicted_probability),
                float(scored.predicted_shift),
            )
    raise AssertionError("selected legal edit is absent from exhaustive scores")


def _beta_prediction(
    student: cr5.FrozenCR5Student,
    case: px9.ResilienceCase,
    beta: NDArray[np.float64],
) -> float:
    altered = StateCase(
        state_id=case.state_id,
        cohort="PX11_POST_BREAK",
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        landmark=case.landmark,
        beta=beta,
        snapshot=case.snapshot,
    )
    return student.predict_case(altered, GardConfig())


def _select_arms(
    case: px9.ResilienceCase,
    students: Mapping[tuple[str, str], cr5.FrozenCR5Student],
    spec: PX11Spec,
) -> tuple[ArmPlan, ...]:
    required = set(_active_arms(spec))
    student = students[("renewal", case.candidate)]
    noop, scores = cr5.score_student_edits(
        student, case.as_state_case(), GardConfig()
    )
    selected, ranks = cr2.select_quantile_edits(scores)
    plans: dict[str, ArmPlan] = {}
    for arm, quantile, scored, rank in zip(
        QUANTILE_ARMS, QUANTILES, selected, ranks, strict=True
    ):
        plans[arm] = ArmPlan(
            arm,
            "model",
            "high" if arm == "Q100" else "low" if arm == "Q00" else "dose",
            scored.edit,
            case.beta,
            float(scored.predicted_probability),
            float(scored.predicted_shift),
            float(quantile),
            int(rank),
            len(scores),
        )

    if "RANDOM_SWAP" in required:
        random_rng = np.random.default_rng(
            _seed(
                spec,
                "random_action",
                case.candidate,
                case.matrix_id,
                case.landmark,
            )
        )
        random_rank = int(random_rng.integers(0, len(scores)))
        random_score = scores[random_rank]
        plans["RANDOM_SWAP"] = ArmPlan(
            "RANDOM_SWAP",
            "control",
            "random",
            random_score.edit,
            case.beta,
            float(random_score.predicted_probability),
            float(random_score.predicted_shift),
            float("nan"),
            random_rank,
            len(scores),
        )

    if required.intersection(("RULE_STABILIZE", "RULE_DESTABILIZE")):
        rules = select_outgoing_rule_edits(case.snapshot.composition, case.beta)
        rule_map = {
            "RULE_STABILIZE": (rules["RULE_DOWN"], "high"),
            "RULE_DESTABILIZE": (rules["RULE_UP"], "low"),
        }
        for arm, (edit, role) in rule_map.items():
            if arm not in required:
                continue
            rank, prediction, shift = _score_for_edit(scores, edit)
            plans[arm] = ArmPlan(
                arm,
                "physical_rule",
                role,
                edit,
                case.beta,
                prediction,
                shift,
                float("nan"),
                rank,
                len(scores),
            )

    if required.intersection(("TIGHTEN", "LOOSEN", "BLOCK_RANDOM")):
        neutral_rng = np.random.default_rng(
            _seed(
                spec,
                "beta_control",
                case.candidate,
                case.matrix_id,
                case.landmark,
            )
        )
        surgeries = {
            "TIGHTEN": p3c.multiplicative_surgery(
                case.snapshot.composition, case.beta, 1.5, "TIGHTEN"
            ),
            "LOOSEN": p3c.multiplicative_surgery(
                case.snapshot.composition, case.beta, 1.0 / 1.5, "LOOSEN"
            ),
        }
        if "BLOCK_RANDOM" in required:
            surgeries["BLOCK_RANDOM"] = p3c.throughput_neutral_pp_surgery(
                case.snapshot.composition,
                case.beta,
                neutral_rng,
                name="BLOCK_RANDOM",
            )
        for arm, family, role in (
            ("TIGHTEN", "beta_surgery", "high"),
            ("LOOSEN", "beta_surgery", "low"),
            ("BLOCK_RANDOM", "control", "random"),
        ):
            if arm not in required:
                continue
            surgery = surgeries[arm]
            prediction = _beta_prediction(student, case, surgery.beta)
            plans[arm] = ArmPlan(
                arm,
                family,
                role,
                None,
                surgery.beta,
                prediction,
                prediction - noop,
                float("nan"),
                -1,
                len(scores),
                surgery,
            )

    plans["NOOP"] = ArmPlan(
        "NOOP",
        "control",
        "noop",
        None,
        case.beta,
        noop,
        0.0,
        float("nan"),
        -1,
        len(scores),
    )
    output = tuple(plans[arm] for arm in _active_arms(spec))
    probabilities = np.asarray(
        [plans[arm].prediction for arm in QUANTILE_ARMS], dtype=np.float64
    )
    if np.any(np.diff(probabilities) < 0.0):
        raise AssertionError("PX11 quantile predictions are not monotone")
    return output


def _active_arms(spec: PX11Spec) -> tuple[str, ...]:
    if spec.stage in {"pilot", "smoke"}:
        return ARMS
    selected: set[str] = {"NOOP"}
    if "model" in spec.active_families or spec.dose_channel_active or spec.sensor_active:
        selected.update(QUANTILE_ARMS)
        selected.add("RANDOM_SWAP")
    if "physical_rule" in spec.active_families:
        selected.update(("RULE_STABILIZE", "RULE_DESTABILIZE", "RANDOM_SWAP"))
    if "beta_surgery" in spec.active_families:
        selected.update(("TIGHTEN", "LOOSEN", "BLOCK_RANDOM"))
    return tuple(arm for arm in ARMS if arm in selected)


def _intervention_row(case: px9.ResilienceCase, plan: ArmPlan) -> dict[str, Any]:
    edit = plan.edit
    surgery = plan.surgery
    before_throughput = p3c.catalytic_throughput(
        case.snapshot.composition, case.beta
    )
    after_throughput = p3c.catalytic_throughput(
        case.snapshot.composition, plan.beta
    )
    outgoing = outgoing_catalytic_influence(
        case.snapshot.composition, case.beta
    )
    return {
        "state_id": case.state_id,
        "candidate": case.candidate,
        "matrix_id": case.matrix_id,
        "landmark": case.landmark,
        "arm": plan.arm,
        "family": plan.family,
        "role": plan.role,
        "remove_type": -1 if edit is None else int(edit.remove_type),
        "add_type": -1 if edit is None else int(edit.add_type),
        "prediction": plan.prediction,
        "predicted_shift": plan.predicted_shift,
        "empirical_quantile": plan.empirical_quantile,
        "selected_rank": plan.selected_rank,
        "legal_edits": plan.legal_edits,
        "outgoing_delta": (
            0.0
            if edit is None
            else float(outgoing[edit.add_type] - outgoing[edit.remove_type])
        ),
        "is_beta_surgery": int(surgery is not None),
        "changed_edges": 0 if surgery is None else int(len(surgery.flat_indices)),
        "changed_edges_json": (
            "[]"
            if surgery is None
            else json.dumps([int(value) for value in surgery.flat_indices])
        ),
        "requested_norm": 0.0 if surgery is None else surgery.requested_norm,
        "observed_norm": 0.0 if surgery is None else surgery.observed_norm,
        "launch_throughput_before": before_throughput,
        "launch_throughput_after": after_throughput,
        "beta_digest": _array_digest(plan.beta),
    }


def _simulate_case(
    case: px9.ResilienceCase,
    students: Mapping[tuple[str, str], cr5.FrozenCR5Student],
    spec: PX11Spec,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    CasePayload,
]:
    plans = _select_arms(case, students, spec)
    branch_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    initial_by_arm: dict[str, NDArray[np.int64]] = {}
    blocks_by_arm: dict[str, tuple[px9.PairBlock, ...]] = {}
    for plan in plans:
        intervention_rows.append(_intervention_row(case, plan))
        initial = (
            np.asarray(case.snapshot.composition, dtype=np.int64).copy()
            if plan.edit is None
            else apply_molecular_edit(case.snapshot.composition, plan.edit)
        )
        initial_by_arm[plan.arm] = initial
        altered_case = replace(case, beta=plan.beta)
        rows: list[dict[str, Any]] = []
        blocks: list[px9.PairBlock] = []
        for branch in range(spec.branches):
            row, block = px9._simulate_branch(
                altered_case, plan.edit, branch, spec.as_px9()
            )
            row.update(
                {
                    "arm": plan.arm,
                    "family": plan.family,
                    "role": plan.role,
                    "prediction": plan.prediction,
                    "predicted_shift": plan.predicted_shift,
                    "empirical_quantile": plan.empirical_quantile,
                }
            )
            rows.append(row)
            blocks.append(block)
        branch_rows.extend(rows)
        blocks_by_arm[plan.arm] = tuple(blocks)
    return (
        branch_rows,
        intervention_rows,
        CasePayload(case, initial_by_arm, blocks_by_arm),
    )


def _as_atom_payload(payload: CasePayload) -> px10.AtomCasePayload:
    return px10.AtomCasePayload(
        payload.case,
        payload.initial_by_arm,
        payload.blocks_by_arm,
    )


def _pooled_atom_rows(
    matrix_id: int,
    beta: NDArray[np.float64],
    payloads: Sequence[CasePayload],
    spec: PX11Spec,
) -> list[dict[str, Any]]:
    """Score the frozen atoms after pooling landmarks within each matrix."""

    first, second = beta_physical_partition(beta)
    rows: list[dict[str, Any]] = []
    halves = _halves(spec)
    for candidate in CANDIDATES:
        local = [item for item in payloads if item.case.candidate == candidate]
        if not local:
            continue
        available_arms = set(local[0].blocks_by_arm) if local else set()
        for arm in (item for item in ATOM_ARMS if item in available_arms):
            for half, indices in halves.items():
                for lag in ALL_LAGS:
                    real_past: list[NDArray] = []
                    real_future: list[NDArray] = []
                    controls: dict[int, tuple[list[NDArray], list[NDArray]]] = {
                        control: ([], [])
                        for control in range(len(TEMPORAL_SHIFTS))
                    }
                    for payload in local:
                        atom_payload = _as_atom_payload(payload)
                        left, right, _ = px10._lag_pairs(
                            atom_payload, arm, indices, lag
                        )
                        if len(left):
                            real_past.append(left)
                            real_future.append(right)
                        for control, shift in enumerate(TEMPORAL_SHIFTS):
                            left_s, right_s, self_pairs = px10._lag_pairs(
                                atom_payload, arm, indices, lag, shift
                            )
                            if self_pairs:
                                raise AssertionError(
                                    "PX11 atom derangement retained a self-pair"
                                )
                            if len(left_s):
                                controls[control][0].append(left_s)
                                controls[control][1].append(right_s)
                    if not real_past:
                        continue
                    past = np.vstack(real_past)
                    future = np.vstack(real_future)
                    key = {
                        "matrix_id": matrix_id,
                        "candidate": candidate,
                        "arm": arm,
                        "source_half": half,
                        "lag": lag,
                        "support_branches": len(indices),
                    }
                    rows.append(
                        {
                            **key,
                            "score_kind": "paired_beta",
                            "control_id": -1,
                            **px10._atom_score(past, future, first, second),
                        }
                    )
                    for control in range(len(TEMPORAL_SHIFTS)):
                        if not controls[control][0]:
                            continue
                        left = np.vstack(controls[control][0])
                        right = np.vstack(controls[control][1])
                        rows.append(
                            {
                                **key,
                                "score_kind": "shuffled_beta",
                                "control_id": control,
                                **px10._atom_score(left, right, first, second),
                            }
                        )
    return rows


def _coordinate_mask(
    case: px9.ResilienceCase,
    profile: SensorProfile,
    spec: PX11Spec,
) -> NDArray[np.bool_]:
    size = GardConfig().n_types
    if profile.coordinate_fraction >= 1.0:
        return np.ones(size, dtype=bool)
    keep = max(2, int(math.floor(profile.coordinate_fraction * size)))
    rng = np.random.default_rng(
        _seed(
            spec,
            "observation_mask",
            profile.name,
            case.candidate,
            case.matrix_id,
        )
    )
    selected = np.sort(rng.choice(size, size=keep, replace=False))
    mask = np.zeros(size, dtype=bool)
    mask[selected] = True
    return mask


def _observe_rows(
    values: NDArray,
    mask: NDArray[np.bool_],
    profile: SensorProfile,
    spec: PX11Spec,
    case: px9.ResilienceCase,
    branch: int,
    role: str,
) -> NDArray[np.int16]:
    observed = np.asarray(values, dtype=np.int64).copy()
    observed[:, ~mask] = 0
    if profile.count_fraction < 1.0:
        for depth in range(len(observed)):
            rng = np.random.default_rng(
                _seed(
                    spec,
                    "observation_noise",
                    profile.name,
                    case.candidate,
                    case.matrix_id,
                    case.landmark,
                    branch,
                    role,
                    depth,
                )
            )
            observed[depth] = rng.binomial(
                observed[depth], profile.count_fraction
            )
    if profile.depth_stride > 1:
        observed = observed[:: profile.depth_stride]
    return observed.astype(np.int16, copy=False)


def _observed_blocks(
    case: px9.ResilienceCase,
    blocks: Sequence[px9.PairBlock],
    indices: Sequence[int],
    profile: SensorProfile,
    spec: PX11Spec,
) -> tuple[px9.PairBlock, ...]:
    mask = _coordinate_mask(case, profile, spec)
    output: list[px9.PairBlock] = []
    empty = np.empty((0, GardConfig().n_types), dtype=np.int16)
    for branch in indices:
        block = blocks[branch]
        past = _observe_rows(
            block.generational_past,
            mask,
            profile,
            spec,
            case,
            branch,
            "past",
        )
        future = _observe_rows(
            block.generational_future,
            mask,
            profile,
            spec,
            case,
            branch,
            "future",
        )
        length = min(len(past), len(future))
        output.append(px9.PairBlock(empty, empty, past[:length], future[:length]))
    return tuple(output)


def _sensor_score(
    case: px9.ResilienceCase,
    blocks: Sequence[px9.PairBlock],
    indices: Sequence[int],
    profile: SensorProfile,
    spec: PX11Spec,
) -> dict[str, Any]:
    selected = tuple(indices[: min(profile.support, len(indices))])
    observed = _observed_blocks(case, blocks, selected, profile, spec)
    first, second = beta_physical_partition(case.beta)
    past, future = px9._concatenate_pairs(observed, "generational")
    paired = px9._fixed_partition_score(past, future, first, second)
    controls: list[float] = []
    self_pairs = 0
    for shift in TEMPORAL_SHIFTS:
        left, right, retained = px9._temporal_derangement(observed, shift)
        self_pairs += retained
        score = px9._fixed_partition_score(left, right, first, second)
        controls.append(float(score["value"]))
    control_mean = float(np.nanmean(controls)) if controls else float("nan")
    return {
        "paired_value": float(paired["value"]),
        "shuffled_mean": control_mean,
        "temporal_value": float(paired["value"] - control_mean),
        "transitions": int(paired["transitions"]),
        "active_dimensions": int(paired["active_dimensions"]),
        "branches_used": len(selected),
        "temporal_controls": len(controls),
        "self_pairs": int(self_pairs),
        "valid": bool(
            np.isfinite(paired["value"])
            and np.isfinite(control_mean)
            and self_pairs == 0
        ),
    }


def _sensor_jobs(spec: PX11Spec) -> tuple[tuple[str, SensorProfile], ...]:
    if not spec.sensor_active:
        return ()
    if spec.stage == "pilot":
        jobs: list[tuple[str, SensorProfile]] = []
        for profile in SENSOR_PROFILES:
            for arm in ("Q00", "Q100"):
                jobs.append((arm, profile))
        for arm in QUANTILE_ARMS[1:-1]:
            jobs.append((arm, SENSOR_BY_NAME["FULL64"]))
        return tuple(jobs)
    if spec.stage == "smoke":
        return tuple(
            (arm, SENSOR_BY_NAME[name])
            for name in ("FULL16", "COMPACT")
            for arm in ("Q00", "Q100")
        )
    selected = SENSOR_BY_NAME[spec.sensor_profile]
    jobs = [(arm, selected) for arm in QUANTILE_ARMS]
    if selected.name != "FULL64":
        jobs.extend((arm, SENSOR_BY_NAME["FULL64"]) for arm in ("Q00", "Q100"))
    return tuple(jobs)


def _sensor_rows(
    payload: CasePayload, spec: PX11Spec
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, profile in _sensor_jobs(spec):
        for half, indices in _halves(spec).items():
            score = _sensor_score(
                payload.case,
                payload.blocks_by_arm[arm],
                indices,
                profile,
                spec,
            )
            rows.append(
                {
                    "state_id": payload.case.state_id,
                    "candidate": payload.case.candidate,
                    "matrix_id": payload.case.matrix_id,
                    "landmark": payload.case.landmark,
                    "arm": arm,
                    "source_half": half,
                    "profile": profile.name,
                    "support": profile.support,
                    "coordinate_fraction": profile.coordinate_fraction,
                    "count_fraction": profile.count_fraction,
                    "depth_stride": profile.depth_stride,
                    **score,
                }
            )
    return rows


def _run_matrix(arguments: tuple[int, PX11Spec, str, str]) -> PX11Batch:
    matrix_id, spec, model_path, contract_path = arguments
    started = time.process_time()
    with threadpool_limits(limits=1), _px9_seed_context(spec):
        config = GardConfig()
        beta = generate_beta(
            config, np.random.default_rng(_seed(spec, "matrix", matrix_id))
        )
        initial = generate_initial_composition(
            config, np.random.default_rng(_seed(spec, "initial", matrix_id))
        ).astype(np.int16)
        students = cr5.load_students(Path(model_path), Path(contract_path))
        natural: list[px9.ResilienceCase] = []
        for candidate in CANDIDATES:
            natural.extend(
                px9._run_natural_candidate(
                    matrix_id, beta, initial, candidate, spec.as_px9()
                )
            )
        cases: list[px9.ResilienceCase] = []
        acquisition_rows: list[dict[str, Any]] = []
        for source in natural:
            broken, acquisition = px9._acquire_break(source, spec.as_px9())
            acquisition_rows.append(acquisition)
            if broken is not None:
                cases.append(broken)
        intervention_rows: list[dict[str, Any]] = []
        branch_rows: list[dict[str, Any]] = []
        sensor_rows: list[dict[str, Any]] = []
        payloads: list[CasePayload] = []
        for case in cases:
            branches, interventions, payload = _simulate_case(
                case, students, spec
            )
            branch_rows.extend(branches)
            intervention_rows.extend(interventions)
            sensor_rows.extend(_sensor_rows(payload, spec))
            payloads.append(payload)
        atom_rows = _pooled_atom_rows(matrix_id, beta, payloads, spec)
        digest = _digest(
            {
                "matrix_id": matrix_id,
                "beta": _array_digest(beta),
                "initial": _array_digest(initial),
                "acquisition": acquisition_rows,
                "interventions": intervention_rows,
                "branches": branch_rows,
                "atoms": atom_rows,
                "sensor": sensor_rows,
            }
        )
        return PX11Batch(
            matrix_id,
            beta,
            initial,
            tuple(acquisition_rows),
            tuple(intervention_rows),
            tuple(branch_rows),
            tuple(atom_rows),
            tuple(sensor_rows),
            float(time.process_time() - started),
            digest,
        )


def _bootstrap_summary(
    series: pd.Series,
    spec: PX11Spec,
    key: str,
    arrays: dict[str, NDArray],
    direction: float = 1.0,
) -> dict[str, Any]:
    local = series.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    values = np.asarray(local, dtype=np.float64)
    matrix_ids = np.asarray(local.index, dtype=np.int64)
    safe = key.replace("/", "__")
    if not values.size:
        return {
            "effect": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "ci90": [float("nan"), float("nan")],
            "one_sided_p": 1.0,
            "matrices": 0,
            "matrices_positive": 0,
            "maximum_absolute_matrix_effect": float("nan"),
            "minimum_leave_one_out_aligned": float("nan"),
        }
    rng = np.random.default_rng(_seed(spec, "bootstrap", key))
    indices = rng.integers(
        0, len(values), size=(spec.bootstrap_draws, len(values))
    )
    bootstrap = values[indices].mean(axis=1)
    sign_rng = np.random.default_rng(_seed(spec, "randomization", key))
    signs = sign_rng.choice(
        (-1.0, 1.0), size=(spec.randomization_draws, len(values))
    )
    observed_aligned = direction * float(values.mean())
    randomized = direction * (signs * values).mean(axis=1)
    if len(values) > 1:
        leave_one_out = (values.sum() - values) / (len(values) - 1)
        minimum_loo = float(np.min(direction * leave_one_out))
        maximum_influence = float(
            np.max(np.abs(leave_one_out - values.mean()))
        )
    else:
        minimum_loo = float("nan")
        maximum_influence = float("nan")
    output = {
        "effect": float(values.mean()),
        "aligned_effect": observed_aligned,
        "ci95": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
        "ci90": [
            float(value) for value in np.quantile(bootstrap, (0.05, 0.95))
        ],
        "one_sided_p": float(
            (1 + np.count_nonzero(randomized >= observed_aligned))
            / (len(randomized) + 1)
        ),
        "matrices": int(len(values)),
        "matrices_positive": int(np.count_nonzero(direction * values > 0)),
        "maximum_absolute_matrix_effect": float(np.max(np.abs(values))),
        "maximum_leave_one_out_influence": maximum_influence,
        "minimum_leave_one_out_aligned": minimum_loo,
    }
    arrays[f"{safe}__matrix_ids"] = matrix_ids
    arrays[f"{safe}__matrix_values"] = values
    arrays[f"{safe}__bootstrap"] = bootstrap
    arrays[f"{safe}__randomization"] = randomized
    return output


def _adjust_family(
    items: list[dict[str, Any]], directions: Sequence[float] | None = None
) -> None:
    if not items:
        return
    if directions is None:
        directions = [1.0] * len(items)
    adjusted = holm_adjust([float(item["one_sided_p"]) for item in items])
    for item, adjusted_p, direction in zip(
        items, adjusted, directions, strict=True
    ):
        lower, upper = item["ci95"]
        interval_pass = lower > 0.0 if direction > 0 else upper < 0.0
        item["holm_adjusted_p"] = float(adjusted_p)
        item["pass"] = bool(
            direction * float(item["effect"]) > 0.0
            and interval_pass
            and adjusted_p < 0.05
        )


def _matrix_arm_means(
    branches: pd.DataFrame, candidate: str, half: str
) -> pd.DataFrame:
    local = branches[
        (branches["candidate"] == candidate) & (branches["half"] == half)
    ]
    return (
        local.groupby(["matrix_id", "arm"], sort=True)["primary"]
        .mean()
        .unstack("arm")
    )


def _outcome_analysis(
    branches: pd.DataFrame,
    spec: PX11Spec,
    arrays: dict[str, NDArray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    causal: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    dose: list[dict[str, Any]] = []
    for family, (high, low) in CONTRASTS.items():
        if high not in set(branches["arm"]) or low not in set(branches["arm"]):
            continue
        family_items: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            for half in _halves(spec):
                means = _matrix_arm_means(branches, candidate, half)
                effect = means[high] - means[low]
                item = _bootstrap_summary(
                    effect,
                    spec,
                    f"outcome/{family}/{candidate}/{half}",
                    arrays,
                )
                item.update(
                    {
                        "family": family,
                        "candidate": candidate,
                        "source_half": half,
                        "high_arm": high,
                        "low_arm": low,
                    }
                )
                family_items.append(item)
                causal.append(item)
        _adjust_family(family_items)

    for control_name, (arm, noop) in CONTROL_CONTRASTS.items():
        if arm not in set(branches["arm"]) or noop not in set(branches["arm"]):
            continue
        for candidate in CANDIDATES:
            for half in _halves(spec):
                means = _matrix_arm_means(branches, candidate, half)
                item = _bootstrap_summary(
                    means[arm] - means[noop],
                    spec,
                    f"outcome_control/{control_name}/{candidate}/{half}",
                    arrays,
                )
                item.update(
                    {
                        "control": control_name,
                        "candidate": candidate,
                        "source_half": half,
                        "margin": OUTCOME_EQUIVALENCE_MARGIN,
                        "equivalent": bool(
                            item["ci90"][0] > -OUTCOME_EQUIVALENCE_MARGIN
                            and item["ci90"][1] < OUTCOME_EQUIVALENCE_MARGIN
                        ),
                    }
                )
                controls.append(item)

    if not set(QUANTILE_ARMS).issubset(set(branches["arm"])):
        return causal, controls, dose
    quantile_index = {arm: index for index, arm in enumerate(QUANTILE_ARMS)}
    local = branches[branches["arm"].isin(QUANTILE_ARMS)]
    state_means = (
        local.groupby(
            ["matrix_id", "candidate", "state_id", "half", "arm"], sort=True
        )
        .agg(primary=("primary", "mean"), predicted_shift=("predicted_shift", "first"))
        .reset_index()
    )
    correlations: list[dict[str, Any]] = []
    for key, frame in state_means.groupby(
        ["matrix_id", "candidate", "state_id", "half"], sort=True
    ):
        ordered = frame.assign(
            arm_order=frame["arm"].map(quantile_index)
        ).sort_values("arm_order")
        if len(ordered) != len(QUANTILE_ARMS):
            continue
        value = spearmanr(
            ordered["predicted_shift"], ordered["primary"]
        ).statistic
        correlations.append(
            {
                "matrix_id": int(key[0]),
                "candidate": str(key[1]),
                "state_id": str(key[2]),
                "source_half": str(key[3]),
                "spearman": float(value) if np.isfinite(value) else 0.0,
            }
        )
    corr_frame = pd.DataFrame(correlations)
    dose_items: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for half in _halves(spec):
            selected = corr_frame[
                (corr_frame["candidate"] == candidate)
                & (corr_frame["source_half"] == half)
            ]
            matrix_values = selected.groupby("matrix_id")["spearman"].mean()
            item = _bootstrap_summary(
                matrix_values,
                spec,
                f"outcome_dose/{candidate}/{half}",
                arrays,
            )
            item.update({"candidate": candidate, "source_half": half})
            dose_items.append(item)
            dose.append(item)
    _adjust_family(dose_items)
    return causal, controls, dose


def _derive_atom_scores(atoms: pd.DataFrame) -> pd.DataFrame:
    keys = ["matrix_id", "candidate", "arm", "source_half", "lag"]
    rows: list[dict[str, Any]] = []
    for key, local in atoms.groupby(keys, sort=True):
        paired = local[local["score_kind"] == "paired_beta"]
        shuffled = local[local["score_kind"] == "shuffled_beta"]
        if len(paired) != 1 or shuffled.empty:
            continue
        row: dict[str, Any] = dict(zip(keys, key, strict=True))
        for name in ATOM_NAMES:
            column = f"atom_{name}"
            row[column] = float(paired.iloc[0][column] - shuffled[column].mean())
        row["revised_phi_r"] = float(
            paired.iloc[0]["revised_phi_r"]
            - shuffled["revised_phi_r"].mean()
        )
        for group, columns in GROUP_COLUMNS.items():
            row[group] = float(sum(row[column] for column in columns))
        row["iri"] = float(
            row["downward_routing"]
            + row["upward_integration"]
            - row["redundant_persistence"]
            - row["synergy_persistence"]
            - row["cross_part_transfer"]
        )
        row["transitions"] = int(paired.iloc[0]["transitions"])
        row["temporal_controls"] = int(len(shuffled))
        rows.append(row)
    return pd.DataFrame(rows)


def _atom_analysis(
    derived: pd.DataFrame,
    spec: PX11Spec,
    arrays: dict[str, NDArray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    iri_items: list[dict[str, Any]] = []
    attenuation_items: list[dict[str, Any]] = []
    group_items: list[dict[str, Any]] = []
    value_columns = (*GROUP_COLUMNS, "iri", "revised_phi_r")
    for family, (high, low) in CONTRASTS.items():
        if high not in set(derived["arm"]) or low not in set(derived["arm"]):
            continue
        family_iri: list[dict[str, Any]] = []
        family_attenuation: list[dict[str, Any]] = []
        local_family = derived[derived["arm"].isin((high, low))]
        for candidate in CANDIDATES:
            for half in _halves(spec):
                local = local_family[
                    (local_family["candidate"] == candidate)
                    & (local_family["source_half"] == half)
                ]
                primary = local[local["lag"].isin(PRIMARY_LAGS)]
                for value_name in value_columns:
                    values = (
                        primary.groupby(["matrix_id", "arm"])[value_name]
                        .mean()
                        .unstack("arm")
                    )
                    effect = values.get(high, pd.Series(dtype=float)) - values.get(
                        low, pd.Series(dtype=float)
                    )
                    item = _bootstrap_summary(
                        effect,
                        spec,
                        f"atom/{family}/{candidate}/{half}/{value_name}",
                        arrays,
                    )
                    item.update(
                        {
                            "family": family,
                            "candidate": candidate,
                            "source_half": half,
                            "measure": value_name,
                            "high_arm": high,
                            "low_arm": low,
                        }
                    )
                    group_items.append(item)
                    if value_name == "iri":
                        family_iri.append(item)
                        iri_items.append(item)

                lag_values = (
                    local.groupby(["matrix_id", "arm", "lag"])["iri"]
                    .mean()
                    .unstack(["arm", "lag"])
                )
                early = sum(
                    lag_values.get((high, lag), pd.Series(dtype=float))
                    - lag_values.get((low, lag), pd.Series(dtype=float))
                    for lag in EARLY_LAGS
                ) / len(EARLY_LAGS)
                late = sum(
                    lag_values.get((high, lag), pd.Series(dtype=float))
                    - lag_values.get((low, lag), pd.Series(dtype=float))
                    for lag in LATE_LAGS
                ) / len(LATE_LAGS)
                attenuation = _bootstrap_summary(
                    early - late,
                    spec,
                    f"atom_attenuation/{family}/{candidate}/{half}",
                    arrays,
                )
                attenuation.update(
                    {
                        "family": family,
                        "candidate": candidate,
                        "source_half": half,
                        "measure": "iri_early_minus_late",
                    }
                )
                family_attenuation.append(attenuation)
                attenuation_items.append(attenuation)
        _adjust_family(family_iri)
        _adjust_family(family_attenuation)
    return iri_items, attenuation_items, group_items


def _log_loss_bits(outcome: NDArray, probability: NDArray) -> NDArray:
    y = np.asarray(outcome, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(y * np.log2(p) + (1.0 - y) * np.log2(1.0 - p))


def _dose_design(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame[frame["arm"].isin(QUANTILE_ARMS)].copy()
    local["centered_shift"] = local["predicted_shift"] - local.groupby(
        "state_id", sort=True
    )["predicted_shift"].transform("mean")
    return local


def _source_noop_offsets(frame: pd.DataFrame) -> pd.DataFrame:
    noop = frame[frame["arm"] == "NOOP"]
    counts = (
        noop.groupby(["state_id", "matrix_id"], sort=True)["primary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    counts["p0"] = (counts["sum"] + 0.5) / (counts["count"] + 1.0)
    counts["offset"] = logit(np.clip(counts["p0"], 1e-8, 1.0 - 1e-8))
    return counts[["state_id", "matrix_id", "p0", "offset"]]


def _fit_dose_coefficient(
    source: pd.DataFrame,
) -> dict[str, float]:
    design = _dose_design(source)
    unique = design.drop_duplicates(["state_id", "arm"])
    scale = float(np.std(unique["centered_shift"], ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        return {
            "coefficient": 0.0,
            "shift_scale": 1.0,
            "objective": float("nan"),
            "converged": False,
        }
    offsets = _source_noop_offsets(source)
    design = design.merge(offsets, on=["state_id", "matrix_id"], how="inner")
    x = design["centered_shift"].to_numpy(dtype=float) / scale
    y = design["primary"].to_numpy(dtype=float)
    offset = design["offset"].to_numpy(dtype=float)

    def objective(coefficient: float) -> float:
        logits = offset + coefficient * x
        return float(np.mean(np.logaddexp(0.0, logits) - y * logits))

    fitted = minimize_scalar(
        objective,
        method="bounded",
        bounds=(-50.0, 50.0),
        options={"xatol": 1e-10, "maxiter": 1000},
    )
    return {
        "coefficient": float(fitted.x),
        "shift_scale": scale,
        "objective": float(fitted.fun),
        "converged": bool(fitted.success),
    }


def _channel_matrix_gain(
    branches: pd.DataFrame,
    candidate: str,
    source_half: str,
    target_half: str,
    model: Mapping[str, Any],
    *,
    label_control: int | None = None,
    spec: PX11Spec,
) -> pd.Series:
    local = branches[branches["candidate"] == candidate]
    source = local[local["half"] == source_half]
    target = _dose_design(local[local["half"] == target_half])
    offsets = _source_noop_offsets(source)
    target = target.merge(offsets, on=["state_id", "matrix_id"], how="inner")
    target["model_shift"] = target["centered_shift"]
    if label_control is not None:
        arm_order = tuple(QUANTILE_ARMS)
        mapping_rows: list[dict[str, Any]] = []
        unique = target.drop_duplicates(["state_id", "arm"])
        for state_id, state in unique.groupby("state_id", sort=True):
            ordered = state.set_index("arm").loc[list(arm_order)]
            rng = np.random.default_rng(
                _seed(
                    spec,
                    "channel_derangement",
                    candidate,
                    source_half,
                    state_id,
                    label_control,
                )
            )
            permutation = px10._derangement(len(arm_order), rng)
            values = ordered["centered_shift"].to_numpy(dtype=float)[permutation]
            mapping_rows.extend(
                {
                    "state_id": state_id,
                    "arm": arm,
                    "deranged_shift": float(value),
                }
                for arm, value in zip(arm_order, values, strict=True)
            )
        target = target.drop(columns="model_shift").merge(
            pd.DataFrame(mapping_rows), on=["state_id", "arm"], how="inner"
        )
        target["model_shift"] = target["deranged_shift"]
    coefficient = float(model["coefficient"])
    scale = float(model["shift_scale"])
    probability = expit(
        target["offset"].to_numpy(dtype=float)
        + coefficient * target["model_shift"].to_numpy(dtype=float) / scale
    )
    baseline = target["p0"].to_numpy(dtype=float)
    outcome = target["primary"].to_numpy(dtype=float)
    target["gain_bits"] = _log_loss_bits(outcome, baseline) - _log_loss_bits(
        outcome, probability
    )
    return target.groupby("matrix_id", sort=True)["gain_bits"].mean()


def _two_action_channel_gain(
    branches: pd.DataFrame,
    candidate: str,
    source_half: str,
    target_half: str,
) -> pd.Series:
    local = branches[
        (branches["candidate"] == candidate)
        & (branches["arm"].isin(("RANDOM_SWAP", "NOOP")))
    ]
    source = local[local["half"] == source_half]
    target = local[local["half"] == target_half].copy()
    arm_counts = (
        source.groupby(["state_id", "arm"], sort=True)["primary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    arm_counts["p_arm"] = (arm_counts["sum"] + 0.5) / (
        arm_counts["count"] + 1.0
    )
    state_counts = (
        source.groupby("state_id", sort=True)["primary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    state_counts["p_state"] = (state_counts["sum"] + 0.5) / (
        state_counts["count"] + 1.0
    )
    target = target.merge(
        arm_counts[["state_id", "arm", "p_arm"]],
        on=["state_id", "arm"],
        how="inner",
    ).merge(
        state_counts[["state_id", "p_state"]], on="state_id", how="inner"
    )
    target["gain_bits"] = _log_loss_bits(
        target["primary"].to_numpy(), target["p_state"].to_numpy()
    ) - _log_loss_bits(
        target["primary"].to_numpy(), target["p_arm"].to_numpy()
    )
    return target.groupby("matrix_id", sort=True)["gain_bits"].mean()


def _dose_channel_analysis(
    branches: pd.DataFrame,
    spec: PX11Spec,
    arrays: dict[str, NDArray],
    frozen_models: Mapping[str, Any] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    primary: list[dict[str, Any]] = []
    label_null: list[dict[str, Any]] = []
    random_null: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    if not set((*QUANTILE_ARMS, "RANDOM_SWAP", "NOOP")).issubset(
        set(branches["arm"])
    ):
        return primary, label_null, random_null, models
    for candidate in CANDIDATES:
        for source_half, target_half in (("A", "B"), ("B", "A")):
            key = f"c{candidate}_{source_half}"
            source = branches[
                (branches["candidate"] == candidate)
                & (branches["half"] == source_half)
            ]
            if frozen_models is None:
                model = _fit_dose_coefficient(source)
            else:
                model = dict(frozen_models[key])
            models[key] = {
                **model,
                "candidate": candidate,
                "source_half": source_half,
                "target_half": target_half,
            }
            gain = _channel_matrix_gain(
                branches,
                candidate,
                source_half,
                target_half,
                model,
                spec=spec,
            )
            item = _bootstrap_summary(
                gain,
                spec,
                f"dose_channel/{candidate}/{source_half}_to_{target_half}",
                arrays,
            )
            item.update(
                {
                    "candidate": candidate,
                    "direction": f"{source_half}_to_{target_half}",
                    "coefficient": float(model["coefficient"]),
                    "shift_scale": float(model["shift_scale"]),
                    "model_frozen_from_pilot": frozen_models is not None,
                }
            )
            primary.append(item)

            controls = [
                _channel_matrix_gain(
                    branches,
                    candidate,
                    source_half,
                    target_half,
                    model,
                    label_control=control,
                    spec=spec,
                )
                for control in range(16)
            ]
            control_frame = pd.concat(controls, axis=1)
            control_item = _bootstrap_summary(
                control_frame.mean(axis=1),
                spec,
                f"dose_channel_null/{candidate}/{source_half}_to_{target_half}",
                arrays,
            )
            control_item.update(
                {
                    "candidate": candidate,
                    "direction": f"{source_half}_to_{target_half}",
                    "control": "arm_label_derangement",
                    "margin_bits": INFORMATION_EQUIVALENCE_MARGIN_BITS,
                    "equivalent": bool(
                        control_item["ci90"][0]
                        > -INFORMATION_EQUIVALENCE_MARGIN_BITS
                        and control_item["ci90"][1]
                        < INFORMATION_EQUIVALENCE_MARGIN_BITS
                    ),
                }
            )
            label_null.append(control_item)

            random_gain = _two_action_channel_gain(
                branches, candidate, source_half, target_half
            )
            random_item = _bootstrap_summary(
                random_gain,
                spec,
                f"dose_channel_random/{candidate}/{source_half}_to_{target_half}",
                arrays,
            )
            random_item.update(
                {
                    "candidate": candidate,
                    "direction": f"{source_half}_to_{target_half}",
                    "control": "random_swap_vs_noop",
                    "margin_bits": INFORMATION_EQUIVALENCE_MARGIN_BITS,
                    "equivalent": bool(
                        random_item["ci90"][0]
                        > -INFORMATION_EQUIVALENCE_MARGIN_BITS
                        and random_item["ci90"][1]
                        < INFORMATION_EQUIVALENCE_MARGIN_BITS
                    ),
                }
            )
            random_null.append(random_item)
    _adjust_family(primary)
    return primary, label_null, random_null, models


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    if np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return 0.0
    value = spearmanr(x[valid], y[valid]).statistic
    return float(value) if np.isfinite(value) else 0.0


def _sensor_matrix_correlations(
    sensor: pd.DataFrame,
    branches: pd.DataFrame,
    candidate: str,
    profile: str,
    kind: str,
    source_half: str | None = None,
) -> pd.Series:
    selected = sensor[
        (sensor["candidate"] == candidate)
        & (sensor["profile"] == profile)
        & (sensor["arm"].isin(("Q00", "Q100")))
        & sensor["valid"].astype(bool)
    ]
    if kind == "reliability":
        wide = selected.pivot_table(
            index=["matrix_id", "state_id", "arm"],
            columns="source_half",
            values="temporal_value",
            aggfunc="first",
        ).dropna(subset=["A", "B"])
        return wide.groupby(level="matrix_id").apply(
            lambda frame: _safe_spearman(frame["A"], frame["B"]),
            include_groups=False,
        )
    if source_half is None:
        raise ValueError("forecast correlation requires a source half")
    target_half = "B" if source_half == "A" else "A"
    score = selected[selected["source_half"] == source_half][
        ["matrix_id", "state_id", "arm", "temporal_value"]
    ]
    outcome = (
        branches[
            (branches["candidate"] == candidate)
            & (branches["half"] == target_half)
            & (branches["arm"].isin(("Q00", "Q100")))
        ]
        .groupby(["matrix_id", "state_id", "arm"], sort=True)["primary"]
        .mean()
        .rename("renewal")
        .reset_index()
    )
    joined = score.merge(outcome, on=["matrix_id", "state_id", "arm"])
    return joined.groupby("matrix_id").apply(
        lambda frame: _safe_spearman(frame["temporal_value"], frame["renewal"]),
        include_groups=False,
    )


def _sensor_analysis(
    sensor: pd.DataFrame,
    branches: pd.DataFrame,
    spec: PX11Spec,
    arrays: dict[str, NDArray],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str | None,
]:
    if sensor.empty:
        return [], [], [], [], None
    responses: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    forecasts: list[dict[str, Any]] = []
    dose_rows: list[dict[str, Any]] = []
    profiles = tuple(dict.fromkeys(sensor["profile"].astype(str)))
    response_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    profile_pass: dict[str, bool] = {}
    for profile in profiles:
        profile_response: list[dict[str, Any]] = []
        profile_reliability: list[dict[str, Any]] = []
        profile_forecast: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            for half in _halves(spec):
                local = sensor[
                    (sensor["candidate"] == candidate)
                    & (sensor["profile"] == profile)
                    & (sensor["source_half"] == half)
                    & (sensor["arm"].isin(("Q00", "Q100")))
                    & sensor["valid"].astype(bool)
                ]
                means = (
                    local.groupby(["matrix_id", "arm"])["temporal_value"]
                    .mean()
                    .unstack("arm")
                )
                effect = means.get("Q100", pd.Series(dtype=float)) - means.get(
                    "Q00", pd.Series(dtype=float)
                )
                item = _bootstrap_summary(
                    effect,
                    spec,
                    f"sensor_response/{profile}/{candidate}/{half}",
                    arrays,
                )
                item.update(
                    {
                        "profile": profile,
                        "candidate": candidate,
                        "source_half": half,
                    }
                )
                responses.append(item)
                profile_response.append(item)
                response_lookup[(profile, candidate, half)] = item

            values = _sensor_matrix_correlations(
                sensor, branches, candidate, profile, "reliability"
            )
            item = _bootstrap_summary(
                values,
                spec,
                f"sensor_reliability/{profile}/{candidate}",
                arrays,
            )
            item.update({"profile": profile, "candidate": candidate})
            reliability.append(item)
            profile_reliability.append(item)

            for source_half in _halves(spec):
                values = _sensor_matrix_correlations(
                    sensor,
                    branches,
                    candidate,
                    profile,
                    "forecast",
                    source_half,
                )
                item = _bootstrap_summary(
                    values,
                    spec,
                    f"sensor_forecast/{profile}/{candidate}/{source_half}",
                    arrays,
                )
                item.update(
                    {
                        "profile": profile,
                        "candidate": candidate,
                        "source_half": source_half,
                        "target_half": "B" if source_half == "A" else "A",
                    }
                )
                forecasts.append(item)
                profile_forecast.append(item)
        _adjust_family(profile_response)
        _adjust_family(profile_reliability)
        _adjust_family(profile_forecast)

    full_effects = {
        (candidate, half): response_lookup.get(("FULL64", candidate, half))
        for candidate in CANDIDATES
        for half in _halves(spec)
    }
    for item in responses:
        full = full_effects[(item["candidate"], item["source_half"])]
        denominator = abs(float(full["effect"])) if full else float("nan")
        item["full_effect_retention"] = (
            abs(float(item["effect"])) / denominator
            if np.isfinite(denominator) and denominator > 1e-12
            else float("nan")
        )
    for profile in profiles:
        local_response = [item for item in responses if item["profile"] == profile]
        local_reliability = [
            item for item in reliability if item["profile"] == profile
        ]
        local_forecast = [item for item in forecasts if item["profile"] == profile]
        profile_pass[profile] = bool(
            len(local_response) == 4
            and len(local_reliability) == 2
            and len(local_forecast) == 4
            and all(item.get("pass", False) for item in local_response)
            and all(
                item["full_effect_retention"] >= SENSOR_RETENTION_FRACTION
                for item in local_response
            )
            and all(item.get("pass", False) for item in local_reliability)
            and all(item.get("pass", False) for item in local_forecast)
        )
    if spec.stage == "confirmation":
        selected_profile = (
            spec.sensor_profile if profile_pass.get(spec.sensor_profile, False) else None
        )
    else:
        selected_profile = next(
            (
                profile
                for profile in SENSOR_SELECTION_ORDER
                if profile in profile_pass and profile_pass[profile]
            ),
            None,
        )

    dose_profile = "FULL64" if spec.stage != "confirmation" else spec.sensor_profile
    dose_sensor = sensor[
        (sensor["profile"] == dose_profile)
        & (sensor["arm"].isin(QUANTILE_ARMS))
        & sensor["valid"].astype(bool)
    ]
    outcomes = (
        branches[branches["arm"].isin(QUANTILE_ARMS)]
        .groupby(["matrix_id", "candidate", "state_id", "half", "arm"])[
            "primary"
        ]
        .mean()
        .rename("renewal")
        .reset_index()
    )
    for candidate in CANDIDATES:
        for source_half in _halves(spec):
            target_half = "B" if source_half == "A" else "A"
            local_score = dose_sensor[
                (dose_sensor["candidate"] == candidate)
                & (dose_sensor["source_half"] == source_half)
            ][["matrix_id", "state_id", "arm", "temporal_value"]]
            local_outcome = outcomes[
                (outcomes["candidate"] == candidate)
                & (outcomes["half"] == target_half)
            ]
            joined = local_score.merge(
                local_outcome[["matrix_id", "state_id", "arm", "renewal"]],
                on=["matrix_id", "state_id", "arm"],
            )
            values = joined.groupby("matrix_id").apply(
                lambda frame: _safe_spearman(
                    frame["temporal_value"], frame["renewal"]
                ),
                include_groups=False,
            )
            item = _bootstrap_summary(
                values,
                spec,
                f"sensor_dose/{dose_profile}/{candidate}/{source_half}",
                arrays,
            )
            item.update(
                {
                    "profile": dose_profile,
                    "candidate": candidate,
                    "source_half": source_half,
                    "target_half": target_half,
                }
            )
            dose_rows.append(item)
    _adjust_family(dose_rows)
    return responses, reliability, forecasts, dose_rows, selected_profile


def analyze_batches(
    batches: Sequence[PX11Batch],
    spec: PX11Spec,
    frozen_dose_models: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    arrays: dict[str, NDArray] = {}
    acquisitions = pd.DataFrame(
        [row for batch in batches for row in batch.acquisition_rows]
    )
    interventions = pd.DataFrame(
        [row for batch in batches for row in batch.intervention_rows]
    )
    branches = pd.DataFrame(
        [row for batch in batches for row in batch.branch_rows]
    )
    atoms = pd.DataFrame([row for batch in batches for row in batch.atom_rows])
    sensor = pd.DataFrame([row for batch in batches for row in batch.sensor_rows])
    derived_atoms = _derive_atom_scores(atoms)

    eligibility: dict[str, Any] = {}
    minimum = (
        MINIMUM_ELIGIBLE_PILOT
        if spec.stage in {"pilot", "smoke"}
        else MINIMUM_ELIGIBLE_CONFIRMATION
    )
    for candidate in CANDIDATES:
        local = acquisitions[
            (acquisitions["candidate"] == candidate)
            & (acquisitions["eligible"] == 1)
        ]
        matrices = int(local["matrix_id"].nunique())
        eligibility[candidate] = {
            "eligible_states": int(len(local)),
            "eligible_matrices": matrices,
            "minimum": min(minimum, spec.matrices),
            "pass": matrices >= min(minimum, spec.matrices),
        }
    eligibility_pass = all(item["pass"] for item in eligibility.values())

    outcome, outcome_controls, outcome_dose = _outcome_analysis(
        branches, spec, arrays
    )
    iri, attenuation, atom_groups = _atom_analysis(
        derived_atoms, spec, arrays
    )
    channel, channel_labels, channel_random, dose_models = _dose_channel_analysis(
        branches, spec, arrays, frozen_dose_models
    )
    (
        sensor_response,
        sensor_reliability,
        sensor_forecast,
        sensor_dose,
        selected_sensor,
    ) = _sensor_analysis(sensor, branches, spec, arrays)

    family_gates: dict[str, Any] = {}
    advancing_families: list[str] = []
    for family in CONTRASTS:
        causal = [item for item in outcome if item["family"] == family]
        iri_local = [item for item in iri if item["family"] == family]
        control_name = "block_random" if family == "beta_surgery" else "random_swap"
        controls = [
            item for item in outcome_controls if item["control"] == control_name
        ]
        causal_pass = bool(
            len(causal) == 4
            and all(item.get("pass", False) for item in causal)
            and len(controls) == 4
            and all(item["equivalent"] for item in controls)
        )
        iri_sign_stable = bool(
            len(iri_local) == 4
            and all(float(item["effect"]) > 0.0 for item in iri_local)
            and all(
                float(item["minimum_leave_one_out_aligned"]) > 0.0
                for item in iri_local
            )
        )
        iri_confirmed = bool(
            len(iri_local) == 4 and all(item.get("pass", False) for item in iri_local)
        )
        advances = bool(
            spec.stage == "pilot"
            and eligibility_pass
            and causal_pass
            and iri_sign_stable
        )
        if advances:
            advancing_families.append(family)
        family_gates[family] = {
            "causal_manipulation": causal_pass,
            "iri_sign_leave_one_out_stable": iri_sign_stable,
            "iri_confirmed": iri_confirmed,
            "advances_from_pilot": advances,
        }

    channel_coefficients_positive = bool(
        len(dose_models) == 4
        and all(float(item["coefficient"]) > 0.0 for item in dose_models.values())
    )
    channel_gains_positive = bool(
        len(channel) == 4 and all(float(item["effect"]) > 0.0 for item in channel)
    )
    channel_confirmed = bool(
        len(channel) == 4
        and all(item.get("pass", False) for item in channel)
        and all(item["equivalent"] for item in channel_labels)
        and all(item["equivalent"] for item in channel_random)
    )
    channel_advances = bool(
        spec.stage == "pilot"
        and eligibility_pass
        and channel_coefficients_positive
        and channel_gains_positive
    )
    common_fingerprint = bool(
        eligibility_pass
        and all(
            family_gates[family]["causal_manipulation"]
            and family_gates[family]["iri_confirmed"]
            for family in CONTRASTS
        )
    )
    if common_fingerprint:
        fingerprint_classification = "common_three_knob_causal_redistribution"
    elif all(
        family_gates[family]["causal_manipulation"]
        and family_gates[family]["iri_confirmed"]
        for family in ("model", "physical_rule")
    ):
        fingerprint_classification = "molecularly_shared_redistribution"
    elif (
        family_gates["model"]["causal_manipulation"]
        and family_gates["model"]["iri_confirmed"]
    ):
        fingerprint_classification = "predictor_specific_redistribution"
    else:
        fingerprint_classification = "causal_redistribution_unconfirmed"

    gates = {
        "eligibility": bool(eligibility_pass),
        "family_gates": family_gates,
        "fingerprint_classification": fingerprint_classification,
        "common_causal_redistribution": common_fingerprint,
        "dose_channel_coefficients_positive": channel_coefficients_positive,
        "dose_channel_developmental_gain_positive": channel_gains_positive,
        "dose_channel_confirmed": channel_confirmed,
        "dose_channel_advances": channel_advances,
        "selected_sensor_profile": selected_sensor,
        "sensor_advances": bool(spec.stage == "pilot" and selected_sensor),
        "public_phi_r_can_win": False,
        "automatic_confirmation_authorized": False,
    }
    metrics = {
        "format": "codex-ch5-phir-px11-primary-metrics-v1",
        "stage": spec.stage,
        "eligibility": eligibility,
        "outcome_contrasts": outcome,
        "outcome_controls": outcome_controls,
        "outcome_dose": outcome_dose,
        "iri_contrasts": iri,
        "iri_attenuation": attenuation,
        "atom_group_contrasts": atom_groups,
        "dose_channel": channel,
        "dose_channel_label_controls": channel_labels,
        "dose_channel_random_control": channel_random,
        "dose_models": dose_models,
        "sensor_response": sensor_response,
        "sensor_reliability": sensor_reliability,
        "sensor_forecast": sensor_forecast,
        "sensor_dose": sensor_dose,
        "pilot_advancement": {
            "advancing_families": advancing_families,
            "dose_channel": channel_advances,
            "sensor_profile": selected_sensor,
            "manual_review_required": True,
        },
        "gates": gates,
    }
    tables = {
        "acquisition": acquisitions,
        "selected_interventions": interventions,
        "branches": branches,
        "atom_scores": atoms,
        "derived_atom_scores": derived_atoms,
        "sensor_scores": sensor,
    }
    return metrics, tables, arrays


def _metric_table(
    items: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> list[str]:
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for item in items:
        values: list[str] = []
        for column in columns:
            value = item.get(column, "")
            if isinstance(value, float):
                value = f"{value:+.6f}"
            values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return rows


def _reports(
    metrics: Mapping[str, Any], registration_id: str
) -> tuple[str, str]:
    gates = metrics["gates"]
    advancement = metrics["pilot_advancement"]
    scientific = [
        f"# PX11 {metrics['stage']} causal information redistribution",
        "",
        f"Registration: `{registration_id}`.",
        "",
        "## Status and classifications",
        "",
        f"- Eligibility: **{gates['eligibility']}**",
        f"- Fingerprint: **{gates['fingerprint_classification']}**",
        f"- Shared dose channel confirmed: **{gates['dose_channel_confirmed']}**",
        f"- Selected compact sensor: **{gates['selected_sensor_profile']}**",
        "- Public Phi-r can win: **False**",
        "- Automatic confirmation: **False**",
        "",
        "## Renewal manipulation",
        "",
        *_metric_table(
            metrics["outcome_contrasts"],
            (
                "family",
                "candidate",
                "source_half",
                "effect",
                "ci95",
                "holm_adjusted_p",
                "pass",
            ),
        ),
        "",
        "## Information Redistribution Index",
        "",
        *_metric_table(
            metrics["iri_contrasts"],
            (
                "family",
                "candidate",
                "source_half",
                "effect",
                "ci95",
                "holm_adjusted_p",
                "pass",
            ),
        ),
        "",
        "## Shared molecular-dose channel",
        "",
        *_metric_table(
            metrics["dose_channel"],
            (
                "candidate",
                "direction",
                "coefficient",
                "effect",
                "ci95",
                "holm_adjusted_p",
                "pass",
            ),
        ),
        "",
        "## Pilot advancement",
        "",
        f"- Intervention families: `{advancement['advancing_families']}`",
        f"- Dose channel: `{advancement['dose_channel']}`",
        f"- Sensor profile: `{advancement['sensor_profile']}`",
        "- A human review and a separate confirmation seal are required.",
        "",
        "## Claim boundary",
        "",
        "IRI is an empirical atom fingerprint, not Phi-r or integrated information. PX11 cannot establish consciousness, life, agency, biological memory, a universal origin-of-life mechanism, Platonic space, or the Ruliad.",
    ]
    if gates["common_causal_redistribution"]:
        fingerprint = (
            "All three independent control knobs produced the same registered "
            "short-timescale information-redistribution pattern."
        )
    elif gates["fingerprint_classification"] != "causal_redistribution_unconfirmed":
        fingerprint = (
            "Part of the proposed information-redistribution pattern replicated, "
            "but it was not common to all three control knobs."
        )
    else:
        fingerprint = (
            "The proposed information-redistribution pattern was not confirmed "
            "across the registered controls."
        )
    lay = [
        f"# Lay summary — PX11 {metrics['stage']}",
        "",
        "PX11 starts several versions of the same already-broken assembly and nudges recovery in three different ways: a predictor-chosen molecule swap, a simple physical molecule rule, or a direct change to the occupied catalytic web.",
        "",
        fingerprint,
        "The separate dose test asks whether one simple dial can predict how strongly molecular edits change recovery. The sensor test asks how much of the temporal signal survives when we observe fewer branches or less of each assembly.",
        "",
        "This is a staged pilot. It does not rescue the published Phi-r score, and the 48-matrix confirmation is not authorized automatically.",
    ]
    return "\n".join(scientific) + "\n", "\n".join(lay) + "\n"


def validate(output: Path = DEFAULT_VALIDATION) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append(
            {"name": name, "pass": bool(passed), "detail": _json_ready(detail)}
        )

    proto = protocol()
    check(
        "staged_shapes",
        PILOT_MATRICES == 24
        and CONFIRMATION_MATRICES == 48
        and BRANCHES == 128
        and HORIZON == 8,
    )
    check("candidate_separation", tuple(CANDIDATES) == ("02", "03"))
    check("manual_confirmation", not proto["cohorts"]["automatic_confirmation"])
    check("arm_panel", tuple(proto["arms"]["ordered"]) == ARMS)
    check("future_seed_arm_free", proto["randomness"]["future_seed_excludes_arm"])
    check("fresh_namespace", SEED_DOMAINS != px10.SEED_DOMAINS)
    check("source_files", all((ROOT / name).exists() for name in SOURCE_FILES))
    check("frozen_models", MODEL_SOURCE.exists() and MODEL_CONTRACT_SOURCE.exists())
    check("px10_seals", PX10_REGISTRATION.exists() and PX10_CALIBRATION.exists())

    rng = np.random.default_rng(_seed(smoke_spec(), "validation", "fixture"))
    config = GardConfig()
    beta = generate_beta(config, rng)
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[:40] = 1
    snapshot = Snapshot(composition, 20, (False,), (0.8,))
    case = px9.ResilienceCase(
        "PX11-validation",
        "02",
        0,
        20,
        beta,
        snapshot,
        np.tile(composition, (32, 1)).astype(np.int16),
    )
    students = cr5.load_students(MODEL_SOURCE, MODEL_CONTRACT_SOURCE)
    plans = _select_arms(case, students, smoke_spec())
    check("all_arms_selected", tuple(item.arm for item in plans) == ARMS)
    check(
        "all_molecular_edits_legal",
        all(
            item.edit is None
            or (
                item.edit.remove_type != item.edit.add_type
                and composition[item.edit.remove_type] >= 1
                and int(apply_molecular_edit(composition, item.edit).sum())
                == int(composition.sum())
            )
            for item in plans
        ),
    )
    plan_by_name = {item.arm: item for item in plans}
    outgoing = outgoing_catalytic_influence(composition, beta)
    stable = plan_by_name["RULE_STABILIZE"].edit
    unstable = plan_by_name["RULE_DESTABILIZE"].edit
    assert stable is not None and unstable is not None
    stable_delta = outgoing[stable.add_type] - outgoing[stable.remove_type]
    unstable_delta = outgoing[unstable.add_type] - outgoing[unstable.remove_type]
    check("outgoing_orientation", np.array_equal(outgoing, composition / 40 @ beta))
    check("physical_extrema_direction", stable_delta > unstable_delta)
    tighten = plan_by_name["TIGHTEN"].surgery
    loosen = plan_by_name["LOOSEN"].surgery
    neutral = plan_by_name["BLOCK_RANDOM"].surgery
    assert tighten is not None and loosen is not None and neutral is not None
    present = np.flatnonzero(composition > 0)
    block = beta[np.ix_(present, present)]
    check(
        "multiplicative_surgery_exact",
        np.array_equal(
            tighten.beta[np.ix_(present, present)], block * 1.5
        )
        and np.array_equal(
            loosen.beta[np.ix_(present, present)], block * (1.0 / 1.5)
        ),
    )
    check(
        "neutral_surgery_exact_norm",
        np.isclose(neutral.observed_norm, 0.5 * np.linalg.norm(block), atol=1e-10),
    )
    check(
        "neutral_surgery_throughput",
        np.isclose(
            p3c.catalytic_throughput(composition, neutral.beta),
            p3c.catalytic_throughput(composition, beta),
            atol=1e-10,
            rtol=1e-12,
        ),
    )

    sample_atoms = {f"atom_{name}": 0.0 for name in ATOM_NAMES}
    sample_atoms.update(
        {
            "atom_s_to_u0": 0.2,
            "atom_s_to_u1": 0.3,
            "atom_u0_to_s": 0.4,
            "atom_u1_to_s": 0.1,
            "atom_r_to_r": -0.2,
            "atom_s_to_s": -0.1,
            "atom_u0_to_u1": -0.05,
            "atom_u1_to_u0": -0.04,
        }
    )
    groups = {
        key: sum(sample_atoms[column] for column in columns)
        for key, columns in GROUP_COLUMNS.items()
    }
    iri = (
        groups["downward_routing"]
        + groups["upward_integration"]
        - groups["redundant_persistence"]
        - groups["synergy_persistence"]
        - groups["cross_part_transfer"]
    )
    swapped = dict(sample_atoms)
    for left, right in (
        ("atom_s_to_u0", "atom_s_to_u1"),
        ("atom_u0_to_s", "atom_u1_to_s"),
        ("atom_u0_to_u1", "atom_u1_to_u0"),
    ):
        swapped[left], swapped[right] = swapped[right], swapped[left]
    swapped_groups = {
        key: sum(swapped[column] for column in columns)
        for key, columns in GROUP_COLUMNS.items()
    }
    swapped_iri = (
        swapped_groups["downward_routing"]
        + swapped_groups["upward_integration"]
        - swapped_groups["redundant_persistence"]
        - swapped_groups["synergy_persistence"]
        - swapped_groups["cross_part_transfer"]
    )
    check("iri_half_label_symmetric", iri == swapped_iri)

    first_mask = _coordinate_mask(case, SENSOR_BY_NAME["COORD25"], smoke_spec())
    second_mask = _coordinate_mask(case, SENSOR_BY_NAME["COORD25"], smoke_spec())
    check(
        "sensor_mask_deterministic",
        np.array_equal(first_mask, second_mask) and first_mask.sum() == 25,
    )
    synthetic: list[dict[str, Any]] = []
    for state in range(10):
        for arm_index, arm in enumerate((*QUANTILE_ARMS, "NOOP")):
            for branch in range(16):
                probability = 0.1 + 0.1 * arm_index
                synthetic.append(
                    {
                        "state_id": f"s{state}",
                        "matrix_id": state // 2,
                        "candidate": "02",
                        "arm": arm,
                        "half": "A",
                        "primary": int(rng.random() < probability),
                        "predicted_shift": float(arm_index),
                    }
                )
    fitted = _fit_dose_coefficient(pd.DataFrame(synthetic))
    check(
        "shared_dose_fit",
        fitted["converged"] and fitted["coefficient"] > 0.0,
        fitted,
    )
    check(
        "sensor_profiles_frozen",
        set(SENSOR_SELECTION_ORDER) == set(SENSOR_BY_NAME)
        and SENSOR_SELECTION_ORDER[0] == "MINIMAL"
        and SENSOR_SELECTION_ORDER[-1] == "FULL64",
    )
    check("public_phi_r_nonwinning", not proto["public_phi_r_can_win"])

    payload = {
        "format": "codex-ch5-phir-px11-validation-v1",
        "checks": checks,
        "passed": bool(checks and all(item["pass"] for item in checks)),
        "scientific_matrices": 0,
        "scientific_effects_disclosed": False,
    }
    output.mkdir(parents=True)
    _atomic_json(output / "validation.json", payload)
    write_checksums(output)
    if not payload["passed"]:
        failed = [item["name"] for item in checks if not item["pass"]]
        raise AssertionError(f"PX11 validation failed: {failed}")
    return payload


def register(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    if directory.exists():
        raise FileExistsError(directory)
    for forbidden in (
        DEFAULT_SMOKE,
        DEFAULT_PILOT,
        DEFAULT_CONFIRMATION_REGISTRATION,
        DEFAULT_CONFIRMATION,
    ):
        if forbidden.exists():
            raise FileExistsError(f"pre-registration PX11 artifact exists: {forbidden}")
    verify_checksums(DEFAULT_VALIDATION)
    validation = json.loads((DEFAULT_VALIDATION / "validation.json").read_text())
    if not validation["passed"]:
        raise ValueError("PX11 validation has not passed")
    px10_registration = px10.verify_registration(PX10_REGISTRATION)
    verify_checksums(PX10_CALIBRATION)
    calibration = json.loads((PX10_CALIBRATION / "calibration.json").read_text())
    proto = protocol()
    source_hashes = _source_hashes()
    model_contract = {
        "format": "codex-ch5-phir-px11-model-contract-v1",
        "renewal_student_sha256": sha256_file(MODEL_SOURCE),
        "renewal_contract_sha256": sha256_file(MODEL_CONTRACT_SOURCE),
        "px10_registration_id": px10_registration["registration_id"],
        "px10_calibration_sha256": sha256_file(
            PX10_CALIBRATION / "calibration.json"
        ),
        "atom_instrument_eligible": bool(
            calibration["gates"]["atom_instrument_eligible"]
        ),
        "refit_before_pilot": False,
    }
    seed_registry = {
        "format": "codex-ch5-phir-px11-seed-registry-v1",
        "label": LABEL,
        "domains": SEED_DOMAINS,
        "disjoint_from_px10": set(SEED_DOMAINS.values()).isdisjoint(
            px10.SEED_DOMAINS.values()
        ),
    }
    registration_id = _digest(
        {
            "protocol": proto,
            "sources": source_hashes,
            "models": model_contract,
            "seeds": seed_registry,
            "validation": validation,
        }
    )
    registration = {
        "format": REGISTRATION_FORMAT,
        "registration_id": registration_id,
        "protocol_id": proto["protocol_id"],
        "source_hashes": source_hashes,
        "model_contract": model_contract,
        "validation_sha256": sha256_file(DEFAULT_VALIDATION / "validation.json"),
        "registered_at_unix": time.time(),
        "scientific_matrices_generated": False,
        "confirmation_authorized": False,
    }
    directory.mkdir(parents=True)
    shutil.copy2(ROOT / DOCUMENT, directory / "preregistration.md")
    _atomic_json(directory / "protocol.json", proto)
    _atomic_json(directory / "seed_registry.json", seed_registry)
    _atomic_json(directory / "model_contract.json", model_contract)
    _atomic_json(directory / "registration.json", registration)
    write_checksums(directory)
    _append_ledger(
        f"<!-- phir-extension-px11-registration-{registration_id} -->",
        [
            "## Phi-r extension PX11 registered",
            "",
            f"- Registration: `{registration_id}`.",
            "- A fresh 24-matrix, three-knob post-break pilot and a manually gated disjoint 48-matrix confirmation were sealed.",
            "- IRI is an empirical atom fingerprint and cannot be called Phi-r.",
            "- No PX11 scientific matrix or outcome existed at registration.",
        ],
    )
    return registration


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    verify_checksums(directory)
    registration = json.loads((directory / "registration.json").read_text())
    if registration["format"] != REGISTRATION_FORMAT:
        raise ValueError("unexpected PX11 registration format")
    for name, expected in registration["source_hashes"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"PX11 sealed source changed: {name}")
    contract = registration["model_contract"]
    if sha256_file(MODEL_SOURCE) != contract["renewal_student_sha256"]:
        raise ValueError("PX11 renewal student changed")
    if sha256_file(MODEL_CONTRACT_SOURCE) != contract["renewal_contract_sha256"]:
        raise ValueError("PX11 renewal model contract changed")
    if sha256_file(PX10_CALIBRATION / "calibration.json") != contract[
        "px10_calibration_sha256"
    ]:
        raise ValueError("PX11 inherited calibration changed")
    return registration


def smoke(output: Path = DEFAULT_SMOKE) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    registration = verify_registration()
    spec = smoke_spec()
    argument = (2, spec, str(MODEL_SOURCE), str(MODEL_CONTRACT_SOURCE))
    first = _run_matrix(argument)
    second = _run_matrix(argument)
    payload = {
        "format": "codex-ch5-phir-px11-smoke-v1",
        "registration_id": registration["registration_id"],
        "artificial_seed_domain": True,
        "scientific_matrices": 0,
        "exact_replay": first.scientific_digest == second.scientific_digest,
        "all_arms_exercised": {row["arm"] for row in first.intervention_rows}
        == set(ARMS),
        "both_candidates_exercised": {
            row["candidate"] for row in first.intervention_rows
        }
        == set(CANDIDATES),
        "branch_rows_created": len(first.branch_rows),
        "atom_rows_created": len(first.atom_rows),
        "sensor_rows_created": len(first.sensor_rows),
        "effect_sizes_disclosed": False,
    }
    payload["passed"] = bool(
        payload["exact_replay"]
        and payload["all_arms_exercised"]
        and payload["both_candidates_exercised"]
        and payload["branch_rows_created"] > 0
        and payload["atom_rows_created"] > 0
        and payload["sensor_rows_created"] > 0
    )
    output.mkdir(parents=True)
    _atomic_json(output / "smoke.json", payload)
    write_checksums(output)
    if not payload["passed"]:
        raise AssertionError("PX11 smoke failed")
    return payload


def _checkpoint_contract(
    registration_id: str, spec: PX11Spec
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "spec": _json_ready(spec.__dict__),
        "renewal_student_sha256": sha256_file(MODEL_SOURCE),
    }


def _status_write(work: Path, payload: Mapping[str, Any]) -> None:
    _atomic_json(
        work / "status.json",
        {"format": STATUS_FORMAT, **payload, "updated_at_unix": time.time()},
    )


class _PX11Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name in {
            "PX11Batch",
            "PX11Spec",
            "CasePayload",
            "ArmPlan",
            "SensorProfile",
        }:
            return globals()[name]
        return super().find_class(module, name)


def _load_checkpoint(path: Path) -> PX11Batch:
    with path.open("rb") as handle:
        value = _PX11Unpickler(handle).load()
    if not isinstance(value, PX11Batch):
        raise TypeError(f"unexpected PX11 checkpoint type: {path}")
    return value


def _prepare_work(
    work: Path, registration_id: str, spec: PX11Spec
) -> None:
    work.mkdir(parents=True, exist_ok=True)
    expected = _checkpoint_contract(registration_id, spec)
    path = work / "checkpoint_contract.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("PX11 checkpoint contract mismatch")
    else:
        _atomic_json(path, expected)


def _run_checkpoint_stage(
    work: Path,
    spec: PX11Spec,
    workers: int,
    stage: str,
    cpu_budget_seconds: float,
    prior_cpu_seconds: float = 0.0,
) -> list[PX11Batch]:
    directory = work / stage
    directory.mkdir(parents=True, exist_ok=True)
    batches: dict[int, PX11Batch] = {}
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:03d}.pkl"
        if path.exists():
            batch = _load_checkpoint(path)
            if batch.matrix_id != matrix_id:
                raise ValueError("PX11 checkpoint matrix mismatch")
            batches[matrix_id] = batch
    consumed = prior_cpu_seconds + sum(
        batch.cpu_seconds for batch in batches.values()
    )
    _status_write(
        work,
        {
            "state": "running",
            "stage": stage,
            "completed_matrices": len(batches),
            "total_matrices": spec.matrices,
            "cpu_seconds": consumed,
        },
    )
    pending = [
        matrix_id for matrix_id in range(spec.matrices) if matrix_id not in batches
    ]
    arguments = [
        (matrix_id, spec, str(MODEL_SOURCE), str(MODEL_CONTRACT_SOURCE))
        for matrix_id in pending
    ]

    def retain(batch: PX11Batch, matrix_id: int) -> None:
        nonlocal consumed
        _atomic_pickle(directory / f"matrix_{matrix_id:03d}.pkl", batch)
        batches[matrix_id] = batch
        consumed += batch.cpu_seconds
        _status_write(
            work,
            {
                "state": "running",
                "stage": stage,
                "completed_matrices": len(batches),
                "total_matrices": spec.matrices,
                "cpu_seconds": consumed,
            },
        )

    if workers == 1:
        for argument in arguments:
            retain(_run_matrix(argument), argument[0])
            if consumed > cpu_budget_seconds:
                raise TimeoutError("PX11 CPU budget reached after checkpoint")
    elif arguments:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_matrix, argument): argument[0]
                for argument in arguments
            }
            for future in as_completed(futures):
                matrix_id = futures[future]
                retain(future.result(), matrix_id)
                if consumed > cpu_budget_seconds:
                    for item in futures:
                        item.cancel()
                    raise TimeoutError("PX11 CPU budget reached after checkpoint")
    if len(batches) != spec.matrices:
        raise AssertionError("PX11 checkpoint stage incomplete")
    return [batches[matrix_id] for matrix_id in range(spec.matrices)]


def _stage_settings(
    stage: str,
) -> tuple[PX11Spec, Path, Path, Path, dict[str, Any] | None]:
    if stage == "pilot":
        return pilot_spec(), DEFAULT_PILOT, DEFAULT_PILOT_WORK, DEFAULT_PILOT_LOG, None
    if stage != "confirmation":
        raise ValueError(f"unknown PX11 stage: {stage}")
    confirmation = verify_confirmation_registration()
    contract = confirmation["confirmation_contract"]
    return (
        confirmation_spec(contract),
        DEFAULT_CONFIRMATION,
        DEFAULT_CONFIRMATION_WORK,
        DEFAULT_CONFIRMATION_LOG,
        contract,
    )


def run(
    stage: str,
    *,
    output: Path | None = None,
    work: Path | None = None,
    workers: int = MAX_WORKERS,
    cpu_budget_hours: float | None = None,
) -> dict[str, Any]:
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"PX11 workers must be in [1,{MAX_WORKERS}]")
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    spec, default_output, default_work, _log, confirmation_contract = (
        _stage_settings(stage)
    )
    output = default_output if output is None else output
    work = default_work if work is None else work
    if output.exists():
        raise FileExistsError(output)
    if stage == "confirmation":
        pilot_manifest = verify_result(DEFAULT_PILOT)
        prior_cpu = (
            float(pilot_manifest["generation_cpu_seconds"])
            + float(pilot_manifest["replay_cpu_seconds"])
        ) / 3600.0
        remaining = MAX_TOTAL_CPU_HOURS - prior_cpu
        if remaining <= 0.0:
            raise TimeoutError("PX11 pilot exhausted the complete CPU budget")
        if cpu_budget_hours is None:
            cpu_budget_hours = min(DEFAULT_CONFIRMATION_CPU_HOURS, remaining)
        if cpu_budget_hours > remaining + 1e-12:
            raise ValueError("PX11 confirmation budget exceeds total program budget")
    elif cpu_budget_hours is None:
        cpu_budget_hours = DEFAULT_PILOT_CPU_HOURS
    assert cpu_budget_hours is not None
    _prepare_work(work, registration["registration_id"], spec)
    if shutil.disk_usage(work).free < MINIMUM_FREE_DISK_BYTES:
        raise OSError("PX11 work volume lacks required free space")
    started = time.time()
    budget_seconds = cpu_budget_hours * 3600.0
    try:
        generated = _run_checkpoint_stage(
            work, spec, workers, "generation", budget_seconds
        )
        generation_cpu = float(sum(batch.cpu_seconds for batch in generated))
        replayed = _run_checkpoint_stage(
            work,
            spec,
            workers,
            "replay",
            budget_seconds,
            generation_cpu,
        )
        replay_cpu = float(sum(batch.cpu_seconds for batch in replayed))
        replay_rows = [
            {
                "matrix_id": left.matrix_id,
                "generation_digest": left.scientific_digest,
                "replay_digest": right.scientific_digest,
                "exact": left.scientific_digest == right.scientific_digest,
            }
            for left, right in zip(generated, replayed, strict=True)
        ]
        replay_audit = {
            "format": "codex-ch5-phir-px11-replay-v1",
            "matrices": replay_rows,
            "complete_exact_replay": bool(
                len(replay_rows) == spec.matrices
                and all(item["exact"] for item in replay_rows)
            ),
        }
        if not replay_audit["complete_exact_replay"]:
            raise AssertionError("PX11 exact replay failed")
        frozen_models = (
            None
            if confirmation_contract is None
            else confirmation_contract["dose_models"]
        )
        metrics, tables, arrays = analyze_batches(
            generated, spec, frozen_models
        )
        staging = output.with_name(f".{output.name}.staging")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _atomic_json(staging / "primary_metrics.json", metrics)
        _atomic_json(staging / "replay_audit.json", replay_audit)
        _atomic_json(staging / "dose_model.json", metrics["dose_models"])
        for name, table in tables.items():
            table.to_csv(
                staging / f"{name}.csv.gz", index=False, compression="gzip"
            )
        np.savez_compressed(staging / "inference_arrays.npz", **arrays)
        scientific_report, lay_summary = _reports(
            metrics, registration["registration_id"]
        )
        (staging / "SCIENTIFIC_REPORT.md").write_text(scientific_report)
        (staging / "LAY_SUMMARY.md").write_text(lay_summary)
        claim_boundaries = {
            "supported": (
                [
                    metrics["gates"]["fingerprint_classification"],
                    "dose_channel" if metrics["gates"]["dose_channel_confirmed"] else None,
                    (
                        f"sensor_{metrics['gates']['selected_sensor_profile']}"
                        if metrics["gates"]["selected_sensor_profile"]
                        else None
                    ),
                ]
            ),
            "failed_or_unresolved": [
                family
                for family, value in metrics["gates"]["family_gates"].items()
                if not value["iri_confirmed"]
            ],
            "prohibited": protocol()["claim_boundary"]["prohibited"],
        }
        _atomic_json(staging / "claim_boundaries.json", claim_boundaries)
        manifest = {
            "format": RESULT_FORMAT,
            "stage": stage,
            "registration_id": registration["registration_id"],
            "matrices": spec.matrices,
            "branches_per_arm": spec.branches,
            "workers": workers,
            "generation_cpu_seconds": generation_cpu,
            "replay_cpu_seconds": replay_cpu,
            "wall_seconds": time.time() - started,
            "work_directory": str(work),
            "scientific_digest": _digest(
                [batch.scientific_digest for batch in generated]
            ),
            "exact_replay": True,
            "manual_stop_after_pilot": stage == "pilot",
            "confirmation_launched": False,
            "gates": metrics["gates"],
        }
        _atomic_json(staging / "manifest.json", manifest)
        write_checksums(staging)
        verify_checksums(staging)
        staging.rename(output)
        _append_ledger(
            f"<!-- phir-extension-px11-{stage}-result-{registration['registration_id']} -->",
            [
                f"## PX11 {stage} completed",
                "",
                f"- Result: `{output.relative_to(ROOT)}`.",
                f"- Fingerprint: `{metrics['gates']['fingerprint_classification']}`.",
                f"- Pilot advancement: `{metrics['pilot_advancement']}`.",
                f"- Exact replay passed for all {spec.matrices} matrices.",
                "- No confirmation was launched automatically.",
            ],
        )
        _status_write(
            work,
            {
                "state": "complete",
                "stage": "sealed",
                "completed_matrices": spec.matrices,
                "total_matrices": spec.matrices,
                "output": str(output),
                "gates": metrics["gates"],
            },
        )
        return manifest
    except Exception as error:
        _status_write(
            work,
            {
                "state": "failed",
                "stage": "error",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def verify_result(directory: Path = DEFAULT_PILOT) -> dict[str, Any]:
    verify_checksums(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    replay = json.loads((directory / "replay_audit.json").read_text())
    if manifest["format"] != RESULT_FORMAT or not replay["complete_exact_replay"]:
        raise ValueError("PX11 result verification failed")
    return manifest


def authorize_confirmation(
    *,
    acknowledged_pilot_review: bool,
    directory: Path = DEFAULT_CONFIRMATION_REGISTRATION,
) -> dict[str, Any]:
    """Create the required human-gated confirmation seal from pilot survivors."""

    if not acknowledged_pilot_review:
        raise PermissionError("explicit pilot-review acknowledgement is required")
    if directory.exists() or DEFAULT_CONFIRMATION.exists():
        raise FileExistsError("PX11 confirmation artifact already exists")
    registration = verify_registration()
    pilot_manifest = verify_result(DEFAULT_PILOT)
    metrics = json.loads((DEFAULT_PILOT / "primary_metrics.json").read_text())
    advancement = metrics["pilot_advancement"]
    families = tuple(str(value) for value in advancement["advancing_families"])
    dose_active = bool(advancement["dose_channel"])
    sensor_profile = advancement["sensor_profile"]
    sensor_active = sensor_profile is not None
    if not families and not dose_active and not sensor_active:
        raise ValueError("PX11 pilot has no preregistered survivor to confirm")
    dose_models = json.loads((DEFAULT_PILOT / "dose_model.json").read_text())
    if dose_active and set(dose_models) != {
        "c02_A",
        "c02_B",
        "c03_A",
        "c03_B",
    }:
        raise ValueError("PX11 pilot dose contract is incomplete")
    confirmation_contract = {
        "format": "codex-ch5-phir-px11-confirmation-contract-v1",
        "pilot_registration_id": registration["registration_id"],
        "pilot_result_sha256": sha256_file(DEFAULT_PILOT / "manifest.json"),
        "pilot_scientific_digest": pilot_manifest["scientific_digest"],
        "advancing_families": list(families),
        "dose_channel_active": dose_active,
        "sensor_active": sensor_active,
        "selected_sensor_profile": sensor_profile if sensor_active else "NONE",
        "dose_models": dose_models if dose_active else {},
        "confirmation_matrices": CONFIRMATION_MATRICES,
        "fresh_disjoint_seed_domain": True,
        "source_changes_after_pilot": False,
        "automatic_launch": False,
    }
    confirmation_id = _digest(
        {
            "base_registration": registration,
            "contract": confirmation_contract,
            "pilot_manifest": pilot_manifest,
        }
    )
    payload = {
        "format": CONFIRMATION_REGISTRATION_FORMAT,
        "confirmation_registration_id": confirmation_id,
        "base_registration_id": registration["registration_id"],
        "confirmation_contract": confirmation_contract,
        "authorized_at_unix": time.time(),
        "scientific_confirmation_matrices_generated": False,
    }
    directory.mkdir(parents=True)
    _atomic_json(directory / "confirmation_contract.json", confirmation_contract)
    _atomic_json(directory / "registration.json", payload)
    write_checksums(directory)
    _append_ledger(
        f"<!-- phir-extension-px11-confirmation-registration-{confirmation_id} -->",
        [
            "## PX11 confirmation separately authorized",
            "",
            f"- Confirmation registration: `{confirmation_id}`.",
            f"- Advancing families: `{list(families)}`.",
            f"- Dose channel: `{dose_active}`; sensor: `{sensor_profile}`.",
            "- The pilot was reviewed; no confirmation matrix existed at this seal.",
        ],
    )
    return payload


def verify_confirmation_registration(
    directory: Path = DEFAULT_CONFIRMATION_REGISTRATION,
) -> dict[str, Any]:
    verify_checksums(directory)
    value = json.loads((directory / "registration.json").read_text())
    if value["format"] != CONFIRMATION_REGISTRATION_FORMAT:
        raise ValueError("unexpected PX11 confirmation registration format")
    registration = verify_registration()
    if value["base_registration_id"] != registration["registration_id"]:
        raise ValueError("PX11 confirmation points to another base registration")
    contract = value["confirmation_contract"]
    if sha256_file(DEFAULT_PILOT / "manifest.json") != contract[
        "pilot_result_sha256"
    ]:
        raise ValueError("PX11 pilot result changed after confirmation seal")
    return value


def _cr6_state_case(
    arrays: Mapping[str, NDArray], index: int, beta: NDArray[np.float64]
) -> StateCase:
    length = int(arrays["history_lengths"][index])
    snapshot = Snapshot(
        composition=np.asarray(arrays["compositions"][index], dtype=np.int64),
        generation=int(arrays["generations"][index]),
        inheritance=tuple(
            bool(value) for value in arrays["inheritance"][index, :length]
        ),
        boundary_h=tuple(
            float(value) for value in arrays["boundary_h"][index, :length]
        ),
        previous_growth_steps=int(arrays["previous_growth_steps"][index]),
        cumulative_growth_steps=int(arrays["cumulative_growth_steps"][index]),
    )
    candidate = str(arrays["candidates"][index])
    return StateCase(
        state_id=str(arrays["state_ids"][index]),
        cohort="CR6_RETAINED_REPLAY",
        candidate=candidate,
        matrix_id=int(arrays["matrix_ids"][index]),
        landmark=int(arrays["landmarks"][index]),
        beta=beta,
        snapshot=snapshot,
    )


def _trace_cr6_branch(
    case: StateCase,
    regime: str,
    arm: str,
    edit: MolecularEdit | None,
    branch: int,
) -> tuple[px9.PairBlock, str]:
    spec = cr6.phase_spec(regime)
    config = cr6.regime_gard(regime)
    composition = (
        np.asarray(case.snapshot.composition, dtype=np.int64).copy()
        if edit is None
        else apply_molecular_edit(case.snapshot.composition, edit)
    )
    snapshot = Snapshot(
        composition,
        case.snapshot.generation,
        case.snapshot.inheritance,
        case.snapshot.boundary_h,
        case.snapshot.previous_growth_steps,
        case.snapshot.cumulative_growth_steps,
    )
    rng = np.random.default_rng(intervention_base._future_seed(spec, case, branch))
    molecular: list[NDArray[np.int64]] = [composition.copy()]
    generational_past: list[NDArray[np.int64]] = []
    generational_future: list[NDArray[np.int64]] = []
    records = []
    for _ in range(cr6.HORIZON):
        try:
            traced = advance_fission_traced(
                snapshot.composition,
                case.beta,
                config,
                CANDIDATES[case.candidate],
                rng,
            )
        except px9.SimulationError:
            break
        molecular.extend(
            np.asarray(value, dtype=np.int64).copy()
            for value in traced.growth_observations
        )
        molecular.append(np.asarray(traced.record.daughter, dtype=np.int64))
        generational_past.append(np.asarray(traced.record.parent, dtype=np.int64))
        generational_future.append(
            np.asarray(traced.record.daughter, dtype=np.int64)
        )
        records.append(traced.record)
        snapshot = px9._snapshot_after_record(snapshot, traced.record)
    n_types = GardConfig().n_types
    molecular_array = np.asarray(molecular, dtype=np.int16)
    past = (
        np.asarray(generational_past, dtype=np.int16)
        if generational_past
        else np.empty((0, n_types), dtype=np.int16)
    )
    future = (
        np.asarray(generational_future, dtype=np.int16)
        if generational_future
        else np.empty((0, n_types), dtype=np.int16)
    )
    return (
        px9.PairBlock(molecular_array[:-1], molecular_array[1:], past, future),
        px9._records_digest(records),
    )


def _transport_spec(regime: str, sensor_profile: str) -> PX11Spec:
    return PX11Spec(
        f"transport_{regime}",
        cr6.MATRICES,
        tuple(cr6.LANDMARKS),
        cr6.BRANCHES,
        cr6.HORIZON,
        0,
        BOOTSTRAP_DRAWS,
        RANDOMIZATION_DRAWS,
        sensor_profile,
        (),
        False,
        True,
    )


def _transport_worker(
    arguments: tuple[str, int, str]
) -> TransportBatch:
    regime, matrix_id, profile_name = arguments
    started = time.process_time()
    directory = CR6_ROOT / regime
    with threadpool_limits(limits=1), np.load(
        directory / "state_and_matrix_arrays.npz", allow_pickle=False
    ) as archive:
        arrays = {name: archive[name] for name in archive.files}
    beta_ids = np.asarray(arrays["beta_matrix_ids"], dtype=int)
    beta_index = int(np.flatnonzero(beta_ids == matrix_id)[0])
    beta = np.asarray(arrays["beta"][beta_index], dtype=np.float64)
    state_indices = np.flatnonzero(
        np.asarray(arrays["matrix_ids"], dtype=int) == matrix_id
    )
    selections = pd.read_csv(directory / "selected_interventions.csv")
    selections["candidate"] = selections["candidate"].map(
        lambda value: f"{int(value):02d}"
    )
    stored = pd.read_csv(
        directory / "branches.csv.gz",
        usecols=[
            "state_id",
            "candidate",
            "matrix_id",
            "arm",
            "branch",
            "branch_half",
            "joint_break_run3",
            "record_digest",
        ],
    )
    stored["candidate"] = stored["candidate"].map(
        lambda value: f"{int(value):02d}"
    )
    stored = stored[stored["matrix_id"] == matrix_id]
    profile = SENSOR_BY_NAME[profile_name]
    spec = _transport_spec(regime, profile_name)
    score_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    for index in state_indices:
        case = _cr6_state_case(arrays, int(index), beta)
        state_selection = selections[selections["state_id"] == case.state_id]
        blocks_by_arm: dict[str, tuple[px9.PairBlock, ...]] = {}
        for arm in ("MODEL_UP", "MODEL_DOWN", "RANDOM", "NOOP"):
            selected = state_selection[state_selection["arm"] == arm]
            if len(selected) != 1:
                raise ValueError(f"CR6 retained selection is incomplete: {case.state_id} {arm}")
            row = selected.iloc[0]
            edit = (
                None
                if bool(row["is_noop"])
                else MolecularEdit(int(row["remove_type"]), int(row["add_type"]))
            )
            blocks: list[px9.PairBlock] = []
            for branch in range(cr6.BRANCHES):
                block, digest = _trace_cr6_branch(case, regime, arm, edit, branch)
                expected = stored[
                    (stored["state_id"] == case.state_id)
                    & (stored["arm"] == arm)
                    & (stored["branch"] == branch)
                ]
                if len(expected) != 1:
                    raise ValueError("CR6 retained branch row is incomplete")
                expected_row = expected.iloc[0]
                exact = digest == str(expected_row["record_digest"])
                replay_rows.append(
                    {
                        "regime": regime,
                        "matrix_id": matrix_id,
                        "state_id": case.state_id,
                        "candidate": case.candidate,
                        "arm": arm,
                        "branch": branch,
                        "half": "A" if branch < cr6.BRANCHES // 2 else "B",
                        "joint_break_run3": int(expected_row["joint_break_run3"]),
                        "expected_digest": str(expected_row["record_digest"]),
                        "replay_digest": digest,
                        "exact": exact,
                    }
                )
                blocks.append(block)
            blocks_by_arm[arm] = tuple(blocks)
        for arm, blocks in blocks_by_arm.items():
            for half, indices in _halves(spec).items():
                score = _sensor_score(
                    px9.ResilienceCase(
                        case.state_id,
                        case.candidate,
                        case.matrix_id,
                        case.landmark,
                        case.beta,
                        case.snapshot,
                        np.empty((0, GardConfig().n_types), dtype=np.int16),
                    ),
                    blocks,
                    indices,
                    profile,
                    spec,
                )
                score_rows.append(
                    {
                        "regime": regime,
                        "matrix_id": matrix_id,
                        "state_id": case.state_id,
                        "candidate": case.candidate,
                        "landmark": case.landmark,
                        "arm": arm,
                        "source_half": half,
                        "profile": profile_name,
                        **score,
                    }
                )
    digest = _digest({"scores": score_rows, "replay": replay_rows})
    return TransportBatch(
        regime,
        matrix_id,
        tuple(score_rows),
        tuple(replay_rows),
        float(time.process_time() - started),
        digest,
    )


def _transport_analysis(
    batches: Sequence[TransportBatch], profile: str
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    scores = pd.DataFrame([row for batch in batches for row in batch.score_rows])
    replay = pd.DataFrame([row for batch in batches for row in batch.replay_rows])
    arrays: dict[str, NDArray] = {}
    regimes: dict[str, Any] = {}
    for regime in cr6.REGIMES:
        spec = _transport_spec(regime, profile)
        local_scores = scores[
            (scores["regime"] == regime) & scores["valid"].astype(bool)
        ]
        local_replay = replay[replay["regime"] == regime]
        response: list[dict[str, Any]] = []
        reliability: list[dict[str, Any]] = []
        forecast: list[dict[str, Any]] = []
        random_control: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            for half in _halves(spec):
                cell = local_scores[
                    (local_scores["candidate"] == candidate)
                    & (local_scores["source_half"] == half)
                ]
                means = (
                    cell.groupby(["matrix_id", "arm"])["temporal_value"]
                    .mean()
                    .unstack("arm")
                )
                item = _bootstrap_summary(
                    means["MODEL_UP"] - means["MODEL_DOWN"],
                    spec,
                    f"transport/{regime}/response/{candidate}/{half}",
                    arrays,
                )
                item.update(
                    {
                        "regime": regime,
                        "candidate": candidate,
                        "source_half": half,
                    }
                )
                response.append(item)
                control = _bootstrap_summary(
                    means["RANDOM"] - means["NOOP"],
                    spec,
                    f"transport/{regime}/random/{candidate}/{half}",
                    arrays,
                )
                control.update(
                    {
                        "regime": regime,
                        "candidate": candidate,
                        "source_half": half,
                    }
                )
                random_control.append(control)

            wide = local_scores[
                (local_scores["candidate"] == candidate)
                & local_scores["arm"].isin(("MODEL_UP", "MODEL_DOWN"))
            ].pivot_table(
                index=["matrix_id", "state_id", "arm"],
                columns="source_half",
                values="temporal_value",
                aggfunc="first",
            ).dropna(subset=["A", "B"])
            values = wide.groupby(level="matrix_id").apply(
                lambda frame: _safe_spearman(frame["A"], frame["B"]),
                include_groups=False,
            )
            item = _bootstrap_summary(
                values,
                spec,
                f"transport/{regime}/reliability/{candidate}",
                arrays,
            )
            item.update({"regime": regime, "candidate": candidate})
            reliability.append(item)

            for source_half in _halves(spec):
                target_half = "B" if source_half == "A" else "A"
                score = local_scores[
                    (local_scores["candidate"] == candidate)
                    & (local_scores["source_half"] == source_half)
                    & local_scores["arm"].isin(("MODEL_UP", "MODEL_DOWN"))
                ][["matrix_id", "state_id", "arm", "temporal_value"]]
                outcome = (
                    local_replay[
                        (local_replay["candidate"] == candidate)
                        & (local_replay["half"] == target_half)
                        & local_replay["arm"].isin(("MODEL_UP", "MODEL_DOWN"))
                    ]
                    .groupby(["matrix_id", "state_id", "arm"])[
                        "joint_break_run3"
                    ]
                    .mean()
                    .rename("outcome")
                    .reset_index()
                )
                joined = score.merge(
                    outcome, on=["matrix_id", "state_id", "arm"]
                )
                values = joined.groupby("matrix_id").apply(
                    lambda frame: _safe_spearman(
                        frame["temporal_value"], frame["outcome"]
                    ),
                    include_groups=False,
                )
                item = _bootstrap_summary(
                    values,
                    spec,
                    f"transport/{regime}/forecast/{candidate}/{source_half}",
                    arrays,
                )
                item.update(
                    {
                        "regime": regime,
                        "candidate": candidate,
                        "source_half": source_half,
                        "target_half": target_half,
                    }
                )
                forecast.append(item)
        _adjust_family(response)
        _adjust_family(reliability)
        _adjust_family(forecast)
        regimes[regime] = {
            "role": cr6.REGIMES[regime][2],
            "response": response,
            "reliability": reliability,
            "forecast": forecast,
            "random_control": random_control,
            "all_response_cells_positive": bool(
                len(response) == 4
                and all(float(item["effect"]) > 0.0 for item in response)
            ),
        }
    metrics = {
        "format": "codex-ch5-phir-px11-cr6-transport-metrics-v1",
        "profile": profile,
        "retrospective": True,
        "fresh_confirmation": False,
        "regimes": regimes,
        "complete_exact_replay": bool(
            len(replay) > 0 and replay["exact"].astype(bool).all()
        ),
    }
    return metrics, {"sensor_scores": scores, "replay_rows": replay}, arrays


def run_transport(
    *,
    output: Path = DEFAULT_TRANSPORT,
    work: Path = DEFAULT_TRANSPORT_WORK,
    workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    registration = verify_registration()
    pilot = verify_result(DEFAULT_PILOT)
    metrics_pilot = json.loads((DEFAULT_PILOT / "primary_metrics.json").read_text())
    profile = metrics_pilot["pilot_advancement"]["sensor_profile"]
    if profile is None:
        raise ValueError("PX11 pilot selected no sensor profile for transport")
    used_cpu = (
        float(pilot["generation_cpu_seconds"]) + float(pilot["replay_cpu_seconds"])
    ) / 3600.0
    if DEFAULT_CONFIRMATION.exists():
        confirmation = verify_result(DEFAULT_CONFIRMATION)
        used_cpu += (
            float(confirmation["generation_cpu_seconds"])
            + float(confirmation["replay_cpu_seconds"])
        ) / 3600.0
    if used_cpu >= MAX_TOTAL_CPU_HOURS:
        raise TimeoutError("PX11 has no remaining registered CPU budget")
    work.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "codex-ch5-phir-px11-transport-contract-v1",
        "registration_id": registration["registration_id"],
        "pilot_digest": pilot["scientific_digest"],
        "profile": profile,
        "regimes": list(cr6.REGIMES),
        "matrices_per_regime": cr6.MATRICES,
        "retrospective": True,
    }
    contract_path = work / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("PX11 transport checkpoint contract mismatch")
    else:
        _atomic_json(contract_path, contract)
    checkpoint = work / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    batches: dict[tuple[str, int], TransportBatch] = {}
    pending: list[tuple[str, int, str]] = []
    for regime in cr6.REGIMES:
        for matrix_id in range(cr6.MATRICES):
            path = checkpoint / f"{regime}__matrix_{matrix_id:03d}.pkl"
            if path.exists():
                with path.open("rb") as handle:
                    batch = pickle.load(handle)
                if not isinstance(batch, TransportBatch):
                    raise ValueError("invalid PX11 transport checkpoint")
                batches[(regime, matrix_id)] = batch
            else:
                pending.append((regime, matrix_id, profile))
    _status_write(
        work,
        {
            "state": "running",
            "stage": "cr6_transport",
            "completed_matrices": len(batches),
            "total_matrices": len(cr6.REGIMES) * cr6.MATRICES,
        },
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_transport_worker, argument): argument[:2]
            for argument in pending
        }
        for future in as_completed(futures):
            key = futures[future]
            batch = future.result()
            _atomic_pickle(
                checkpoint / f"{key[0]}__matrix_{key[1]:03d}.pkl", batch
            )
            batches[key] = batch
            _status_write(
                work,
                {
                    "state": "running",
                    "stage": "cr6_transport",
                    "completed_matrices": len(batches),
                    "total_matrices": len(cr6.REGIMES) * cr6.MATRICES,
                },
            )
    ordered = [
        batches[(regime, matrix_id)]
        for regime in cr6.REGIMES
        for matrix_id in range(cr6.MATRICES)
    ]
    metrics, tables, arrays = _transport_analysis(ordered, profile)
    if not metrics["complete_exact_replay"]:
        raise AssertionError("PX11 CR6 transport replay disagrees with retained data")
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _atomic_json(staging / "primary_metrics.json", metrics)
    for name, table in tables.items():
        table.to_csv(staging / f"{name}.csv.gz", index=False, compression="gzip")
    np.savez_compressed(staging / "inference_arrays.npz", **arrays)
    report = [
        "# PX11 retrospective CR6 sensor transport",
        "",
        f"Frozen sensor profile: `{profile}`.",
        "",
        "This is deterministic remeasurement of retained CR6 trajectories, not a fresh confirmation. Every replayed branch matched its retained digest.",
    ]
    for regime, value in metrics["regimes"].items():
        report.append(
            f"- {regime}: all targeted-response cells positive = **{value['all_response_cells_positive']}**."
        )
    (staging / "SCIENTIFIC_REPORT.md").write_text("\n".join(report) + "\n")
    (staging / "LAY_SUMMARY.md").write_text(
        "# Lay summary — PX11 regime transport\n\n"
        "We replayed the older parameter-regime experiments and asked whether the compact temporal gauge travelled with them. This reuses old outcomes, so it is a stress test rather than a new confirmation.\n"
    )
    manifest = {
        "format": "codex-ch5-phir-px11-transport-result-v1",
        "registration_id": registration["registration_id"],
        "profile": profile,
        "matrices": len(cr6.REGIMES) * cr6.MATRICES,
        "cpu_seconds": float(sum(item.cpu_seconds for item in ordered)),
        "exact_replay": True,
        "retrospective": True,
    }
    _atomic_json(staging / "manifest.json", manifest)
    write_checksums(staging)
    verify_checksums(staging)
    staging.rename(output)
    _status_write(
        work,
        {
            "state": "complete",
            "stage": "cr6_transport",
            "completed_matrices": len(ordered),
            "total_matrices": len(ordered),
            "output": str(output),
        },
    )
    return manifest


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def launch_detached(
    stage: str,
    *,
    workers: int = MAX_WORKERS,
    cpu_budget_hours: float | None = None,
    work: Path | None = None,
) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    spec, output, default_work, log, _contract = _stage_settings(stage)
    work = default_work if work is None else work
    if output.exists():
        raise FileExistsError(output)
    work.mkdir(parents=True, exist_ok=True)
    launch_path = work / "detached_launch.json"
    if launch_path.exists():
        old = json.loads(launch_path.read_text())
        if _pid_alive(int(old.get("pid", -1))):
            raise RuntimeError(f"PX11 {stage} already runs as PID {old['pid']}")
    if cpu_budget_hours is None:
        cpu_budget_hours = (
            DEFAULT_PILOT_CPU_HOURS
            if stage == "pilot"
            else DEFAULT_CONFIRMATION_CPU_HOURS
        )
    command = [
        sys.executable,
        "-m",
        "plastic_heredity.phir_extension_px11",
        "run",
        stage,
        "--workers",
        str(workers),
        "--cpu-budget-hours",
        str(cpu_budget_hours),
        "--work-dir",
        str(work),
    ]
    with log.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    payload = {
        "format": "codex-ch5-phir-px11-detached-launch-v1",
        "registration_id": registration["registration_id"],
        "stage": stage,
        "pid": process.pid,
        "workers": workers,
        "cpu_budget_hours": cpu_budget_hours,
        "matrices": spec.matrices,
        "work": str(work),
        "log": str(log),
        "command": command,
        "launched_at_unix": time.time(),
    }
    _atomic_json(launch_path, payload)
    _append_ledger(
        f"<!-- phir-extension-px11-{stage}-launch-{process.pid} -->",
        [
            f"## PX11 {stage} detached launch",
            "",
            f"- PID `{process.pid}`, eight-worker ceiling, work `{work}`.",
            f"- The sealed {spec.matrices}-matrix stage started with no automatic successor.",
        ],
    )
    return payload


def launch_transport_detached(
    *, workers: int = MAX_WORKERS, work: Path = DEFAULT_TRANSPORT_WORK
) -> dict[str, Any]:
    verify_registration()
    verify_result(DEFAULT_PILOT)
    if DEFAULT_TRANSPORT.exists():
        raise FileExistsError(DEFAULT_TRANSPORT)
    work.mkdir(parents=True, exist_ok=True)
    launch_path = work / "detached_launch.json"
    if launch_path.exists():
        old = json.loads(launch_path.read_text())
        if _pid_alive(int(old.get("pid", -1))):
            raise RuntimeError(f"PX11 transport already runs as PID {old['pid']}")
    command = [
        sys.executable,
        "-m",
        "plastic_heredity.phir_extension_px11",
        "transport",
        "--workers",
        str(workers),
        "--work-dir",
        str(work),
    ]
    with DEFAULT_TRANSPORT_LOG.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    payload = {
        "format": "codex-ch5-phir-px11-transport-launch-v1",
        "pid": process.pid,
        "workers": workers,
        "work": str(work),
        "log": str(DEFAULT_TRANSPORT_LOG),
        "command": command,
        "launched_at_unix": time.time(),
    }
    _atomic_json(launch_path, payload)
    return payload


def status(stage: str = "pilot", work: Path | None = None) -> dict[str, Any]:
    _spec, output, default_work, log, _contract = _stage_settings(stage)
    work = default_work if work is None else work
    payload: dict[str, Any] = {
        "format": "codex-ch5-phir-px11-status-report-v1",
        "stage": stage,
        "validation": DEFAULT_VALIDATION.exists(),
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "output": output.exists(),
        "confirmation_registration": DEFAULT_CONFIRMATION_REGISTRATION.exists(),
        "log": str(log),
    }
    launch = work / "detached_launch.json"
    if launch.exists():
        value = json.loads(launch.read_text())
        value["alive"] = _pid_alive(int(value.get("pid", -1)))
        payload["launch"] = value
    state = work / "status.json"
    if state.exists():
        payload["work_status"] = json.loads(state.read_text())
    if log.exists():
        payload["log_bytes"] = log.stat().st_size
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("register")
    sub.add_parser("smoke")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("stage", choices=("pilot", "confirmation"))
    run_parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    run_parser.add_argument("--cpu-budget-hours", type=float)
    run_parser.add_argument("--work-dir", type=Path)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("stage", choices=("pilot", "confirmation"))
    launch_parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    launch_parser.add_argument("--cpu-budget-hours", type=float)
    launch_parser.add_argument("--work-dir", type=Path)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("stage", choices=("pilot", "confirmation"), nargs="?", default="pilot")
    status_parser.add_argument("--work-dir", type=Path)
    authorize = sub.add_parser("authorize-confirmation")
    authorize.add_argument("--acknowledge-pilot-reviewed", action="store_true")
    transport = sub.add_parser("transport")
    transport.add_argument("--workers", type=int, default=MAX_WORKERS)
    transport.add_argument("--work-dir", type=Path, default=DEFAULT_TRANSPORT_WORK)
    launch_transport = sub.add_parser("launch-transport")
    launch_transport.add_argument("--workers", type=int, default=MAX_WORKERS)
    launch_transport.add_argument("--work-dir", type=Path, default=DEFAULT_TRANSPORT_WORK)
    verify = sub.add_parser("verify")
    verify.add_argument("stage", choices=("pilot", "confirmation"), nargs="?", default="pilot")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        value = validate()
    elif args.command == "register":
        value = register()
    elif args.command == "smoke":
        value = smoke()
    elif args.command == "run":
        value = run(
            args.stage,
            work=args.work_dir,
            workers=args.workers,
            cpu_budget_hours=args.cpu_budget_hours,
        )
    elif args.command == "launch":
        value = launch_detached(
            args.stage,
            work=args.work_dir,
            workers=args.workers,
            cpu_budget_hours=args.cpu_budget_hours,
        )
    elif args.command == "status":
        value = status(args.stage, args.work_dir)
    elif args.command == "authorize-confirmation":
        value = authorize_confirmation(
            acknowledged_pilot_review=args.acknowledge_pilot_reviewed
        )
    elif args.command == "transport":
        value = run_transport(work=args.work_dir, workers=args.workers)
    elif args.command == "launch-transport":
        value = launch_transport_detached(work=args.work_dir, workers=args.workers)
    elif args.command == "verify":
        directory = DEFAULT_PILOT if args.stage == "pilot" else DEFAULT_CONFIRMATION
        value = verify_result(directory)
    else:
        raise AssertionError(args.command)
    print(json.dumps(_json_ready(value), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
