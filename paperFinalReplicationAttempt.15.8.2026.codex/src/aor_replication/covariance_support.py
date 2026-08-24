"""Registered, label-blind covariance-support ladder for Phi instruments."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import CausalConfig, GardConfig
from .gard import RunTrace, simulate_gard
from .storage import load_trace, save_trace, write_json
from .support_information import (
    ALL_SUPPORT_INSTRUMENTS,
    SUPPORT_INSTRUMENTS,
    InstrumentReading,
    prepare_support_window,
    score_operational_window,
    score_prepared_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAT = "aor-phi-covariance-support-v1"
DEVELOPMENT_SEEDS = tuple(range(26_083_101, 26_083_107))
SUPPORT_LADDER = (64, 96, 128, 192, 256, 384, 512)
REPEATS = 12
POOL_PAIRS = 512
SUBSAMPLE_SEED = 26_083_900

HASHED_SOURCE_FILES = (
    "COVARIANCE_SUPPORT_AMENDMENT_001.md",
    "COVARIANCE_SUPPORT_PROTOCOL.md",
    "LABEL_CONTRACT_STATUS.md",
    "FORMULATION_BRIDGE_PROTOCOL.md",
    "FORMULATION_BRIDGE_PILOT_REPORT.md",
    "REPLICATION_REPORT.md",
    "pyproject.toml",
    "scripts/run-covariance-support-detached.sh",
    "scripts/status-covariance-support.sh",
    "src/aor_replication/bridge_information.py",
    "src/aor_replication/composition.py",
    "src/aor_replication/config.py",
    "src/aor_replication/covariance_support.py",
    "src/aor_replication/gard.py",
    "src/aor_replication/information.py",
    "src/aor_replication/storage.py",
    "src/aor_replication/support_information.py",
    "tests/test_covariance_support.py",
    "tests/test_formulation_bridge.py",
)

PRIOR_BRIDGE_REGISTRATION_ID = (
    "95f8359f17e5c14790dc4fe0cc6c4014e0b78deb54c76d328436dc96f385695c"
)


@dataclass(frozen=True)
class SupportAuditConfig:
    seeds: Tuple[int, ...] = DEVELOPMENT_SEEDS
    supports: Tuple[int, ...] = SUPPORT_LADDER
    repeats: int = REPEATS
    pool_pairs: int = POOL_PAIRS
    subsample_seed: int = SUBSAMPLE_SEED
    gard: GardConfig = field(
        default_factory=lambda: replace(GardConfig(), generations=160)
    )
    causal: CausalConfig = field(default_factory=CausalConfig)

    def validate(self, *, require_frozen: bool = False) -> None:
        if len(self.seeds) != 6 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("support audit requires six unique development seeds")
        if self.supports != SUPPORT_LADDER:
            raise ValueError("support ladder drifted")
        if self.supports[-1] != self.pool_pairs:
            raise ValueError("largest support must equal the fixed pair pool")
        if self.repeats != REPEATS or self.repeats < 2:
            raise ValueError("support repeat count drifted")
        if self.gard.generations != 160:
            raise ValueError("development trajectory length drifted")
        if self.causal.lag != 1:
            raise ValueError("support audit is frozen at lag one")
        self.gard.validate()
        self.causal.validate()
        if require_frozen and self != frozen_support_config():
            raise ValueError("scientific support execution requires frozen config")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def frozen_support_config() -> SupportAuditConfig:
    return SupportAuditConfig()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_hashes() -> Dict[str, str]:
    missing = [relative for relative in HASHED_SOURCE_FILES if not (PROJECT_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"support registration sources missing: {missing}")
    return {
        relative: _sha256_file(PROJECT_ROOT / relative)
        for relative in HASHED_SOURCE_FILES
    }


def _imported_modules(path: Path) -> Tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return tuple(modules)


def _assert_label_isolation() -> None:
    for relative in (
        "src/aor_replication/support_information.py",
        "src/aor_replication/covariance_support.py",
    ):
        imported = _imported_modules(PROJECT_ROOT / relative)
        if any("replicator" in module for module in imported):
            raise RuntimeError(f"label-blind module imports detector code: {relative}")


def _registration_core(config: SupportAuditConfig) -> Dict[str, Any]:
    _assert_label_isolation()
    return {
        "format": FORMAT,
        "config": config.to_dict(),
        "source_sha256": _source_hashes(),
        "instruments": list(ALL_SUPPORT_INSTRUMENTS),
        "selectable_instrument": "pca8_full_revised",
        "diagnostic_only_instrument": "raw100_full_revised",
        "prior_bridge_registration_id": PRIOR_BRIDGE_REGISTRATION_ID,
        "replicator_label_gate": "unresolved",
        "labels_computed_or_read": False,
        "outcome_pilot_authorized": False,
        "interventions_authorized": False,
    }


def register_support_audit(output: Path) -> Dict[str, Any]:
    config = frozen_support_config()
    config.validate(require_frozen=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_covariance_support.py",
    ]
    validation = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if validation.returncode != 0:
        detail = (validation.stdout + "\n" + validation.stderr)[-8000:]
        raise RuntimeError(f"support validation failed; registration refused:\n{detail}")
    core = _registration_core(config)
    registration_id = _digest_json(core)
    payload = {
        **core,
        "registration_id": registration_id,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "validation": {
            "command": command,
            "returncode": validation.returncode,
            "stdout": validation.stdout.strip(),
            "stderr": validation.stderr.strip(),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "registration.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing.get("registration_id") != registration_id:
            raise RuntimeError("support registration directory contains a different seal")
        return existing
    write_json(path, payload)
    return payload


def verify_support_registration(
    registration: Path, config: SupportAuditConfig
) -> Dict[str, Any]:
    path = registration / "registration.json" if registration.is_dir() else registration
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    expected = _registration_core(config)
    expected_id = _digest_json(expected)
    if payload.get("registration_id") != expected_id:
        raise RuntimeError("support registration/config/source hash mismatch")
    for key, value in expected.items():
        if payload.get(key) != _canonical(value):
            raise RuntimeError(f"support registration field drifted: {key}")
    if payload.get("labels_computed_or_read") is not False:
        raise RuntimeError("support registration is not label blind")
    return payload


def _trace_digest(trace: RunTrace) -> str:
    digest = hashlib.sha256()
    for array in (
        trace.counts,
        trace.generations,
        trace.phases,
        trace.joins,
        trace.leaves,
        trace.intervention_species,
        trace.intervention_delta,
        trace.intervention_score,
        trace.beta,
        np.asarray(trace.seed, dtype=np.int64),
    ):
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _pair_digest(indices: np.ndarray) -> str:
    value = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _reading_row(
    reading: InstrumentReading,
    *,
    mode: str,
    run_index: int,
    seed: int,
    repeat: int,
    support: int,
    trace_sha256: str,
    pair_sha256: str,
    transform_sha256: str,
    partition_sha256: str,
    pca_sha256: str,
) -> Dict[str, Any]:
    return {
        "mode": mode,
        "run_index": run_index,
        "seed": seed,
        "repeat": repeat,
        "support": support,
        "instrument": reading.name,
        "ordinary_score": reading.score,
        "active_dimensions": reading.active_dimensions,
        "state_dimensions": reading.state_dimensions,
        "part_a_dimensions": reading.part_a_dimensions,
        "part_b_dimensions": reading.part_b_dimensions,
        "joint_past_future_dimensions": reading.whole_joint_dimension,
        "effective_transition_pairs": reading.pairs,
        "samples_per_joint_dimension": reading.samples_per_joint_dimension,
        "whole_joint_covariance_rank": reading.whole_joint_rank,
        "whole_joint_rank_fraction": reading.whole_joint_rank_fraction,
        "whole_joint_ridge": reading.whole_joint_ridge,
        "covariance_rule": reading.covariance_rule,
        "redundancy_channel": reading.redundancy_channel,
        "trace_sha256": trace_sha256,
        "pair_sha256": pair_sha256,
        "transform_sha256": transform_sha256,
        "partition_sha256": partition_sha256,
        "pca_sha256": pca_sha256,
    }


def _diagnostic_rows(
    reading: InstrumentReading,
    *,
    mode: str,
    run_index: int,
    seed: int,
    repeat: int,
    support: int,
) -> list[Dict[str, Any]]:
    identity = {
        "mode": mode,
        "run_index": run_index,
        "seed": seed,
        "repeat": repeat,
        "support": support,
        "instrument": reading.name,
    }
    rows = [
        {**identity, "diagnostic_type": "component", "name": name, "value": value}
        for name, value in reading.components.items()
    ]
    rows.extend(
        {**identity, "diagnostic_type": "rank", "name": name, "value": value}
        for name, value in reading.ranks.items()
    )
    rows.extend(
        {**identity, "diagnostic_type": "ridge", "name": name, "value": value}
        for name, value in reading.ridges.items()
    )
    return rows


def _nested_indices(
    pool_pairs: int, support: int, permutation: np.ndarray
) -> np.ndarray:
    if support == pool_pairs:
        return np.arange(pool_pairs, dtype=np.int64)
    indices = np.concatenate(
        (
            np.asarray(permutation[: support - 1], dtype=np.int64),
            np.asarray([pool_pairs - 1], dtype=np.int64),
        )
    )
    return np.sort(indices).astype(np.int64)


def _pairwise_order_agreement(
    current: Mapping[int, float], reference: Mapping[int, float]
) -> Tuple[int, int]:
    agreements = 0
    comparisons = 0
    common = sorted(set(current) & set(reference))
    for first, second in itertools.combinations(common, 2):
        reference_difference = reference[first] - reference[second]
        current_difference = current[first] - current[second]
        if np.isclose(reference_difference, 0.0, atol=1e-12):
            continue
        comparisons += 1
        agreements += int(np.sign(reference_difference) == np.sign(current_difference))
    return agreements, comparisons


def _stability_summary(scores: pd.DataFrame) -> pd.DataFrame:
    primary = scores[scores["mode"] == "paired_subsample"].copy()
    operational = scores[scores["mode"] == "end_anchored"].copy()
    rows = []
    for instrument in ALL_SUPPORT_INSTRUMENTS:
        selected = primary[primary["instrument"] == instrument]
        full = selected[selected["support"] == POOL_PAIRS]
        references = full.groupby("run_index")["ordinary_score"].mean().to_dict()
        reference_scale = float(np.median(np.abs(list(references.values()))))
        operational_selected = operational[operational["instrument"] == instrument]
        operational_reference = (
            operational_selected[
                operational_selected["support"] == POOL_PAIRS
            ]
            .set_index("run_index")["ordinary_score"]
            .to_dict()
        )
        for support in SUPPORT_LADDER:
            support_rows = selected[selected["support"] == support]
            agreements = 0
            comparisons = 0
            for _, repeated in support_rows.groupby("repeat", sort=True):
                current = repeated.set_index("run_index")["ordinary_score"].to_dict()
                matched, total = _pairwise_order_agreement(current, references)
                agreements += matched
                comparisons += total
            medians = (
                support_rows.groupby("run_index")["ordinary_score"]
                .median()
                .to_dict()
            )
            common = sorted(set(medians) & set(references))
            if len(common) >= 3:
                rho = float(
                    stats.spearmanr(
                        [medians[index] for index in common],
                        [references[index] for index in common],
                    ).statistic
                )
            else:
                rho = float("nan")
            drifts = []
            for item in support_rows.itertuples(index=False):
                denominator = abs(references[item.run_index]) + reference_scale
                drifts.append(
                    abs(item.ordinary_score - references[item.run_index])
                    / max(denominator, np.finfo(float).eps)
                )
            current_operational = (
                operational_selected[operational_selected["support"] == support]
                .set_index("run_index")["ordinary_score"]
                .to_dict()
            )
            op_agreements, op_comparisons = _pairwise_order_agreement(
                current_operational, operational_reference
            )
            ordering = float(agreements / comparisons) if comparisons else float("nan")
            operational_ordering = (
                float(op_agreements / op_comparisons)
                if op_comparisons
                else float("nan")
            )
            rows.append(
                {
                    "instrument": instrument,
                    "support": support,
                    "eligible_trajectories": len(references),
                    "ordering_agreement": ordering,
                    "contrast_sign_flip_rate": 1.0 - ordering,
                    "median_score_spearman_vs_512": rho,
                    "median_normalized_drift": float(np.median(drifts)),
                    "operational_ordering_agreement": operational_ordering,
                    "operational_contrast_sign_flip_rate": 1.0
                    - operational_ordering,
                    "descriptively_unstable": bool(
                        np.isfinite(ordering) and (1.0 - ordering) > 0.20
                    ),
                }
            )
    return pd.DataFrame(rows)


def _score_level_summary(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(["mode", "instrument", "support"], sort=False)
        .agg(
            readings=("ordinary_score", "size"),
            score_mean=("ordinary_score", "mean"),
            score_median=("ordinary_score", "median"),
            score_std=("ordinary_score", "std"),
            score_min=("ordinary_score", "min"),
            score_max=("ordinary_score", "max"),
            joint_dimension=("joint_past_future_dimensions", "median"),
            covariance_rank_median=("whole_joint_covariance_rank", "median"),
            rank_fraction_median=("whole_joint_rank_fraction", "median"),
            ridge_median=("whole_joint_ridge", "median"),
            samples_per_joint_dimension=("samples_per_joint_dimension", "median"),
        )
        .reset_index()
    )


def _stability_gate(stability: pd.DataFrame, eligible: int) -> Dict[str, Any]:
    candidate = stability[
        stability["instrument"] == "pca8_full_revised"
    ].copy()
    evaluated = candidate[candidate["support"] < POOL_PAIRS]
    checks = []
    for row in evaluated.itertuples(index=False):
        checks.append(
            {
                "support": int(row.support),
                "ordering_pass": bool(row.ordering_agreement >= 0.80),
                "spearman_pass": bool(row.median_score_spearman_vs_512 >= 0.70),
                "drift_pass": bool(row.median_normalized_drift <= 0.25),
                "operational_ordering_pass": bool(
                    row.operational_ordering_agreement >= 0.80
                ),
            }
        )
    numerical_pass = bool(
        eligible >= 5
        and len(checks) == len(SUPPORT_LADDER) - 1
        and all(all(value for key, value in check.items() if key != "support") for check in checks)
    )
    return {
        "candidate": "pca8_full_revised",
        "eligible_trajectories": eligible,
        "synthetic_gates_passed_at_registration": True,
        "support_checks": checks,
        "numerical_stability_pass": numerical_pass,
        "outcome_pilot_authorized": False,
        "label_gate": "unresolved",
        "next_action": (
            "freeze_candidate_and_request_human_review_of_provisional_label_protocol"
            if numerical_pass
            else "stop_and_retain_numerical_instability_null"
        ),
    }


def _runtime_versions() -> Dict[str, str]:
    packages = ("numpy", "scipy", "pandas", "scikit-learn")
    result = {}
    for package in packages:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _write_summary(
    path: Path,
    registration_id: str,
    manifest: pd.DataFrame,
    levels: pd.DataFrame,
    stability: pd.DataFrame,
    gate: Mapping[str, Any],
    permutation: pd.DataFrame,
) -> None:
    candidate_levels = levels[
        (levels["mode"] == "paired_subsample")
        & (
            levels["instrument"].isin(
                ["raw100_full_revised", "pca8_full_revised"]
            )
        )
    ][
        [
            "instrument",
            "support",
            "score_median",
            "joint_dimension",
            "covariance_rank_median",
            "samples_per_joint_dimension",
        ]
    ]
    candidate_stability = stability[
        stability["instrument"] == "pca8_full_revised"
    ][
        [
            "support",
            "ordering_agreement",
            "median_score_spearman_vs_512",
            "median_normalized_drift",
            "operational_ordering_agreement",
        ]
    ]
    permutation_max = (
        permutation.groupby("instrument")["absolute_difference"].max().to_dict()
    )
    lines = [
        "# Phi-family covariance-support diagnostic",
        "",
        f"Registration: `{registration_id}`.",
        "",
        f"Eligible fixed trajectories: {int(manifest['eligible'].sum())}/{len(manifest)}. No replicator labels were imported, computed, or read.",
        "",
        "## Raw versus stabilized score levels",
        "",
        "```text",
        candidate_levels.to_string(index=False),
        "```",
        "",
        "## PCA8 frozen stability gate",
        "",
        "```text",
        candidate_stability.to_string(index=False),
        "```",
        "",
        f"Numerical stability pass: **{'yes' if gate['numerical_stability_pass'] else 'no'}**.",
        f"Registered next action: `{gate['next_action']}`.",
        "",
        "## Molecule-label permutation audit",
        "",
    ]
    for instrument, maximum in permutation_max.items():
        lines.append(f"- `{instrument}` maximum absolute score difference: {maximum:.6g}.")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The self-replicator label gate remains unresolved. This numerical audit authorizes neither an outcome pilot nor an intervention.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_support_audit(
    output: Path,
    registration: Path,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    config = frozen_support_config()
    config.validate(require_frozen=True)
    sealed = verify_support_registration(registration, config)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and not overwrite:
        with config_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != _canonical(config.to_dict()):
            raise RuntimeError("support output contains a different configuration")
    write_json(config_path, config.to_dict())
    write_json(output / "registration.json", sealed)
    write_json(
        output / "provenance.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "registration_id": sealed["registration_id"],
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _runtime_versions(),
            "source_sha256": sealed["source_sha256"],
            "labels_computed_or_read": False,
            "interventions_run": False,
            "outcome_pilot_run": False,
        },
    )

    score_rows: list[Dict[str, Any]] = []
    diagnostic_rows: list[Dict[str, Any]] = []
    manifest_rows: list[Dict[str, Any]] = []
    permutation_rows: list[Dict[str, Any]] = []

    for run_index, seed in enumerate(config.seeds):
        trace_path = output / "traces" / f"run-{run_index:03d}.npz"
        if trace_path.exists() and not overwrite:
            trace = load_trace(trace_path)
            trace.validate(config.gard)
        else:
            trace = simulate_gard(config.gard, seed)
            save_trace(trace_path, trace)
        trace_hash = _trace_digest(trace)
        replay = simulate_gard(config.gard, seed)
        replay_exact = _trace_digest(replay) == trace_hash
        transitions = trace.counts.shape[0] - 1
        eligible = transitions >= config.pool_pairs
        manifest_rows.append(
            {
                "run_index": run_index,
                "seed": seed,
                "trace_sha256": trace_hash,
                "molecular_observations": int(trace.counts.shape[0]),
                "transition_pairs": int(transitions),
                "eligible": bool(eligible),
                "replay_exact": bool(replay_exact),
                "interventions": int(np.count_nonzero(trace.intervention_delta)),
                "labels_computed_or_read": False,
            }
        )
        if not replay_exact:
            raise RuntimeError(f"development trace replay failed for seed {seed}")
        if not eligible:
            print(
                f"support trajectory {run_index + 1}/{len(config.seeds)} ineligible: "
                f"{transitions} < {config.pool_pairs} pairs",
                flush=True,
            )
            continue

        pool_counts = trace.counts[-(config.pool_pairs + 1) :]
        prepared = prepare_support_window(pool_counts, trace.beta, config.causal)
        full_cache: Optional[Dict[str, InstrumentReading]] = None
        for repeat_index in range(config.repeats):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [config.subsample_seed, seed, repeat_index]
                )
            )
            permutation = rng.permutation(config.pool_pairs - 1)
            for support in config.supports:
                indices = _nested_indices(config.pool_pairs, support, permutation)
                if support == config.pool_pairs and full_cache is not None:
                    readings = full_cache
                else:
                    readings = score_prepared_pairs(prepared, indices)
                    if support == config.pool_pairs:
                        full_cache = readings
                pair_hash = _pair_digest(indices)
                for reading in readings.values():
                    score_rows.append(
                        _reading_row(
                            reading,
                            mode="paired_subsample",
                            run_index=run_index,
                            seed=seed,
                            repeat=repeat_index,
                            support=support,
                            trace_sha256=trace_hash,
                            pair_sha256=pair_hash,
                            transform_sha256=prepared.transform_digest,
                            partition_sha256=prepared.partition_digest,
                            pca_sha256=prepared.pca_digest,
                        )
                    )
                    diagnostic_rows.extend(
                        _diagnostic_rows(
                            reading,
                            mode="paired_subsample",
                            run_index=run_index,
                            seed=seed,
                            repeat=repeat_index,
                            support=support,
                        )
                    )

        for support in config.supports:
            window = pool_counts[-(support + 1) :]
            operational_prepared, readings = score_operational_window(
                window, trace.beta, config.causal
            )
            indices = np.arange(support, dtype=np.int64)
            for reading in readings.values():
                score_rows.append(
                    _reading_row(
                        reading,
                        mode="end_anchored",
                        run_index=run_index,
                        seed=seed,
                        repeat=-1,
                        support=support,
                        trace_sha256=trace_hash,
                        pair_sha256=_pair_digest(indices),
                        transform_sha256=operational_prepared.transform_digest,
                        partition_sha256=operational_prepared.partition_digest,
                        pca_sha256=operational_prepared.pca_digest,
                    )
                )
                diagnostic_rows.extend(
                    _diagnostic_rows(
                        reading,
                        mode="end_anchored",
                        run_index=run_index,
                        seed=seed,
                        repeat=-1,
                        support=support,
                    )
                )

        molecule_permutation = np.random.default_rng(seed + 700_000).permutation(
            trace.counts.shape[1]
        )
        permuted_prepared = prepare_support_window(
            pool_counts[:, molecule_permutation],
            trace.beta[np.ix_(molecule_permutation, molecule_permutation)],
            config.causal,
        )
        original_full = full_cache
        if original_full is None:
            original_full = score_prepared_pairs(
                prepared, np.arange(config.pool_pairs, dtype=np.int64)
            )
        permuted_full = score_prepared_pairs(
            permuted_prepared, np.arange(config.pool_pairs, dtype=np.int64)
        )
        for instrument in ALL_SUPPORT_INSTRUMENTS:
            difference = permuted_full[instrument].score - original_full[instrument].score
            permutation_rows.append(
                {
                    "run_index": run_index,
                    "seed": seed,
                    "instrument": instrument,
                    "original_score": original_full[instrument].score,
                    "permuted_score": permuted_full[instrument].score,
                    "difference": float(difference),
                    "absolute_difference": float(abs(difference)),
                    "all_coordinate_invariance_required": instrument
                    in {
                        "public_nine_atom",
                        "pca8_full_revised",
                        "raw100_full_revised",
                    },
                }
            )
        print(
            f"completed label-blind support trajectory {run_index + 1}/{len(config.seeds)}",
            flush=True,
        )

    manifest = pd.DataFrame(manifest_rows)
    scores = pd.DataFrame(score_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    permutation_frame = pd.DataFrame(permutation_rows)
    eligible_count = int(manifest["eligible"].sum())
    if scores.empty:
        raise RuntimeError("no eligible support trajectories were scored")
    stability = _stability_summary(scores)
    levels = _score_level_summary(scores)
    gate = _stability_gate(stability, eligible_count)
    required_permutation = permutation_frame[
        permutation_frame["all_coordinate_invariance_required"]
    ]
    permutation_pass = bool(
        np.all(required_permutation["absolute_difference"].to_numpy() <= 2e-7)
    )
    gate["molecule_label_permutation_pass"] = permutation_pass
    gate["numerical_stability_pass"] = bool(
        gate["numerical_stability_pass"] and permutation_pass
    )
    if not gate["numerical_stability_pass"]:
        gate["next_action"] = "stop_and_retain_numerical_instability_null"

    manifest.to_csv(output / "trace_manifest.csv", index=False)
    scores.to_csv(output / "support_scores.csv", index=False)
    diagnostics.to_csv(output / "covariance_diagnostics.csv", index=False)
    levels.to_csv(output / "score_level_summary.csv", index=False)
    stability.to_csv(output / "stability_summary.csv", index=False)
    permutation_frame.to_csv(output / "molecule_permutation_audit.csv", index=False)
    write_json(
        output / "stability_gate.json",
        {
            "format": FORMAT,
            "registration_id": sealed["registration_id"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "labels_computed_or_read": False,
            "interventions_run": False,
            "outcome_pilot_run": False,
            **gate,
        },
    )
    _write_summary(
        output / "SUMMARY.md",
        sealed["registration_id"],
        manifest,
        levels,
        stability,
        gate,
        permutation_frame,
    )
    return gate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aor_replication.covariance_support",
        description="Label-blind Phi covariance-support diagnostic",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "covariance-support-registration",
    )
    run = subparsers.add_parser("run")
    run.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "covariance-support-audit",
    )
    run.add_argument(
        "--registration",
        type=Path,
        default=Path("results") / "covariance-support-registration",
    )
    run.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "register":
        registration = register_support_audit(args.output)
        print(
            f"covariance-support audit registered as {registration['registration_id']} "
            f"in {args.output.resolve()}"
        )
        return 0
    result = run_support_audit(
        args.output, args.registration, overwrite=args.overwrite
    )
    print(
        f"covariance-support audit written to {args.output.resolve()}; "
        f"next action: {result['next_action']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEVELOPMENT_SEEDS",
    "POOL_PAIRS",
    "REPEATS",
    "SUBSAMPLE_SEED",
    "SUPPORT_LADDER",
    "SupportAuditConfig",
    "frozen_support_config",
    "register_support_audit",
    "run_support_audit",
    "verify_support_registration",
]
