"""Registered 12-seed formulation bridge on identical untreated trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
from typing import Any, Dict, Mapping, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .bridge_information import ESTIMATOR_ORDER, fit_bridge_estimators
from .config import CausalConfig, GardConfig, ReplicatorConfig
from .gard import RunTrace, simulate_gard
from .replicators import detect_replicators
from .storage import load_trace, save_trace, write_json


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "aor-formulation-bridge-v1"
PROTOCOL_PATH = PROJECT_ROOT / "FORMULATION_BRIDGE_PROTOCOL.md"
PILOT_SEEDS = tuple(range(26_082_101, 26_082_113))
FORECAST_SEED = 26_082_999

HASHED_SOURCE_FILES = (
    "FORMULATION_BRIDGE_PROTOCOL.md",
    "pyproject.toml",
    "REPLICATION_REPORT.md",
    "scripts/run-formulation-bridge-detached.sh",
    "scripts/status-formulation-bridge.sh",
    "src/aor_replication/analysis.py",
    "src/aor_replication/bridge_information.py",
    "src/aor_replication/cli.py",
    "src/aor_replication/composition.py",
    "src/aor_replication/config.py",
    "src/aor_replication/formulation_bridge.py",
    "src/aor_replication/gard.py",
    "src/aor_replication/information.py",
    "src/aor_replication/replicators.py",
    "src/aor_replication/storage.py",
    "tests/test_formulation_bridge.py",
)

REFERENCE_SOURCE_HASHES = {
    "plastic_heredity/phir_extension_common.py": (
        "550f871092f2b05079293db3e75c5a1337f1a097665fbf845acf7f1073572c7a"
    ),
    "plastic_heredity/phir_rescue_instruments.py": (
        "55b1b8cb328a25ca497330ef8770f758c2807364cf22408200859663e252c62c"
    ),
    "plastic_heredity/phir_instruments.py": (
        "69132410f668a2d1c4767a75bf9f4e9c25a9182d12be887c15e75bd6e4f29205"
    ),
}


@dataclass(frozen=True)
class BridgePilotConfig:
    """Every result-determining choice in the bounded scientific pilot."""

    seeds: Tuple[int, ...] = PILOT_SEEDS
    early_fraction: float = 0.25
    grid_points: int = 128
    forecast_seed: int = FORECAST_SEED
    gard: GardConfig = field(default_factory=GardConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    replicator: ReplicatorConfig = field(default_factory=ReplicatorConfig)

    def validate(self, *, require_frozen_pilot: bool = False) -> None:
        if len(self.seeds) < 4 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("bridge seeds must contain at least four unique values")
        if not 0.0 < self.early_fraction < 0.5:
            raise ValueError("early_fraction must be between zero and one half")
        if self.grid_points < 16:
            raise ValueError("grid_points must be at least 16")
        if int(round(self.grid_points * self.early_fraction)) < 2:
            raise ValueError("early feature grid must contain at least two points")
        self.gard.validate()
        self.causal.validate()
        self.replicator.validate()
        if self.causal.lag != 1:
            raise ValueError("the frozen bridge uses lag one")
        if require_frozen_pilot and self != frozen_pilot_config():
            raise ValueError("scientific execution requires the exact frozen pilot config")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def frozen_pilot_config() -> BridgePilotConfig:
    return BridgePilotConfig()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
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


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        _canonical_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_hashes() -> Dict[str, str]:
    missing = [relative for relative in HASHED_SOURCE_FILES if not (PROJECT_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"registration sources are missing: {missing}")
    return {
        relative: _sha256_file(PROJECT_ROOT / relative)
        for relative in HASHED_SOURCE_FILES
    }


def _registration_core(config: BridgePilotConfig) -> Dict[str, Any]:
    return {
        "format": PROTOCOL_VERSION,
        "config": config.to_dict(),
        "estimator_order": list(ESTIMATOR_ORDER),
        "source_sha256": _source_hashes(),
        "reference_source_sha256": REFERENCE_SOURCE_HASHES,
        "historical_result_policy": "preserve_original_negative_replication",
        "replicator_label_policy": "existing_reconstruction_labels_retained",
        "interventions_authorized": False,
    }


def register_formulation_bridge(output: Path) -> Dict[str, Any]:
    """Run synthetic gates and seal the exact scientific pilot sources."""

    config = frozen_pilot_config()
    config.validate(require_frozen_pilot=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_formulation_bridge.py",
    ]
    validation = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if validation.returncode != 0:
        detail = (validation.stdout + "\n" + validation.stderr)[-6000:]
        raise RuntimeError(f"bridge validation failed; registration refused:\n{detail}")
    core = _registration_core(config)
    registration_id = _canonical_digest(core)
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
            raise RuntimeError("registration directory contains a different source/config seal")
        return existing
    write_json(path, payload)
    return payload


def verify_registration(path: Path, config: BridgePilotConfig) -> Dict[str, Any]:
    """Verify both registration integrity and current source hashes."""

    registration_path = path / "registration.json" if path.is_dir() else path
    if not registration_path.is_file():
        raise FileNotFoundError(f"bridge registration not found: {registration_path}")
    with registration_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    expected = _registration_core(config)
    expected_id = _canonical_digest(expected)
    if payload.get("registration_id") != expected_id:
        raise RuntimeError("bridge registration/config/source hash mismatch")
    for key, value in expected.items():
        if payload.get(key) != _canonical_json(value):
            raise RuntimeError(f"bridge registration field drifted: {key}")
    if payload.get("interventions_authorized") is not False:
        raise RuntimeError("bridge registration must not authorize interventions")
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


def _safe_spearman(first: FloatArray, second: FloatArray) -> Tuple[float, float]:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _resample_continuous(values: FloatArray, points: int) -> FloatArray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not data.size:
        raise ValueError("continuous bridge feature must be a nonempty vector")
    if data.size == 1:
        return np.repeat(data, points)
    return np.interp(
        np.linspace(0.0, 1.0, points),
        np.linspace(0.0, 1.0, data.size),
        data,
    )


def _resample_binary(values: NDArray[np.bool_], points: int) -> IntArray:
    data = np.asarray(values, dtype=np.int64)
    if data.ndim != 1 or not data.size:
        raise ValueError("binary bridge target must be a nonempty vector")
    positions = np.rint(np.linspace(0, data.size - 1, points)).astype(np.int64)
    return np.asarray(data[positions], dtype=np.int64)


def _association_row(
    run_index: int,
    seed: int,
    estimator: str,
    values: FloatArray,
    labels: NDArray[np.bool_],
) -> Dict[str, Any]:
    rho, rho_p = _safe_spearman(values, labels.astype(np.float64))
    replicating = values[labels]
    drift = values[~labels]
    if replicating.size and drift.size:
        test = stats.mannwhitneyu(
            replicating, drift, alternative="greater", method="auto"
        )
        mann_p = float(test.pvalue)
    else:
        mann_p = float("nan")
    return {
        "run_index": run_index,
        "seed": seed,
        "estimator": estimator,
        "transitions": int(values.size),
        "label_probability": float(labels.mean()),
        "score_mean": float(values.mean()),
        "score_std": float(values.std(ddof=1)),
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "mann_whitney_greater_p": mann_p,
        "mean_score_replicating": (
            float(replicating.mean()) if replicating.size else float("nan")
        ),
        "mean_score_drift": float(drift.mean()) if drift.size else float("nan"),
        "replicating_transitions": int(replicating.size),
        "drift_transitions": int(drift.size),
    }


def _summarize_association(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for estimator in ESTIMATOR_ORDER:
        selected = frame[frame.estimator == estimator]
        finite = selected[np.isfinite(selected.spearman_rho)]
        evaluable = int(len(finite))
        positive = int((finite.spearman_rho > 0.0).sum())
        target_positive = int(math.ceil(0.73 * evaluable))
        median_rho = (
            float(finite.spearman_rho.median()) if evaluable else float("nan")
        )
        rows.append(
            {
                "estimator": estimator,
                "runs": int(len(selected)),
                "evaluable_runs": evaluable,
                "positive_runs": positive,
                "positive_target": target_positive,
                "mean_rho": (
                    float(finite.spearman_rho.mean()) if evaluable else float("nan")
                ),
                "median_rho": median_rho,
                "positive_significant_runs": int(
                    (
                        (finite.spearman_rho > 0.0)
                        & (finite.spearman_p < 0.05)
                    ).sum()
                ),
                "replicating_score_higher_runs": int(
                    (
                        selected.mean_score_replicating
                        > selected.mean_score_drift
                    ).sum()
                ),
                "association_screen_pass": bool(
                    evaluable >= 10
                    and positive >= target_positive
                    and np.isfinite(median_rho)
                    and median_rho > 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _forecast_leave_one_out(
    features: Mapping[str, FloatArray],
    targets: IntArray,
    config: BridgePilotConfig,
) -> pd.DataFrame:
    n_runs = targets.shape[0]
    rows = []
    for held_out in range(n_runs):
        train = np.asarray([index for index in range(n_runs) if index != held_out])
        y_train = targets[train]
        y_test = targets[held_out : held_out + 1]
        random_state = config.forecast_seed + held_out
        for estimator in ESTIMATOR_ORDER:
            values = np.asarray(features[estimator], dtype=np.float64)
            model = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    batch_size="auto",
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=min(0.2, max(0.1, 2 / len(train))),
                    n_iter_no_change=30,
                    random_state=random_state,
                ),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(values[train], y_train)
            prediction = np.asarray(model.predict(values[held_out : held_out + 1]))
            prediction = prediction.reshape(y_test.shape)
            rows.append(
                {
                    "held_out_run": held_out,
                    "held_out_seed": config.seeds[held_out],
                    "model": estimator,
                    "accuracy": float(np.mean(prediction == y_test)),
                    "target_positive_fraction": float(y_test.mean()),
                    "predicted_positive_fraction": float(prediction.mean()),
                    "convergence_warning": bool(
                        any(issubclass(item.category, ConvergenceWarning) for item in caught)
                    ),
                    "random_state": random_state,
                }
            )
        majority = int(np.mean(y_train) >= 0.5)
        dummy = np.full_like(y_test, majority)
        rows.append(
            {
                "held_out_run": held_out,
                "held_out_seed": config.seeds[held_out],
                "model": "majority_dummy",
                "accuracy": float(np.mean(dummy == y_test)),
                "target_positive_fraction": float(y_test.mean()),
                "predicted_positive_fraction": float(dummy.mean()),
                "convergence_warning": False,
                "random_state": random_state,
            }
        )
    return pd.DataFrame(rows)


def _summarize_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    dummy = (
        frame[frame.model == "majority_dummy"]
        .set_index("held_out_run")
        .sort_index()
    )
    rows = []
    for model in (*ESTIMATOR_ORDER, "majority_dummy"):
        selected = frame[frame.model == model].set_index("held_out_run").sort_index()
        if model == "majority_dummy":
            differences = np.zeros(len(selected), dtype=np.float64)
        else:
            differences = selected.accuracy.to_numpy() - dummy.accuracy.to_numpy()
        wins = int(np.sum(differences > 0.0))
        rows.append(
            {
                "model": model,
                "held_out_runs": int(len(selected)),
                "mean_accuracy": float(selected.accuracy.mean()),
                "std_accuracy": float(selected.accuracy.std(ddof=1)),
                "majority_mean_accuracy": float(dummy.accuracy.mean()),
                "mean_accuracy_difference": float(differences.mean()),
                "wins_vs_majority": wins,
                "ties_vs_majority": int(np.sum(np.isclose(differences, 0.0))),
                "prediction_screen_pass": bool(
                    model != "majority_dummy"
                    and float(differences.mean()) > 0.0
                    and wins >= 8
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired_contrasts(
    association: pd.DataFrame, forecast: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    association_wide = association.pivot(
        index="run_index", columns="estimator", values="spearman_rho"
    )
    forecast_wide = forecast[forecast.model != "majority_dummy"].pivot(
        index="held_out_run", columns="model", values="accuracy"
    )
    for first, second in itertools.combinations(ESTIMATOR_ORDER, 2):
        for metric, wide in (
            ("spearman_rho", association_wide),
            ("forecast_accuracy", forecast_wide),
        ):
            differences = (wide[second] - wide[first]).dropna().to_numpy()
            rows.append(
                {
                    "metric": metric,
                    "contrast": f"{second}_minus_{first}",
                    "pairs": int(differences.size),
                    "mean_difference": (
                        float(differences.mean()) if differences.size else float("nan")
                    ),
                    "median_difference": (
                        float(np.median(differences))
                        if differences.size
                        else float("nan")
                    ),
                    "positive_pairs": int(np.sum(differences > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _runtime_versions() -> Dict[str, str]:
    packages = (
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "matplotlib",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _instrument_contract_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "estimator": "macro_wms",
                "transform": "original_clr_drop_last",
                "partition": "lagged_mi_fiedler",
                "score_dimensions": 2,
                "subtraction": "each_part_to_whole_future",
                "redundancy": "none",
            },
            {
                "estimator": "macro_mmi",
                "transform": "original_clr_drop_last",
                "partition": "lagged_mi_fiedler",
                "score_dimensions": 2,
                "subtraction": "each_part_to_whole_future",
                "redundancy": "minimum_mean_part_to_whole_future",
            },
            {
                "estimator": "public_nine_atom",
                "transform": "all_clr_rank_gaussian",
                "partition": "beta_physical_fiedler",
                "score_dimensions": 2,
                "subtraction": "public_phi_id_lattice",
                "redundancy": "public_nine_atom_sum",
            },
            {
                "estimator": "full_revised",
                "transform": "all_clr_rank_gaussian",
                "partition": "beta_physical_fiedler",
                "score_dimensions": "all_active_molecular_dimensions",
                "subtraction": "a_to_a_future_and_b_to_b_future",
                "redundancy": "minimum_of_aa_ab_ba_bb_mean_channels",
            },
        ]
    )


def _write_summary(
    path: Path,
    registration_id: str,
    trace_manifest: pd.DataFrame,
    association_summary: pd.DataFrame,
    forecast_summary: pd.DataFrame,
    screens: Mapping[str, Any],
) -> None:
    label_probability = float(trace_manifest.label_probability.mean())
    lines = [
        "# Arrivals formulation bridge: 12-seed pilot",
        "",
        f"Registration: `{registration_id}`.",
        "",
        "This is a new observational formulation study. It does not alter the original negative replication and it does not resolve the paper-versus-reconstruction replicator-label mismatch.",
        "",
        "## Label audit",
        "",
        f"The existing 0.95 cosine detector labeled a mean {label_probability:.1%} of molecular observations as replicating across these 12 trajectories; the paper reported 88% for its control definition. These outcomes therefore remain reconstruction-label results.",
        "",
        "## Retrospective association",
        "",
        "```text",
        association_summary.to_string(index=False),
        "```",
        "",
        "## Leakage-free early prediction",
        "",
        "```text",
        forecast_summary.to_string(index=False),
        "```",
        "",
        "## Frozen pilot screen",
        "",
    ]
    for estimator in ESTIMATOR_ORDER:
        status = screens[estimator]
        lines.append(
            f"- `{estimator}`: association={'pass' if status['association_screen_pass'] else 'fail'}; prediction={'pass' if status['prediction_screen_pass'] else 'fail'}; pilot viable={'yes' if status['pilot_viable'] else 'no'}."
        )
    lines.extend(
        [
            "",
            "## Stop rule",
            "",
            "No Phi-guided intervention was run or authorized. A pilot-viable instrument would still require a frozen larger observational validation and human review. If no instrument is viable, this branch stops pending author code or a separately justified estimator.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_formulation_bridge_pilot(
    output: Path,
    registration: Path,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Run the frozen pilot without interventions or estimator tuning."""

    config = frozen_pilot_config()
    config.validate(require_frozen_pilot=True)
    sealed = verify_registration(registration, config)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and not overwrite:
        with config_path.open("r", encoding="utf-8") as stream:
            existing_config = json.load(stream)
        if existing_config != _canonical_json(config.to_dict()):
            raise RuntimeError("pilot output contains a different configuration")
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
            "reference_source_sha256": sealed["reference_source_sha256"],
            "declared_dependency_note": (
                "Runtime versions are recorded exactly; the local Python 3.13 "
                "validation environment uses compatible newer wheels because "
                "the declared NumPy-1.x range has no Python-3.13 wheel."
            ),
            "interventions_run": False,
        },
    )

    association_rows = []
    component_rows = []
    trace_rows = []
    early_features: Dict[str, list[FloatArray]] = {
        estimator: [] for estimator in ESTIMATOR_ORDER
    }
    early_targets = []
    input_points = int(round(config.grid_points * config.early_fraction))
    output_points = config.grid_points - input_points

    for run_index, seed in enumerate(config.seeds):
        trace_path = output / "traces" / f"run-{run_index:03d}.npz"
        if trace_path.exists() and not overwrite:
            trace = load_trace(trace_path)
            trace.validate(config.gard)
            if trace.seed != seed:
                raise RuntimeError(f"trace seed mismatch at run {run_index}")
        else:
            trace = simulate_gard(config.gard, seed)
            save_trace(trace_path, trace)
        labels_result = detect_replicators(trace, config.replicator)
        full = fit_bridge_estimators(trace.counts, trace.beta, config.causal)
        early_boundary = max(
            6,
            min(
                trace.counts.shape[0] - 2,
                int(math.floor(config.early_fraction * trace.counts.shape[0])),
            ),
        )
        early = fit_bridge_estimators(
            trace.counts[:early_boundary], trace.beta, config.causal
        )
        target = _resample_binary(
            labels_result.labels[early_boundary:], output_points
        )
        early_targets.append(target)
        for estimator in ESTIMATOR_ORDER:
            result = full[estimator]
            aligned_labels = labels_result.labels[result.time_indices]
            association_rows.append(
                _association_row(
                    run_index,
                    seed,
                    estimator,
                    result.values,
                    aligned_labels,
                )
            )
            for component, value in result.components.items():
                component_rows.append(
                    {
                        "run_index": run_index,
                        "seed": seed,
                        "estimator": estimator,
                        "component": component,
                        "value": float(value),
                        "redundancy_channel": result.redundancy_channel,
                        "active_dimensions": result.active_dimensions,
                        "part_a_dimensions": int(result.partition_a.size),
                        "part_b_dimensions": int(result.partition_b.size),
                    }
                )
            early_features[estimator].append(
                _resample_continuous(early[estimator].values, input_points)
            )
        score_path = output / "scores" / f"run-{run_index:03d}.npz"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            score_path,
            labels=labels_result.labels,
            early_boundary=np.asarray(early_boundary, dtype=np.int64),
            early_target=target,
            **{f"full_{name}": full[name].values for name in ESTIMATOR_ORDER},
            **{f"early_{name}": early[name].values for name in ESTIMATOR_ORDER},
        )
        trace_rows.append(
            {
                "run_index": run_index,
                "seed": seed,
                "trace_sha256": _trace_digest(trace),
                "molecular_observations": int(trace.counts.shape[0]),
                "early_boundary": early_boundary,
                "replicator_support": labels_result.support,
                "label_probability": float(labels_result.labels.mean()),
                "constant_labels": bool(np.unique(labels_result.labels).size < 2),
                "interventions": int(np.count_nonzero(trace.intervention_delta)),
            }
        )
        print(
            f"completed formulation bridge trajectory {run_index + 1}/{len(config.seeds)}",
            flush=True,
        )

    trace_manifest = pd.DataFrame(trace_rows)
    association = pd.DataFrame(association_rows)
    components = pd.DataFrame(component_rows)
    feature_arrays = {
        estimator: np.vstack(rows) for estimator, rows in early_features.items()
    }
    target_array = np.vstack(early_targets).astype(np.int64)
    forecast = _forecast_leave_one_out(feature_arrays, target_array, config)
    association_summary = _summarize_association(association)
    forecast_summary = _summarize_forecast(forecast)
    contrasts = _paired_contrasts(association, forecast)

    screens = {}
    for estimator in ESTIMATOR_ORDER:
        association_pass = bool(
            association_summary.loc[
                association_summary.estimator == estimator,
                "association_screen_pass",
            ].iloc[0]
        )
        prediction_pass = bool(
            forecast_summary.loc[
                forecast_summary.model == estimator,
                "prediction_screen_pass",
            ].iloc[0]
        )
        screens[estimator] = {
            "association_screen_pass": association_pass,
            "prediction_screen_pass": prediction_pass,
            "pilot_viable": bool(association_pass and prediction_pass),
        }
    result = {
        "format": PROTOCOL_VERSION,
        "registration_id": sealed["registration_id"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runs": len(config.seeds),
        "mean_reconstruction_label_probability": float(
            trace_manifest.label_probability.mean()
        ),
        "paper_reported_control_probability": 0.88,
        "replicator_definition_discrepancy_resolved": False,
        "screens": screens,
        "any_pilot_viable": bool(
            any(value["pilot_viable"] for value in screens.values())
        ),
        "interventions_run": False,
        "next_action": (
            "design_larger_untouched_observational_validation_after_human_review"
            if any(value["pilot_viable"] for value in screens.values())
            else "stop_pending_author_code_or_new_instrument"
        ),
    }
    trace_manifest.to_csv(output / "trace_manifest.csv", index=False)
    _instrument_contract_frame().to_csv(
        output / "instrument_contract.csv", index=False
    )
    association.to_csv(output / "association_runs.csv", index=False)
    association_summary.to_csv(output / "association_summary.csv", index=False)
    components.to_csv(output / "estimator_components.csv", index=False)
    forecast.to_csv(output / "early_prediction_runs.csv", index=False)
    forecast_summary.to_csv(output / "early_prediction_summary.csv", index=False)
    contrasts.to_csv(output / "paired_estimator_contrasts.csv", index=False)
    np.savez_compressed(
        output / "early_prediction_dataset.npz",
        targets=target_array,
        **feature_arrays,
    )
    write_json(output / "pilot_screen.json", result)
    _write_summary(
        output / "SUMMARY.md",
        sealed["registration_id"],
        trace_manifest,
        association_summary,
        forecast_summary,
        screens,
    )
    return result


__all__ = [
    "BridgePilotConfig",
    "FORECAST_SEED",
    "PILOT_SEEDS",
    "frozen_pilot_config",
    "register_formulation_bridge",
    "run_formulation_bridge_pilot",
    "verify_registration",
]
