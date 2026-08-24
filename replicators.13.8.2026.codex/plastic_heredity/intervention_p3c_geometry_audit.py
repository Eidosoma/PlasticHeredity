"""Checksum-sealed post-hoc geometry audit of the sealed P3b campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .experiment import _json_ready
from .intervention_metrics import _interval
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .seeds import derive_seed


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULT = (
    REPOSITORY_ROOT
    / "results_intervention_replication/p3b_beta_surgery_dose_bridge"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "results_intervention_replication/p3c_geometry_audit"
)
DOCUMENT = "CODEX_INTERVENTION_P3C_GEOMETRY_AUDIT.md"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_p3c_geometry_audit.py",
    "tests/test_intervention_p3c_geometry_audit.py",
)
FORMAT = "codex-intervention-p3c-posthoc-geometry-audit-v1"
BOOTSTRAP_REPETITIONS = 4_096
STANDARD_LANDMARKS = (20, 35, 50, 65, 80)
COMPATIBILITY_LANDMARKS = (60,)
ARMS = (
    "SMALL_LOOSEN",
    "SMALL_TIGHTEN",
    "SMALL_RANDOM_PP",
    "FABLE_LOOSEN",
    "FABLE_TIGHTEN",
    "FABLE_RANDOM_PP",
    "NOOP",
)
GEOMETRY_SHIFTS = (
    "log_throughput_ratio",
    "relative_block_sum",
    "log_spectral_radius_ratio",
    "relative_block_frobenius",
    "relative_row_strength_dispersion",
    "relative_column_strength_dispersion",
)
SEED = hashlib.sha256(
    b"codex-clean-room-p3c-posthoc-geometry-audit-v1::matrix-bootstrap"
).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def catalytic_throughput(composition: NDArray, beta: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    matrix = np.asarray(beta, dtype=np.float64)
    if values.ndim != 1 or matrix.shape != (values.size, values.size):
        raise ValueError("composition and beta dimensions differ")
    if values.sum() <= 0 or not np.isfinite(matrix).all():
        raise ValueError("invalid state geometry")
    x = values / values.sum()
    return float(x @ matrix @ x)


def block_geometry(composition: NDArray, beta: NDArray) -> dict[str, float | int]:
    values = np.asarray(composition, dtype=np.float64)
    matrix = np.asarray(beta, dtype=np.float64)
    present = np.flatnonzero(values > 0)
    if present.size == 0 or matrix.shape != (values.size, values.size):
        raise ValueError("invalid occupied block")
    block = matrix[np.ix_(present, present)]
    if np.any(block <= 0.0) or not np.isfinite(block).all():
        raise ValueError("occupied beta block must be finite and positive")
    eigenvalues = np.linalg.eigvals(block)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    row_strength = block.sum(axis=1)
    column_strength = block.sum(axis=0)
    return {
        "present_types": int(present.size),
        "present_present_edges": int(block.size),
        "throughput": catalytic_throughput(values, matrix),
        "block_sum": float(block.sum()),
        "block_arithmetic_mean": float(block.mean()),
        "block_geometric_mean": float(np.exp(np.log(block).mean())),
        "block_frobenius": float(np.linalg.norm(block)),
        "spectral_radius": spectral_radius,
        "row_strength_dispersion": float(np.std(row_strength) / np.mean(row_strength)),
        "column_strength_dispersion": float(
            np.std(column_strength) / np.mean(column_strength)
        ),
    }


def _safe_log_ratio(after: float, before: float) -> float:
    if after <= 0.0 or before <= 0.0:
        raise ValueError("log-ratio geometry must be positive")
    return float(np.log(after / before))


def geometry_shift(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    def relative(name: str) -> float:
        denominator = float(before[name])
        if denominator == 0.0:
            return 0.0 if float(after[name]) == 0.0 else float("nan")
        return float((float(after[name]) - denominator) / denominator)

    return {
        "log_throughput_ratio": _safe_log_ratio(
            float(after["throughput"]), float(before["throughput"])
        ),
        "relative_block_sum": relative("block_sum"),
        "log_spectral_radius_ratio": _safe_log_ratio(
            float(after["spectral_radius"]), float(before["spectral_radius"])
        ),
        "relative_block_frobenius": relative("block_frobenius"),
        "relative_row_strength_dispersion": relative("row_strength_dispersion"),
        "relative_column_strength_dispersion": relative(
            "column_strength_dispersion"
        ),
    }


def radial_projection(before: NDArray, after: NDArray) -> float:
    original = np.asarray(before, dtype=np.float64)
    changed = np.asarray(after, dtype=np.float64)
    denominator = float(original @ original)
    if denominator <= 0.0:
        raise ValueError("radial projection requires a nonzero original block")
    return float((changed - original) @ original / denominator)


def _load_source() -> tuple[dict[str, NDArray], NDArray, NDArray, pd.DataFrame]:
    verify_checksums(SOURCE_RESULT)
    with np.load(
        SOURCE_RESULT / "state_and_matrix_arrays.npz", allow_pickle=False
    ) as archive:
        state = {name: archive[name].copy() for name in archive.files}
    with np.load(SOURCE_RESULT / "branch_arrays.npz", allow_pickle=False) as archive:
        targets = archive["targets"].copy()
    edges = pd.read_csv(SOURCE_RESULT / "beta_surgery_edges.csv.gz")
    probabilities = pd.read_csv(SOURCE_RESULT / "state_probabilities.csv")
    return state, targets, edges, probabilities


def compute_geometry_table(
    state: dict[str, NDArray], edges: pd.DataFrame
) -> pd.DataFrame:
    state_ids = [str(value) for value in state["state_ids"]]
    beta_by_matrix = {
        int(matrix_id): matrix
        for matrix_id, matrix in zip(
            state["beta_matrix_ids"], state["beta"], strict=True
        )
    }
    edge_groups = {
        (str(state_id), str(arm)): frame
        for (state_id, arm), frame in edges.groupby(["state_id", "arm"], sort=False)
    }
    rows: list[dict[str, Any]] = []
    for index, state_id in enumerate(state_ids):
        composition = np.asarray(state["compositions"][index], dtype=np.int64)
        matrix_id = int(state["matrix_ids"][index])
        beta = np.asarray(beta_by_matrix[matrix_id], dtype=np.float64)
        noop = block_geometry(composition, beta)
        present = np.flatnonzero(composition > 0)
        before_block = beta[np.ix_(present, present)].ravel()
        for arm in ARMS:
            altered = beta
            group = edge_groups.get((state_id, arm))
            if group is not None:
                altered = beta.copy()
                flat = group["flat_beta_index"].to_numpy(dtype=np.int64)
                altered.ravel()[flat] = group["after"].to_numpy(dtype=np.float64)
                if not np.allclose(
                    beta.ravel()[flat],
                    group["before"].to_numpy(dtype=np.float64),
                    atol=1e-15,
                    rtol=1e-12,
                ):
                    raise ValueError(f"edge audit before-values changed for {state_id} {arm}")
            observed = block_geometry(composition, altered)
            shift = geometry_shift(observed, noop)
            after_block = altered[np.ix_(present, present)].ravel()
            row: dict[str, Any] = {
                "state_id": state_id,
                "candidate": str(state["candidates"][index]),
                "matrix_id": matrix_id,
                "landmark": int(state["landmarks"][index]),
                "arm": arm,
                "structural_no_action": group is None and arm != "NOOP",
                "radial_projection": radial_projection(before_block, after_block),
            }
            row.update({f"noop_{name}": value for name, value in noop.items()})
            row.update({f"arm_{name}": value for name, value in observed.items()})
            row.update(shift)
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != len(state_ids) * len(ARMS):
        raise AssertionError("geometry table is incomplete")
    return result


def _rank(values: NDArray) -> NDArray[np.float64]:
    return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy()


def _spearman(x: NDArray, y: NDArray) -> float:
    x_rank = _rank(x)
    y_rank = _rank(y)
    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _centred_slope(x: NDArray, y: NDArray, groups: NDArray) -> float:
    xv = np.asarray(x, dtype=np.float64).copy()
    yv = np.asarray(y, dtype=np.float64).copy()
    identifiers = np.asarray(groups)
    for identifier in np.unique(identifiers):
        mask = identifiers == identifier
        xv[mask] -= xv[mask].mean()
        yv[mask] -= yv[mask].mean()
    denominator = float(xv @ xv)
    return float(xv @ yv / denominator) if denominator > 0.0 else float("nan")


def _bootstrap_statistics(
    frame: pd.DataFrame,
    geometry_name: str,
    bootstrap_indices: NDArray[np.int64],
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    matrices = np.sort(frame["matrix_id"].unique())
    work = frame[["matrix_id", "state_id", geometry_name, "realized_shift"]].copy()
    work["x_centered"] = work[geometry_name] - work.groupby("state_id")[
        geometry_name
    ].transform("mean")
    work["y_centered"] = work["realized_shift"] - work.groupby("state_id")[
        "realized_shift"
    ].transform("mean")
    work["numerator"] = work["x_centered"] * work["y_centered"]
    work["denominator"] = work["x_centered"] ** 2
    contributions = work.groupby("matrix_id", sort=True)[
        ["numerator", "denominator"]
    ].sum()
    if not np.array_equal(contributions.index.to_numpy(), matrices):
        raise AssertionError("matrix ordering changed in geometry bootstrap")
    numerator = contributions["numerator"].to_numpy(dtype=np.float64)
    denominator = contributions["denominator"].to_numpy(dtype=np.float64)
    estimate_slope = float(numerator.sum() / denominator.sum())
    slopes = numerator[bootstrap_indices].sum(axis=1) / denominator[
        bootstrap_indices
    ].sum(axis=1)

    state_correlations: list[dict[str, Any]] = []
    for state_id, state_frame in work.groupby("state_id", sort=False):
        value = _spearman(
            state_frame[geometry_name].to_numpy(),
            state_frame["realized_shift"].to_numpy(),
        )
        if np.isfinite(value):
            state_correlations.append(
                {
                    "state_id": state_id,
                    "matrix_id": int(state_frame["matrix_id"].iloc[0]),
                    "rho": value,
                }
            )
    correlations = pd.DataFrame(state_correlations)
    matrix_rho = correlations.groupby("matrix_id", sort=True)["rho"].mean().reindex(
        matrices
    )
    rho_values = matrix_rho.to_numpy(dtype=np.float64)
    finite = np.isfinite(rho_values)
    if not finite.any():
        raise ValueError("no finite within-state correlations in this cell")
    # A branch half can be all tied for every state in a particular matrix.
    # Such a matrix has no defined rank correlation; it is retained in every
    # other analysis and is explicitly omitted only from this descriptive rank
    # estimand.
    estimate_rho = float(np.nanmean(rho_values))
    with np.errstate(invalid="ignore"):
        rhos = np.nanmean(rho_values[bootstrap_indices], axis=1)
    if not np.isfinite(rhos).all():  # pragma: no cover - fantastically unlikely
        raise ValueError("a bootstrap draw contained no informative rank block")
    return estimate_slope, estimate_rho, slopes, rhos


def analyze_geometry(
    geometry: pd.DataFrame, probabilities: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, NDArray]]:
    outcomes: list[dict[str, Any]] = []
    candidate_values = probabilities["candidate"].astype(int).map(lambda v: f"{v:02d}")
    probability_lookup = probabilities.assign(candidate_text=candidate_values).set_index(
        "state_id"
    )
    for row in geometry.itertuples(index=False):
        source = probability_lookup.loc[row.state_id]
        for half in ("A", "B"):
            q = float(source[f"q_half_{half}_{row.arm}"])
            q_noop = float(source[f"q_half_{half}_NOOP"])
            item = row._asdict()
            item["branch_half"] = half
            item["realized_probability"] = q
            item["realized_shift"] = q - q_noop
            outcomes.append(item)
    frame = pd.DataFrame(outcomes)
    matrix_count = int(frame["matrix_id"].nunique())
    rng = np.random.default_rng(derive_seed(SEED, "P3C.geometry.bootstrap"))
    bootstrap_indices = rng.integers(
        0, matrix_count, size=(BOOTSTRAP_REPETITIONS, matrix_count), dtype=np.int64
    )
    scopes = {
        "five_standard_landmarks": STANDARD_LANDMARKS,
        "landmark60_compatibility": COMPATIBILITY_LANDMARKS,
    }
    cells: list[dict[str, Any]] = []
    arrays: dict[str, NDArray] = {"bootstrap_indices": bootstrap_indices}
    for scope, landmarks in scopes.items():
        for candidate in ("02", "03"):
            for half in ("A", "B"):
                selected = frame[
                    (frame["candidate"] == candidate)
                    & (frame["branch_half"] == half)
                    & (frame["landmark"].isin(landmarks))
                ].copy()
                # NOOP fixes the state origin and is retained in the across-arm
                # state-centred geometry/outcome association.
                cell: dict[str, Any] = {
                    "scope": scope,
                    "candidate": candidate,
                    "branch_half": half,
                    "matrices": int(selected["matrix_id"].nunique()),
                    "states": int(selected["state_id"].nunique()),
                    "geometry_associations": {},
                    "arm_summaries": {},
                }
                key = f"{scope}__c{candidate}_{half}"
                for name in GEOMETRY_SHIFTS:
                    slope, rho, slope_draws, rho_draws = _bootstrap_statistics(
                        selected, name, bootstrap_indices
                    )
                    cell["geometry_associations"][name] = {
                        "state_centered_slope": slope,
                        "slope_bootstrap_ci95": _interval(
                            slope_draws[np.isfinite(slope_draws)]
                        ),
                        "mean_within_state_spearman": rho,
                        "spearman_bootstrap_ci95": _interval(
                            rho_draws[np.isfinite(rho_draws)]
                        ),
                    }
                    arrays[f"{key}__{name}__slope"] = slope_draws
                    arrays[f"{key}__{name}__spearman"] = rho_draws
                for arm in ARMS:
                    arm_frame = selected[selected["arm"] == arm]
                    by_matrix = arm_frame.groupby("matrix_id", sort=True)[
                        ["log_throughput_ratio", "realized_shift"]
                    ].mean()
                    q_boot = by_matrix["realized_shift"].to_numpy()[
                        bootstrap_indices
                    ].mean(axis=1)
                    throughput_boot = by_matrix["log_throughput_ratio"].to_numpy()[
                        bootstrap_indices
                    ].mean(axis=1)
                    cell["arm_summaries"][arm] = {
                        "mean_log_throughput_ratio": float(
                            by_matrix["log_throughput_ratio"].mean()
                        ),
                        "log_throughput_bootstrap_ci95": _interval(throughput_boot),
                        "mean_realized_shift": float(by_matrix["realized_shift"].mean()),
                        "realized_shift_bootstrap_ci95": _interval(q_boot),
                        "matrix_spearman_geometry_outcome": _spearman(
                            by_matrix["log_throughput_ratio"].to_numpy(),
                            by_matrix["realized_shift"].to_numpy(),
                        ),
                    }
                cells.append(cell)
    summary: dict[str, Any] = {
        "format": FORMAT,
        "classification": "posthoc_exploratory_existing_data_audit",
        "source_result": str(SOURCE_RESULT.relative_to(REPOSITORY_ROOT)),
        "source_checksum_manifest_sha256": sha256_file(SOURCE_RESULT / "SHA256SUMS"),
        "states": int(geometry["state_id"].nunique()),
        "matrices": matrix_count,
        "geometry_rows": len(geometry),
        "outcome_geometry_rows": len(frame),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "inference_unit": "whole catalytic matrix",
        "cells": cells,
        "confirmatory": False,
        "can_rescue_p3b_gate": False,
    }
    summary["audit_id"] = _canonical_digest(summary)
    return summary, frame, arrays


def _technical_report(summary: dict[str, Any]) -> str:
    lines = [
        "# P3c post-hoc geometry audit",
        "",
        "This is an exploratory analysis of the sealed P3b data, not a new simulation and not a repair of P3b's failed specificity gate.",
        "",
        "## Main finding",
        "",
    ]
    for cell in summary["cells"]:
        if cell["scope"] != "five_standard_landmarks":
            continue
        association = cell["geometry_associations"]["log_throughput_ratio"]
        random_arm = cell["arm_summaries"]["FABLE_RANDOM_PP"]
        lines.append(
            f"- c{cell['candidate']} half {cell['branch_half']}: the high-dose balanced-log random arm changed mean log throughput by "
            f"{random_arm['mean_log_throughput_ratio']:+.5f} and JOINT_BREAK_RUN3 by "
            f"{random_arm['mean_realized_shift']:+.5f}; the state-centred throughput slope was "
            f"{association['state_centered_slope']:+.5f} "
            f"(95% matrix bootstrap {association['slope_bootstrap_ci95']})."
        )
    lines.extend(
        [
            "",
            "A log-balanced perturbation is not automatically neutral in ordinary catalytic throughput: exponentiating positive and negative log changes can raise weighted arithmetic support even when their unweighted log sum is zero. This is the prospective rationale for P3c's throughput-neutral control.",
            "",
            "## Claim boundary",
            "",
            "These associations are post-hoc. They can explain why the old random arm moved the outcome, but cannot prove mediation or turn P3b into a formal pass. That requires P3c's fresh pilot and untouched confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def _lay_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Lay summary",
            "",
            "P3b's supposedly random comparison was random in direction, but it was not neutral in what mattered physically. Because catalytic strengths are multiplied, balancing the plus and minus changes on a logarithmic scale can still increase the assembly's overall catalytic support. The random arm therefore behaved partly like a mild strengthening treatment.",
            "",
            "P3c will solve that by using a random change constrained to leave the assembly's starting catalytic throughput exactly unchanged. The old P3b result stays exactly as reported: strengthening versus weakening worked, while its formal random-control gate failed.",
            "",
            "This audit only diagnoses the old data. Fresh prospectively registered simulations are still required before calling throughput a causal control axis.",
            "",
        ]
    )


def run(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    state, _targets, edges, probabilities = _load_source()
    geometry = compute_geometry_table(state, edges)
    summary, outcome_geometry, arrays = analyze_geometry(geometry, probabilities)
    with _atomic_destination(output) as destination:
        geometry.to_csv(destination / "state_arm_geometry.csv.gz", index=False)
        outcome_geometry.to_csv(
            destination / "state_arm_half_geometry_outcomes.csv.gz", index=False
        )
        np.savez_compressed(destination / "bootstrap_arrays.npz", **arrays)
        (destination / "audit_summary.json").write_text(
            json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "SCIENTIFIC_REPORT.md").write_text(
            _technical_report(summary), encoding="utf-8"
        )
        (destination / "LAY_SUMMARY.md").write_text(
            _lay_report(summary), encoding="utf-8"
        )
        manifest = {
            "format": FORMAT,
            "audit_id": summary["audit_id"],
            "classification": summary["classification"],
            "source_hashes": source_hashes(),
            "source_result_checksum_manifest_sha256": summary[
                "source_checksum_manifest_sha256"
            ],
            "new_scientific_matrices": 0,
            "new_simulated_futures": 0,
            "p3b_result_modified": False,
            "confirmatory": False,
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"P3c geometry audit sealed: {output}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    run(arguments.output)


if __name__ == "__main__":
    main()
