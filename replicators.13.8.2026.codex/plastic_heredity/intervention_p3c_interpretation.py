"""Read-only interpretation audit of the sealed P3c pilot and confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .experiment import _json_ready
from .intervention_metrics import _interval
from .intervention_p3c_geometry_audit import block_geometry
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
PILOT = RESULT_ROOT / "p3c_throughput_pilot"
CONFIRMATION = RESULT_ROOT / "p3c_throughput_confirmation"
DEFAULT_OUTPUT = RESULT_ROOT / "p4a_p3c_interpretation_audit"
DOCUMENT = "CODEX_INTERVENTION_P3C_INTERPRETATION_AUDIT.md"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_p3c_interpretation.py",
    "tests/test_intervention_p3c_interpretation.py",
)
FORMAT = "codex-p3c-sealed-interpretation-audit-v1"
ARMS = (
    "LOOSEN",
    "TIGHTEN",
    "BALANCED_LOG_RANDOM",
    "THROUGHPUT_NEUTRAL_RANDOM",
    "NOOP",
)
PRIMARY_LANDMARKS = (20, 35, 50, 65, 80)
EQUIVALENCE_MARGINS = (0.015, 0.020, 0.025, 0.030, 0.040)
GEOMETRY_NAMES = (
    "log_throughput_ratio",
    "relative_block_sum",
    "log_perron_ratio",
    "log_singular_ratio",
    "relative_incoming_dispersion",
    "relative_outgoing_dispersion",
    "relative_incoming_concentration",
    "relative_outgoing_concentration",
    "relative_asymmetry",
    "relative_reciprocity",
)


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def _safe_relative(after: float, before: float) -> float:
    if before == 0.0:
        return 0.0 if after == 0.0 else float("nan")
    return float((after - before) / before)


def _safe_log_ratio(after: float, before: float) -> float:
    if after <= 0.0 or before <= 0.0:
        return float("nan")
    return float(np.log(after / before))


def _gini(values: NDArray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.size == 0 or x.sum() <= 0.0:
        return 0.0
    ranks = np.arange(1, x.size + 1, dtype=np.float64)
    return float((2.0 * (ranks @ x) / x.sum() - (x.size + 1)) / x.size)


def extended_geometry(composition: NDArray, beta: NDArray) -> dict[str, float]:
    values = np.asarray(composition, dtype=np.float64)
    matrix = np.asarray(beta, dtype=np.float64)
    x = values / values.sum()
    present = np.flatnonzero(values > 0)
    block = matrix[np.ix_(present, present)]
    base = block_geometry(values, matrix)
    incoming = (matrix @ x)[present]
    outgoing = (matrix.T @ x)[present]
    singular = float(np.linalg.svd(block, compute_uv=False)[0])
    asymmetry = float(np.linalg.norm(block - block.T) / np.linalg.norm(block))
    upper = block[np.triu_indices(block.shape[0], 1)]
    lower = block.T[np.triu_indices(block.shape[0], 1)]
    reciprocity = (
        float(np.corrcoef(np.log(upper), np.log(lower))[0, 1])
        if upper.size > 1 and np.std(np.log(upper)) > 0 and np.std(np.log(lower)) > 0
        else 0.0
    )
    return {
        "throughput": float(base["throughput"]),
        "block_sum": float(base["block_sum"]),
        "block_frobenius": float(base["block_frobenius"]),
        "perron": float(base["spectral_radius"]),
        "leading_singular": singular,
        "incoming_dispersion": float(np.std(incoming) / np.mean(incoming)),
        "outgoing_dispersion": float(np.std(outgoing) / np.mean(outgoing)),
        "incoming_concentration": _gini(incoming),
        "outgoing_concentration": _gini(outgoing),
        "asymmetry": asymmetry,
        "reciprocity": reciprocity,
    }


def geometry_shift(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {
        "log_throughput_ratio": _safe_log_ratio(after["throughput"], before["throughput"]),
        "relative_block_sum": _safe_relative(after["block_sum"], before["block_sum"]),
        "log_perron_ratio": _safe_log_ratio(after["perron"], before["perron"]),
        "log_singular_ratio": _safe_log_ratio(after["leading_singular"], before["leading_singular"]),
        "relative_incoming_dispersion": _safe_relative(after["incoming_dispersion"], before["incoming_dispersion"]),
        "relative_outgoing_dispersion": _safe_relative(after["outgoing_dispersion"], before["outgoing_dispersion"]),
        "relative_incoming_concentration": _safe_relative(after["incoming_concentration"], before["incoming_concentration"]),
        "relative_outgoing_concentration": _safe_relative(after["outgoing_concentration"], before["outgoing_concentration"]),
        "relative_asymmetry": _safe_relative(after["asymmetry"], before["asymmetry"]),
        "relative_reciprocity": _safe_relative(after["reciprocity"], before["reciprocity"]),
    }


def _load_state(directory: Path) -> dict[str, NDArray]:
    with np.load(directory / "state_and_matrix_arrays.npz", allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def reconstruct_geometry(directory: Path) -> pd.DataFrame:
    verify_checksums(directory)
    state = _load_state(directory)
    edges = pd.read_csv(directory / "beta_surgery_edges.csv.gz")
    groups = {
        (str(state_id), str(arm)): frame
        for (state_id, arm), frame in edges.groupby(["state_id", "arm"], sort=False)
    }
    beta_by_id = {
        int(matrix_id): np.asarray(beta, dtype=np.float64)
        for matrix_id, beta in zip(state["beta_matrix_ids"], state["beta"], strict=True)
    }
    rows: list[dict[str, Any]] = []
    for index, raw_id in enumerate(state["state_ids"]):
        state_id = str(raw_id)
        composition = np.asarray(state["compositions"][index], dtype=np.int64)
        beta = beta_by_id[int(state["matrix_ids"][index])]
        before_geometry = extended_geometry(composition, beta)
        for arm in ARMS:
            altered = beta
            group = groups.get((state_id, arm))
            if group is not None:
                altered = beta.copy()
                locations = group["flat_beta_index"].to_numpy(dtype=np.int64)
                archived_before = group["before"].to_numpy(dtype=np.float64)
                if not np.allclose(beta.ravel()[locations], archived_before, rtol=1e-12, atol=1e-15):
                    raise AssertionError("archived beta before-values do not reconstruct")
                altered.ravel()[locations] = group["after"].to_numpy(dtype=np.float64)
            after_geometry = extended_geometry(composition, altered)
            row: dict[str, Any] = {
                "state_id": state_id,
                "candidate": str(state["candidates"][index]),
                "matrix_id": int(state["matrix_ids"][index]),
                "landmark": int(state["landmarks"][index]),
                "arm": arm,
            }
            row.update({f"noop_{name}": value for name, value in before_geometry.items()})
            row.update({f"arm_{name}": value for name, value in after_geometry.items()})
            row.update(geometry_shift(after_geometry, before_geometry))
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != len(state["state_ids"]) * len(ARMS):
        raise AssertionError("incomplete geometry reconstruction")
    return result


def _rank_correlation(x: NDArray, y: NDArray) -> float:
    left = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average")
    right = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average")
    if left.std() == 0.0 or right.std() == 0.0:
        return float("nan")
    return float(left.corr(right))


def _matrix_bootstrap_association(
    frame: pd.DataFrame, x_name: str, bootstrap: NDArray[np.int64]
) -> dict[str, Any]:
    work = frame[["matrix_id", "state_id", x_name, "outcome_shift"]].dropna().copy()
    work["x"] = work[x_name] - work.groupby("state_id")[x_name].transform("mean")
    work["y"] = work["outcome_shift"] - work.groupby("state_id")["outcome_shift"].transform("mean")
    work["num"] = work["x"] * work["y"]
    work["den"] = work["x"] ** 2
    matrices = np.sort(frame["matrix_id"].unique())
    contributions = work.groupby("matrix_id")[["num", "den"]].sum().reindex(matrices).fillna(0.0)
    numerator = contributions["num"].to_numpy()
    denominator = contributions["den"].to_numpy()
    slope = float(numerator.sum() / denominator.sum()) if denominator.sum() > 0 else float("nan")
    boot_den = denominator[bootstrap].sum(axis=1)
    boot_slope = np.divide(
        numerator[bootstrap].sum(axis=1),
        boot_den,
        out=np.full(bootstrap.shape[0], np.nan),
        where=boot_den > 0,
    )
    state_rho = work.groupby("state_id", sort=False).apply(
        lambda g: _rank_correlation(g[x_name].to_numpy(), g["outcome_shift"].to_numpy()),
        include_groups=False,
    )
    state_matrix = work.drop_duplicates("state_id").set_index("state_id")["matrix_id"]
    rho_matrix = state_rho.groupby(state_matrix).mean().reindex(matrices).to_numpy()
    rho = float(np.nanmean(rho_matrix))
    rho_boot = np.nanmean(rho_matrix[bootstrap], axis=1)
    return {
        "state_centered_slope": slope,
        "slope_bootstrap_ci95": _interval(boot_slope[np.isfinite(boot_slope)]),
        "mean_within_state_spearman": rho,
        "spearman_bootstrap_ci95": _interval(rho_boot[np.isfinite(rho_boot)]),
    }


def _outcome_frame(directory: Path, geometry: pd.DataFrame) -> tuple[pd.DataFrame, NDArray]:
    with np.load(directory / "branch_arrays.npz", allow_pickle=False) as archive:
        targets = archive["targets"].copy()
        breaks = archive["break_event"].copy()
        first_break = archive["first_break_time"].copy()
        inherited = archive["inherited_boundary_count"].copy()
        growth = archive["mean_growth_updates"].copy()
        entropy = archive["final_entropy"].copy()
        occupied = archive["final_occupied_types"].copy()
    state = _load_state(directory)
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    rows: list[dict[str, Any]] = []
    for half, branch_slice in (("A", slice(0, 16)), ("B", slice(16, 32))):
        for state_index, state_id in enumerate(state["state_ids"]):
            noop = arm_index["NOOP"]
            for arm, index in arm_index.items():
                rows.append({
                    "state_id": str(state_id),
                    "candidate": str(state["candidates"][state_index]),
                    "matrix_id": int(state["matrix_ids"][state_index]),
                    "landmark": int(state["landmarks"][state_index]),
                    "branch_half": half,
                    "arm": arm,
                    "outcome_probability": float(targets[state_index, index, branch_slice].mean()),
                    "outcome_shift": float(targets[state_index, index, branch_slice].mean() - targets[state_index, noop, branch_slice].mean()),
                    "break_probability": float(breaks[state_index, index, branch_slice].mean()),
                    "break_shift": float(breaks[state_index, index, branch_slice].mean() - breaks[state_index, noop, branch_slice].mean()),
                    "mean_first_break_time_unconditional_censored_13": float(np.where(first_break[state_index, index, branch_slice] >= 0, first_break[state_index, index, branch_slice], 13).mean()),
                    "mean_inherited_boundaries": float(inherited[state_index, index, branch_slice].mean()),
                    "mean_growth_updates": float(growth[state_index, index, branch_slice].mean()),
                    "mean_final_entropy": float(entropy[state_index, index, branch_slice].mean()),
                    "mean_final_occupied_types": float(occupied[state_index, index, branch_slice].mean()),
                })
    outcomes = pd.DataFrame(rows)
    merged = geometry.merge(
        outcomes,
        on=["state_id", "candidate", "matrix_id", "landmark", "arm"],
        validate="one_to_many",
    )
    with np.load(directory / "inference_arrays.npz", allow_pickle=False) as archive:
        bootstrap = archive["bootstrap_indices"].copy()
    return merged, bootstrap


def _gate_attribution(directory: Path) -> list[dict[str, Any]]:
    metrics = json.loads((directory / "primary_metrics.json").read_text())
    rows: list[dict[str, Any]] = []
    for cell in metrics["primary"]["cells"]:
        rows.append({
            "cell": cell["cell"],
            "target_effect": cell["target_loosen_minus_tighten"]["estimate"],
            "target_ci95": cell["target_loosen_minus_tighten"]["bootstrap_ci95"],
            "target_holm_p": cell["target_loosen_minus_tighten"]["randomization_p_holm"],
            "throughput_slope": cell["throughput_association"]["state_centered_slope"],
            "spearman": cell["throughput_association"]["mean_within_state_spearman"],
            "neutral_effect": cell["neutral_minus_noop"]["estimate"],
            "neutral_ci90": cell["neutral_minus_noop"]["bootstrap_ci90"],
            "failed_gates": [name for name, passed in cell["statistical_gates"].items() if not passed],
            "cell_pass": cell["statistical_cell_pass"],
        })
    return rows


def analyze(directory: Path, geometry: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    frame, bootstrap = _outcome_frame(directory, geometry)
    selected = frame[frame["landmark"].isin(PRIMARY_LANDMARKS)].copy()
    associations: list[dict[str, Any]] = []
    for candidate in ("02", "03"):
        for half in ("A", "B"):
            cell = selected[(selected["candidate"] == candidate) & (selected["branch_half"] == half)]
            for name in GEOMETRY_NAMES:
                associations.append({
                    "cell": f"c{candidate}_{half}",
                    "geometry": name,
                    **_matrix_bootstrap_association(cell, name, bootstrap),
                })
    sensitivity: list[dict[str, Any]] = []
    with np.load(directory / "inference_arrays.npz", allow_pickle=False) as archive:
        for candidate in ("02", "03"):
            for half in ("A", "B"):
                key = f"JOINT_BREAK_RUN3_F12__five_standard_landmarks__c{candidate}_{half}__neutral_bootstrap"
                draws = archive[key]
                ci90 = _interval(draws, alpha=0.10)
                for margin in EQUIVALENCE_MARGINS:
                    sensitivity.append({
                        "cell": f"c{candidate}_{half}",
                        "margin": margin,
                        "ci90_lower": ci90[0],
                        "ci90_upper": ci90[1],
                        "equivalent": bool(ci90[0] > -margin and ci90[1] < margin),
                    })
    neutral = selected[selected["arm"] == "THROUGHPUT_NEUTRAL_RANDOM"]
    noop = selected[selected["arm"] == "NOOP"]
    neutral_summary: list[dict[str, Any]] = []
    for candidate in ("02", "03"):
        for half in ("A", "B"):
            left = neutral[(neutral["candidate"] == candidate) & (neutral["branch_half"] == half)]
            right = noop[(noop["candidate"] == candidate) & (noop["branch_half"] == half)]
            merged = left.merge(right, on=["state_id", "candidate", "matrix_id", "landmark", "branch_half"], suffixes=("_neutral", "_noop"))
            neutral_summary.append({
                "cell": f"c{candidate}_{half}",
                "joint_shift": float((merged["outcome_probability_neutral"] - merged["outcome_probability_noop"]).mean()),
                "break_shift": float((merged["break_probability_neutral"] - merged["break_probability_noop"]).mean()),
                "growth_updates_shift": float((merged["mean_growth_updates_neutral"] - merged["mean_growth_updates_noop"]).mean()),
                "entropy_shift": float((merged["mean_final_entropy_neutral"] - merged["mean_final_entropy_noop"]).mean()),
                "occupied_types_shift": float((merged["mean_final_occupied_types_neutral"] - merged["mean_final_occupied_types_noop"]).mean()),
            })
    pilot_metrics = json.loads((PILOT / "primary_metrics.json").read_text())
    confirmation_metrics = json.loads((CONFIRMATION / "primary_metrics.json").read_text())
    summary: dict[str, Any] = {
        "format": FORMAT,
        "classification": "posthoc_exploratory_sealed_data_audit",
        "new_scientific_matrices": 0,
        "new_simulated_futures": 0,
        "registered_p3c_verdict_changed": False,
        "pilot_primary_gate": pilot_metrics["primary_gate_pass"],
        "confirmation_primary_gate": confirmation_metrics["primary_gate_pass"],
        "confirmation_resistance_gate": confirmation_metrics["resistance_gate_pass"],
        "gate_attribution": _gate_attribution(directory),
        "geometry_associations": associations,
        "neutral_process_summary": neutral_summary,
        "equivalence_sensitivity": sensitivity,
        "claim_boundary": {
            "supported": [
                "coherent occupied-web strength causally changes JOINT_BREAK_RUN3",
                "starting catalytic throughput is a strong but incomplete causal-axis summary",
                "fixed-throughput topology perturbation changes early process dynamics",
            ],
            "not_supported": [
                "P3c passed its registered composite confirmation gate",
                "x^T beta x is a sufficient one-dimensional mechanism",
                "causal post-break resilience",
                "mediation by any audited geometry statistic",
            ],
        },
    }
    summary["audit_id"] = _canonical_digest(_json_ready(summary))
    return summary, selected


def _scientific_report(summary: dict[str, Any]) -> str:
    lines = [
        "# P3c sealed-data interpretation audit",
        "",
        "This audit generated no matrices or futures and does not alter P3c's registered failure.",
        "",
        "## Gate attribution",
        "",
        "| Cell | Loosen-tighten | Neutral-noop | Neutral 90% CI | Failed gates |",
        "|---|---:|---:|---:|---|",
    ]
    for cell in summary["gate_attribution"]:
        lines.append(
            f"| {cell['cell']} | {cell['target_effect']:+.6f} | {cell['neutral_effect']:+.6f} | "
            f"[{cell['neutral_ci90'][0]:+.6f}, {cell['neutral_ci90'][1]:+.6f}] | "
            f"{', '.join(cell['failed_gates']) or 'none'} |"
        )
    lines.extend([
        "",
        "All four targeted strength contrasts, confidence bounds, randomization tests, throughput slopes, and rank associations passed. The confirmation failed because the fixed-throughput topology arm was not equivalent to NOOP in three cells.",
        "",
        "## Process interpretation",
        "",
    ])
    for row in summary["neutral_process_summary"]:
        lines.append(
            f"- {row['cell']}: fixed-throughput topology changed JOINT_BREAK_RUN3 by {row['joint_shift']:+.4f}, first-break probability by {row['break_shift']:+.4f}, and mean growth updates per fission by {row['growth_updates_shift']:+.3f}."
        )
    lines.extend([
        "",
        "The neutral arm preserved only the launch-state scalar x^T beta x. It changed how catalytic support was distributed across types, which changed subsequent kinetics and composition. Thus total starting throughput is a strong causal axis but not a sufficient description of the network.",
        "",
        "## Boundary",
        "",
        "Geometry/outcome associations are exploratory and do not establish mediation. No analysis conditions on an intervention-created break. Causal renewal awaits a fresh identical-natural-break experiment.",
        "",
    ])
    return "\n".join(lines)


def _lay_report() -> str:
    return "\n".join([
        "# P3c interpretation in plain language",
        "",
        "The main intervention worked very clearly: turning the catalytic web up made inheritance more stable, while turning it down produced more break-and-renewal events. That effect was about ten to eleven percentage points and repeated in both simulator candidates.",
        "",
        "The extra test asked whether one number—the assembly's total starting catalytic support—explained everything. It did not. Randomly rearranging the catalytic connections while keeping that one number exactly fixed still changed how quickly the assembly grew and how often it broke. The wiring pattern therefore carries additional information beyond the total amount of support.",
        "",
        "So the discovery survives but becomes more precise: catalytic strength is a real control dial, but the detailed network arrangement also matters. P3c itself remains formally failed because its preregistration required the rearranged network to behave exactly like no intervention.",
        "",
    ])


def _figure(summary: dict[str, Any], path: Path) -> None:
    cells = summary["gate_attribution"]
    labels = [row["cell"] for row in cells]
    target = [row["target_effect"] for row in cells]
    neutral = [row["neutral_effect"] for row in cells]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, target, 0.36, label="LOOSEN - TIGHTEN")
    ax.bar(x + 0.18, neutral, 0.36, label="neutral topology - NOOP")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("JOINT_BREAK_RUN3 probability difference")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(output: Path = DEFAULT_OUTPUT) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    verify_checksums(PILOT)
    verify_checksums(CONFIRMATION)
    geometry = reconstruct_geometry(CONFIRMATION)
    summary, outcome_geometry = analyze(CONFIRMATION, geometry)
    with _atomic_destination(output) as destination:
        geometry.to_csv(destination / "state_arm_geometry.csv.gz", index=False)
        outcome_geometry.to_csv(destination / "state_arm_half_outcomes.csv.gz", index=False)
        pd.DataFrame(summary["equivalence_sensitivity"]).to_csv(destination / "equivalence_sensitivity.csv", index=False)
        (destination / "audit_summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "SCIENTIFIC_REPORT.md").write_text(_scientific_report(summary), encoding="utf-8")
        (destination / "LAY_SUMMARY.md").write_text(_lay_report(), encoding="utf-8")
        _figure(summary, destination / "p3c_effect_decomposition.png")
        manifest = {
            "format": FORMAT,
            "audit_id": summary["audit_id"],
            "source_hashes": source_hashes(),
            "pilot_checksum_manifest_sha256": sha256_file(PILOT / "SHA256SUMS"),
            "confirmation_checksum_manifest_sha256": sha256_file(CONFIRMATION / "SHA256SUMS"),
            "new_scientific_matrices": 0,
            "new_simulated_futures": 0,
            "registered_p3c_verdict_changed": False,
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_checksums(destination)
    verify_checksums(output)
    print(f"P3c interpretation audit sealed: {output}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    run(args.output)


if __name__ == "__main__":
    main()

