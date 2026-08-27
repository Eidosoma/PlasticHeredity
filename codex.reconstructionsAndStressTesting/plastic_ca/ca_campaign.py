"""Resumable E23/E24 clean-room CA campaign orchestration."""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from .e19 import E19Contract
from .evolution_gps import run_evolution_gps
from .life_family import (
    NAMED_RULE_NOTATIONS,
    compare_life_family,
    contract_for_condition,
    fixed_scale_subset,
    load_rule_registry,
    parse_life_rule,
    run_life_family_condition,
)
from .reconciliation import run_e19_atlas
from .stats import spearman


PRIMARY_CONDITIONS = ("frozen-b48", "budget-b256", "area-b1024")
HORIZON_CONDITIONS = ("horizon-b1024-t128", "horizon-b1024-t256")
SCALE_CONDITIONS = ("scale-16", "scale-32")
LAUNCH_CONDITIONS = (
    "launch-v2-b48",
    "launch-v2-b1024",
    "launch-broad-b48",
    "launch-broad-b1024",
)
ALL_CONDITIONS = PRIMARY_CONDITIONS + HORIZON_CONDITIONS + SCALE_CONDITIONS + LAUNCH_CONDITIONS

CAMPAIGN_PROFILES: dict[str, dict[str, int]] = {
    "smoke": {"full_rules": 8, "subset_rules": 8, "primary_futures": 2, "stress_futures": 1},
    "pilot": {"full_rules": 32, "subset_rules": 32, "primary_futures": 8, "stress_futures": 4},
    "reference": {"full_rules": 1024, "subset_rules": 128, "primary_futures": 64, "stress_futures": 32},
}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _status(root: Path, state: str, stage: str, **extra: Any) -> None:
    payload = {
        "state": state,
        "stage": stage,
        "pid": os.getpid(),
        "updated_unix": time.time(),
        **extra,
    }
    _atomic_json(root / "STATUS.json", payload)
    progress = f" {extra['completed']}/{extra['total']}" if "completed" in extra else ""
    print(f"[{state}] {stage}{progress}", flush=True)


def _read_summary(root: Path, condition: str) -> dict[str, Any]:
    return json.loads((root / "life-family" / condition / "family_summary.json").read_text(encoding="utf-8"))


def _csv_vectors(path: Path, rules: set[int] | None = None) -> dict[str, list[float]]:
    values = {"strict": [], "break_by_8": [], "median_gen_sweeps": [], "library_size": []}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if rules is not None and int(row["rule"]) not in rules:
                continue
            for field_name in values:
                values[field_name].append(float(row[field_name]))
    return values


def _rank_comparison(left: Path, right: Path, rules: Sequence[int] | None = None) -> dict[str, float]:
    selected = set(rules) if rules is not None else None
    a = _csv_vectors(left, selected)
    b = _csv_vectors(right, selected)
    result: dict[str, float] = {}
    for field_name in a:
        left_values = a[field_name]
        right_values = b[field_name]
        # Rank correlation is mathematically undefined for two identical
        # constant vectors.  For a robustness comparison they are perfect
        # agreement, not evidence of disagreement.
        if left_values == right_values and len(set(left_values)) <= 1:
            result[field_name] = 1.0
        else:
            result[field_name] = spearman(left_values, right_values)
    return result


def _life_named(gates: dict[str, Any]) -> dict[str, Any] | None:
    return next((row for row in gates["named_rules"] if row["name"] == "life"), None)


def _adjudicate_campaign(root: Path, subset: Sequence[int], profile: str) -> dict[str, Any]:
    primary = {name: _read_summary(root, name) for name in PRIMARY_CONDITIONS}
    family_laws = all(
        bool(summary["gates"]["gate_smoothness"]) and bool(summary["gates"]["gate_heavy_tail"])
        for summary in primary.values()
    )
    area = primary["area-b1024"]["gates"]
    clock_rescued = (
        int(area["n_capable"]) >= 20
        and float(area["family_clock_iqr_width"]) >= 1.0
        and bool(area["gate_boundary_of_order"])
    ) if profile == "reference" else None
    life_rows = [_life_named(summary["gates"]) for summary in primary.values()]
    life_maintainer = all(
        row is not None
        and int(row["library_rank"]) <= 103
        and float(row["strict"]) < 0.005
        for row in life_rows
    ) if profile == "reference" else None

    life_root = root / "life-family"
    launch_pairs = {
        "v2-b48": ("frozen-b48", "launch-v2-b48"),
        "v2-b1024": ("area-b1024", "launch-v2-b1024"),
        "broad-b48": ("frozen-b48", "launch-broad-b48"),
        "broad-b1024": ("area-b1024", "launch-broad-b1024"),
    }
    launch_correlations = {
        label: _rank_comparison(
            life_root / primary_name / "family.csv",
            life_root / alternative / "family.csv",
            subset,
        )
        for label, (primary_name, alternative) in launch_pairs.items()
    }
    launch_laws = all(
        _read_summary(root, alternative)["gates"]["gate_smoothness"]
        == primary[primary_name]["gates"]["gate_smoothness"]
        and _read_summary(root, alternative)["gates"]["gate_heavy_tail"]
        == primary[primary_name]["gates"]["gate_heavy_tail"]
        for primary_name, alternative in launch_pairs.values()
    )
    launch_robust = all(
        values["strict"] >= 0.70 and values["break_by_8"] >= 0.70
        for values in launch_correlations.values()
    ) and launch_laws
    scale_correlation = _rank_comparison(
        life_root / "scale-16" / "family.csv",
        life_root / "scale-32" / "family.csv",
        subset,
    )
    scale_stable = scale_correlation["strict"] >= 0.70
    return {
        "family_laws_robust": family_laws,
        "clock_rescued": clock_rescued,
        "life_remains_maintainer": life_maintainer,
        "launch_robustness": launch_robust,
        "scale_stability": scale_stable,
        "launch_rank_correlations": launch_correlations,
        "scale_rank_correlations": scale_correlation,
        "primary_condition_gates": {name: summary["gates"] for name, summary in primary.items()},
    }


def _reference_gate_comparison(ours: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    keys = ("gate_smoothness", "gate_heavy_tail", "gate_life_rare_book", "gate_boundary_of_order")
    ours_gates = ours["gates"]
    reference_gates = reference["gates"]
    matches = {key: bool(ours_gates[key]) == bool(reference_gates[key]) for key in keys}
    return {"matches": matches, "all_match": all(matches.values())}


def _render_report(results: dict[str, Any]) -> str:
    life = results["life_family"]
    frozen = life["conditions"]["frozen-b48"]["gates"]
    area = life["conditions"]["area-b1024"]["gates"]
    evolution = results["evolution_gps"]
    comparison = life.get("retained_comparison", {})
    smoothness = frozen["smoothness_ratio"]
    smoothness_text = f"{smoothness:.3f}" if smoothness is not None else "undefined (zero random-null overlap)"
    return "\n".join(
        (
            "# Full CA Campaign — E23 GPS and E24 Life-family Atlas",
            "",
            f"Profile: `{results['profile']}`. State: **complete**.",
            "",
            "## E23",
            "",
            f"- Selection finds the boundary: `{evolution['evolution']['gate_selection_finds_boundary']}`.",
            f"- Original sticky-walk gate: `{evolution['evolution']['gate_sticky_walk']}`.",
            f"- Nonempty all-edge smoothness ratio: "
            f"`{evolution['evolution']['stickiness_all_edges_nonempty_reanalysis']['ratio']:.3f}`.",
            f"- Fresh-truth GPS: `{evolution['gps']['gate_gps']}`; planned "
            f"`{evolution['gps']['planned_success']:.3f}`, random `{evolution['gps']['random_walk_success']:.3f}`, "
            f"oracle `{evolution['gps']['oracle_success']:.3f}`.",
            "",
            "## E24 frozen replication",
            "",
            f"- Smoothness: `{smoothness_text}` (gate `{frozen['gate_smoothness']}`).",
            f"- Heavy-tail share: `{frozen['heavy_tail_share']:.3f}` (gate `{frozen['gate_heavy_tail']}`).",
            f"- Life strict `{frozen['life_strict']}` and library rank `{frozen['life_library_rank']}`; "
            f"rare-book gate `{frozen['gate_life_rare_book']}`.",
            f"- Frozen clock gate: `{frozen['gate_boundary_of_order']}`.",
            f"- Retained gate verdict match: `{comparison.get('gate_verdicts', {}).get('all_match')}`.",
            "",
            "## Clock and robustness follow-up",
            "",
            f"- Area-scaled clock IQR `{area['family_clock_iqr']}`, capable median "
            f"`{area['capable_median_clock']}`, gate `{area['gate_boundary_of_order']}`.",
            f"- Family laws robust: `{life['campaign_gates']['family_laws_robust']}`.",
            f"- Clock rescued: `{life['campaign_gates']['clock_rescued']}`.",
            f"- Life remains a maintainer: `{life['campaign_gates']['life_remains_maintainer']}`.",
            f"- Launch robustness: `{life['campaign_gates']['launch_robustness']}`.",
            f"- Scale stability: `{life['campaign_gates']['scale_stability']}`.",
            "",
            "Numerical identity is not claimed without Fable's Life RNG and launch traces. Exact E23 truth and all "
            "Life-family contracts, launch rows, checkpoints, and comparisons are retained beside this report.",
            "",
        )
    )


def run_ca_campaign(
    output: Path,
    reference_root: Path,
    *,
    dev_atlas: Path,
    workers: int = 16,
    resume: bool = False,
    profile: str = "reference",
    stages: Sequence[str] | None = None,
) -> dict[str, Any]:
    if profile not in CAMPAIGN_PROFILES:
        raise ValueError(f"unknown campaign profile {profile!r}")
    if not dev_atlas.exists():
        raise FileNotFoundError(f"the exact dev atlas is required: {dev_atlas}")
    output.mkdir(parents=True, exist_ok=True)
    selected_stages = set(stages or ("truth", "gps", "life"))
    settings = CAMPAIGN_PROFILES[profile]
    registry = load_rule_registry()
    subset_all = fixed_scale_subset(registry, max(settings["full_rules"], settings["subset_rules"]))
    full_rules = tuple(sorted(subset_all[: settings["full_rules"]])) if settings["full_rules"] < 1024 else registry
    subset = fixed_scale_subset(registry, settings["subset_rules"])
    started = time.time()

    try:
        truth_dir = output / "evolution" / "truth"
        if "truth" in selected_stages and not (resume and (truth_dir / "COMPLETE").exists()):
            _status(output, "running", "e23_fresh_truth")
            truth_contract = E19Contract(rng_tag="rulial-evo-truth-v1")
            if profile == "smoke":
                truth_contract = replace(truth_contract, n_seeds=4, futures_per_seed=4)
            elif profile == "pilot":
                truth_contract = replace(truth_contract, n_seeds=8, futures_per_seed=16)
            run_e19_atlas(
                truth_dir,
                truth_contract,
                workers=workers,
                resume=resume,
                status_root=output,
            )
        truth_atlas = truth_dir / "eca_rules.csv"
        if "gps" in selected_stages and not (resume and (output / "evolution" / "gps" / "COMPLETE").exists()):
            if not truth_atlas.exists():
                raise FileNotFoundError("fresh truth atlas is unavailable; include the truth stage")
            _status(output, "running", "e23_evolution_gps")
            run_evolution_gps(dev_atlas, truth_atlas, output / "evolution" / "gps")

        if "life" in selected_stages:
            for condition in ALL_CONDITIONS:
                condition_dir = output / "life-family" / condition
                primary = condition in PRIMARY_CONDITIONS
                condition_rules = full_rules if primary else subset
                futures = settings["primary_futures"] if primary or condition in HORIZON_CONDITIONS else settings["stress_futures"]
                contract = contract_for_condition(condition, futures=futures)
                existing_summary = condition_dir / "family_summary.json"
                can_resume_complete = False
                if resume and (condition_dir / "COMPLETE").exists() and existing_summary.exists():
                    existing = json.loads(existing_summary.read_text(encoding="utf-8"))
                    can_resume_complete = existing.get("contract_digest") == contract.digest
                if can_resume_complete:
                    _status(output, "skipped", f"e24_{condition}", reason="COMPLETE marker exists")
                    continue
                stage_started = time.time()

                def progress(completed: int, total: int, rule: int, _worker_elapsed: int) -> None:
                    elapsed = max(1e-9, time.time() - stage_started)
                    rate = completed / elapsed
                    _status(
                        output,
                        "running",
                        f"e24_{condition}",
                        condition=condition,
                        completed=completed,
                        total=total,
                        last_rule=rule,
                        rules_per_second=rate,
                        elapsed_seconds=elapsed,
                        eta_seconds=(total - completed) / rate if rate else None,
                    )

                _status(output, "running", f"e24_{condition}", completed=0, total=len(condition_rules))
                run_life_family_condition(
                    contract,
                    condition_dir,
                    rules=condition_rules,
                    workers=workers,
                    resume=resume,
                    progress=progress,
                )

        gps_summary_path = output / "evolution" / "gps" / "evolution_summary.json"
        condition_paths = [output / "life-family" / name / "family_summary.json" for name in ALL_CONDITIONS]
        if not gps_summary_path.exists() or not all(path.exists() for path in condition_paths):
            partial = {
                "experiment": "full_ca_campaign_e23_e24",
                "profile": profile,
                "state": "selected_stages_complete",
                "selected_stages": sorted(selected_stages),
                "gps_ready": gps_summary_path.exists(),
                "life_conditions_ready": sum(path.exists() for path in condition_paths),
                "life_conditions_total": len(condition_paths),
            }
            _atomic_json(output / "PARTIAL_RESULTS.json", partial)
            _status(
                output,
                "selected_stages_complete",
                "ca_campaign",
                profile=profile,
                selected_stages=sorted(selected_stages),
                gps_ready=gps_summary_path.exists(),
                life_conditions_ready=sum(path.exists() for path in condition_paths),
                life_conditions_total=len(condition_paths),
            )
            return partial

        gps_summary = json.loads(
            gps_summary_path.read_text(encoding="utf-8")
        )
        condition_summaries = {name: _read_summary(output, name) for name in ALL_CONDITIONS}
        campaign_gates = _adjudicate_campaign(output, subset, profile)
        retained_comparison: dict[str, Any] = {}
        retained_csv = reference_root / "life_family" / "results" / "full" / "family.csv"
        retained_summary = reference_root / "life_family" / "results" / "full" / "family_summary.json"
        if retained_csv.exists():
            retained_comparison["per_rule"] = compare_life_family(
                output / "life-family" / "frozen-b48" / "family.csv",
                retained_csv,
            )
        if retained_summary.exists():
            retained_comparison["gate_verdicts"] = _reference_gate_comparison(
                condition_summaries["frozen-b48"],
                json.loads(retained_summary.read_text(encoding="utf-8")),
            )
        results = {
            "experiment": "full_ca_campaign_e23_e24",
            "profile": profile,
            "elapsed_seconds": time.time() - started,
            "cleanroom": {
                "sibling_code_read": False,
                "permitted_reference_root": str(reference_root),
                "dev_atlas": str(dev_atlas),
                "dev_atlas_sha256": _sha256(dev_atlas),
            },
            "evolution_gps": gps_summary,
            "life_family": {
                "conditions": condition_summaries,
                "campaign_gates": campaign_gates,
                "retained_comparison": retained_comparison,
                "full_rule_count": len(full_rules),
                "stress_subset": list(subset),
            },
        }
        _atomic_json(output / "RESULTS.json", results)
        _atomic_text(output / "REPORT.md", _render_report(results))
        manifest = {
            "results_sha256": _sha256(output / "RESULTS.json"),
            "report_sha256": _sha256(output / "REPORT.md"),
            "dev_atlas_sha256": _sha256(dev_atlas),
            "registry_sha256": _sha256(Path(__file__).with_name("data") / "life_family_rule_ids.json"),
            "condition_contract_digests": {
                name: condition_summaries[name]["contract_digest"] for name in ALL_CONDITIONS
            },
        }
        _atomic_json(output / "MANIFEST.json", manifest)
        _atomic_text(output / "COMPLETE", "complete\n")
        _status(output, "complete", "ca_campaign", elapsed_seconds=time.time() - started)
        return results
    except BaseException as error:
        _status(output, "failed", "ca_campaign", error=repr(error), elapsed_seconds=time.time() - started)
        raise
