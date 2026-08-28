"""Resumable reconciliation campaign built around the bit-exact E19 engine."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import mean, median
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .config import config_for_profile
from .e19 import (
    E19Contract,
    E19RuleResult,
    GOLDEN_FIXTURE,
    PINNED_NUMPY,
    evaluate_e19_rule,
    require_pinned_numpy,
    validate_golden_fixture,
)
from .eca import CLASS_1, CLASS_3_CORE, canonical_rule, canonical_rules
from .evolution import run_evolution
from .experiments import RuleResult, adjudicate_atlas
from .life import life_config_for_profile, run_life
from .particle_e19 import run_e19_particle
from .stats import spearman


DIAGNOSTIC_RULES = (8, 13, 35, 110, 172)
PHASE_ETAS = (0.0025, 0.005, 0.01, 0.02, 0.04)
RAW_FIELDS = ("strict", "break_by_8", "median_gen_sweeps", "mean_survival")


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
    progress = ""
    if "completed" in extra and "total" in extra:
        progress = f" {extra['completed']}/{extra['total']}"
    print(f"[{state}] {stage}{progress}", flush=True)


def _task(arguments: tuple[int, E19Contract]) -> dict[str, Any]:
    rule, contract = arguments
    return evaluate_e19_rule(rule, contract).to_dict()


def _load_checkpoint(path: Path, digest: str) -> E19RuleResult | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract_digest") != digest:
        return None
    return E19RuleResult.from_dict(payload["result"])


def run_rule_set(
    contract: E19Contract,
    checkpoint_root: Path,
    rules: Sequence[int],
    *,
    workers: int,
    resume: bool,
    progress: Callable[[int, int, int], None] | None = None,
) -> list[E19RuleResult]:
    """Evaluate independent rules with atomic, digest-checked checkpoints."""

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results: dict[int, E19RuleResult] = {}
    missing: list[int] = []
    for rule in rules:
        path = checkpoint_root / f"rule-{rule:03d}.json"
        found = _load_checkpoint(path, contract.digest) if resume else None
        if found is None:
            missing.append(rule)
        else:
            results[rule] = found
    completed = len(results)
    total = len(rules)
    if progress and completed:
        progress(completed, total, -1)

    def save(result: E19RuleResult) -> None:
        nonlocal completed
        _atomic_json(
            checkpoint_root / f"rule-{result.rule:03d}.json",
            {"contract_digest": contract.digest, "result": result.to_dict()},
        )
        results[result.rule] = result
        completed += 1
        if progress:
            progress(completed, total, result.rule)

    if workers <= 1:
        for rule in missing:
            save(evaluate_e19_rule(rule, contract))
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_task, (rule, contract)): rule for rule in missing}
            for future in as_completed(futures):
                save(E19RuleResult.from_dict(future.result()))
    return [results[rule] for rule in sorted(rules)]


def _legacy_result(result: E19RuleResult) -> RuleResult:
    return RuleResult(
        rule=result.rule,
        wolfram_class=result.wolfram_class,
        disputed=result.disputed,
        strict=result.strict,
        break_by_8=result.break_by_8,
        median_gen_sweeps=result.median_gen_sweeps,
        mean_survival=result.mean_survival,
        form_supports=result.form_supports,
        descriptors=result.descriptors,
        n_futures=result.n_futures,
    )


def _write_atlas_csv(path: Path, results: Sequence[E19RuleResult]) -> dict[int, int]:
    # The retained registry is append-only in deterministic encounter order:
    # ascending rule, then seed/future discovery order.  Form IDs therefore
    # encode first appearance, not numeric support-mask rank.
    registry: dict[int, int] = {}
    for result in results:
        for support in result.form_supports:
            if support and support not in registry:
                registry[support] = len(registry)
    fields = (
        "rule",
        "wolfram_class",
        "disputed",
        "strict",
        "break_by_8",
        "median_gen_sweeps",
        "mean_survival",
        "library_size",
        "library",
        "support_masks",
        "d_lambda",
        "d_asymmetry",
        "d_quiescent",
        "d_sensitivity",
        "n_futures",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            library = sorted(result.library)
            writer.writerow(
                {
                    "rule": result.rule,
                    "wolfram_class": result.wolfram_class,
                    "disputed": result.disputed,
                    "strict": result.strict,
                    "break_by_8": result.break_by_8,
                    "median_gen_sweeps": result.median_gen_sweeps,
                    "mean_survival": result.mean_survival,
                    "library_size": len(library),
                    "library": "|".join(str(value) for value in sorted(registry[item] for item in library)),
                    "support_masks": "|".join(str(value) for value in library),
                    "d_lambda": result.descriptors["lambda"],
                    "d_asymmetry": result.descriptors["asymmetry"],
                    "d_quiescent": result.descriptors["quiescent_defect"],
                    "d_sensitivity": result.descriptors["sensitivity"],
                    "n_futures": result.n_futures,
                }
            )
    os.replace(temporary, path)
    return registry


def _read_csv(path: Path, key: Callable[[dict[str, str]], Any]) -> dict[Any, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {key(row): row for row in csv.DictReader(handle)}


def compare_raw_atlas(ours: Path, reference: Path) -> dict[str, Any]:
    left = _read_csv(ours, lambda row: int(row["rule"]))
    right = _read_csv(reference, lambda row: int(row["rule"]))
    rules = sorted(set(left) & set(right))
    field_matches: dict[str, int] = {}
    mismatches: dict[str, list[int]] = {}
    correlations: dict[str, float] = {}
    mean_absolute_errors: dict[str, float] = {}
    for field in RAW_FIELDS:
        bad = [rule for rule in rules if float(left[rule][field]) != float(right[rule][field])]
        mismatches[field] = bad
        field_matches[field] = len(rules) - len(bad)
        correlations[field] = spearman(
            [float(left[rule][field]) for rule in rules],
            [float(right[rule][field]) for rule in rules],
        )
        mean_absolute_errors[field] = mean(
            abs(float(left[rule][field]) - float(right[rule][field])) for rule in rules
        )
    library_bad = [rule for rule in rules if left[rule]["library"] != right[rule]["library"]]
    exact = (
        len(rules) == len(left) == len(right) == 88
        and not any(mismatches.values())
        and not library_bad
    )
    secondary_strong = (
        correlations["strict"] >= 0.95
        and correlations["break_by_8"] >= 0.95
        and mean_absolute_errors["strict"] <= 0.02
        and mean_absolute_errors["break_by_8"] <= 0.02
        and correlations["median_gen_sweeps"] >= 0.95
        and correlations["mean_survival"] >= 0.95
    )
    return {
        "exact_reproduction": exact,
        "secondary_strong_reproduction": secondary_strong,
        "n_common": len(rules),
        "field_exact_matches": field_matches,
        "field_mismatch_rules": mismatches,
        "library_exact_matches": len(rules) - len(library_bad),
        "library_mismatch_rules": library_bad,
        "spearman": correlations,
        "mean_absolute_error": mean_absolute_errors,
        "thresholds": {
            "endpoint_spearman_min": 0.95,
            "strict_break_mae_max": 0.02,
        },
    }


def run_e19_atlas(
    output: Path,
    contract: E19Contract,
    *,
    workers: int,
    resume: bool,
    status_root: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    rules = canonical_rules()

    def progress(completed: int, total: int, rule: int) -> None:
        if status_root is not None:
            _status(status_root, "running", "eca_atlas", completed=completed, total=total, last_rule=rule)

    results = run_rule_set(
        contract,
        output / "checkpoints",
        rules,
        workers=workers,
        resume=resume,
        progress=progress,
    )
    registry = _write_atlas_csv(output / "eca_rules.csv", results)
    gates = adjudicate_atlas([_legacy_result(result) for result in results], contract.rng_tag)
    summary = {
        "experiment": "e19_bit_exact_atlas",
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation": "clean-room NumPy shrinking-batch PCG64",
        },
        "contract": contract.to_dict(),
        "contract_digest": contract.digest,
        "n_rules": len(results),
        "n_futures": sum(result.n_futures for result in results),
        "n_form_ids": len(registry),
        "gates": gates,
        "death_counts": {
            key: sum(result.death_counts.get(key, 0) for result in results)
            for key in sorted({key for result in results for key in result.death_counts})
        },
    }
    _atomic_json(output / "atlas_summary.json", summary)
    _atomic_text(output / "COMPLETE", "complete\n")
    return summary


def _unimodal(values: Sequence[float], tolerance: float = 0.01) -> bool:
    for peak in range(len(values)):
        left = all(values[index + 1] + tolerance >= values[index] for index in range(peak))
        right = all(
            values[index + 1] <= values[index] + tolerance
            for index in range(peak, len(values) - 1)
        )
        if left and right:
            return True
    return False


def _phase_gates(by_eta: dict[float, dict[int, E19RuleResult]]) -> dict[str, Any]:
    per_eta: list[dict[str, Any]] = []
    robust = True
    for eta in PHASE_ETAS:
        band = [result for result in by_eta[eta].values() if 0.005 <= result.strict <= 0.5]
        class2_fraction = sum(result.wolfram_class == 2 for result in band) / len(band) if band else 0.0
        core_in_band = sum(result.rule in CLASS_3_CORE for result in band)
        passed = bool(band) and class2_fraction > 0.5 and core_in_band == 0
        robust &= passed
        per_eta.append(
            {
                "eta": eta,
                "n_band": len(band),
                "band_rules": [result.rule for result in band],
                "band_class2_fraction": class2_fraction,
                "core3_in_band": core_in_band,
                "cell_pass": passed,
            }
        )
    ever = sorted(
        rule
        for rule in canonical_rules()
        if any(0.005 <= by_eta[eta][rule].strict <= 0.5 for eta in PHASE_ETAS)
    )
    profiles = {rule: [by_eta[eta][rule].strict for eta in PHASE_ETAS] for rule in ever}
    unimodal = {rule: _unimodal(values) for rule, values in profiles.items()}
    fraction = sum(unimodal.values()) / len(unimodal) if unimodal else 0.0
    awakened = sorted(
        rule for rule in CLASS_1 if any(0.005 <= by_eta[eta][rule].strict <= 0.5 for eta in PHASE_ETAS)
    )
    return {
        "gate_regime_robustness": robust,
        "gate_ridge_law": fraction >= 2 / 3,
        "gate_class1_awakening": bool(awakened),
        "awakened_class1": awakened,
        "n_ever_capable": len(ever),
        "unimodal_fraction": fraction,
        "peak_eta_by_rule": {
            str(rule): PHASE_ETAS[max(range(len(PHASE_ETAS)), key=lambda index: profiles[rule][index])]
            for rule in ever
        },
        "per_eta": per_eta,
        "rule110_by_eta": [
            {
                "eta": eta,
                "strict": by_eta[eta][110].strict,
                "break_by_8": by_eta[eta][110].break_by_8,
            }
            for eta in PHASE_ETAS
        ],
    }


def _write_phase_csv(path: Path, rows: Sequence[tuple[float, E19RuleResult]]) -> None:
    fields = ("break_by_8", "eta", "median_gen_sweeps", "rule", "strict", "in_band", "wolfram_class")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for eta, result in rows:
            writer.writerow(
                {
                    "break_by_8": result.break_by_8,
                    "eta": eta,
                    "median_gen_sweeps": result.median_gen_sweeps,
                    "rule": result.rule,
                    "strict": result.strict,
                    "in_band": 0.005 <= result.strict <= 0.5,
                    "wolfram_class": result.wolfram_class,
                }
            )
    os.replace(temporary, path)


def compare_phase(ours: Path, reference: Path) -> dict[str, Any]:
    key = lambda row: (float(row["eta"]), int(row["rule"]))
    left = _read_csv(ours, key)
    right = _read_csv(reference, key)
    points = sorted(set(left) & set(right))
    fields = ("strict", "break_by_8", "median_gen_sweeps")
    mismatches = {
        field: [list(point) for point in points if float(left[point][field]) != float(right[point][field])]
        for field in fields
    }
    return {
        "exact_reproduction": len(points) == len(left) == len(right) == 440 and not any(mismatches.values()),
        "n_common": len(points),
        "field_exact_matches": {field: len(points) - len(mismatches[field]) for field in fields},
        "field_mismatch_points": mismatches,
        "spearman": {
            field: spearman(
                [float(left[point][field]) for point in points],
                [float(right[point][field]) for point in points],
            )
            for field in fields
        },
        "mean_absolute_error": {
            field: mean(
                abs(float(left[point][field]) - float(right[point][field]))
                for point in points
            )
            for field in fields
        },
    }


def run_e19_phase(
    output: Path,
    base: E19Contract,
    *,
    workers: int,
    resume: bool,
    status_root: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    all_rows: list[tuple[float, E19RuleResult]] = []
    by_eta: dict[float, dict[int, E19RuleResult]] = {}
    for point_index, eta in enumerate(PHASE_ETAS):
        contract = replace(base, flip_noise=eta, copy_error=1.5 * eta)

        def progress(completed: int, total: int, rule: int, *, eta: float = eta) -> None:
            if status_root is not None:
                _status(
                    status_root,
                    "running",
                    "eca_phase",
                    eta=eta,
                    eta_index=point_index + 1,
                    eta_total=len(PHASE_ETAS),
                    completed=completed,
                    total=total,
                    last_rule=rule,
                )

        results = run_rule_set(
            contract,
            output / "checkpoints" / f"eta-{eta:g}",
            canonical_rules(),
            workers=workers,
            resume=resume,
            progress=progress,
        )
        by_eta[eta] = {result.rule: result for result in results}
        all_rows.extend((eta, result) for result in results)
    all_rows.sort(key=lambda item: (item[0], item[1].rule))
    _write_phase_csv(output / "phase.csv", all_rows)
    summary = {
        "experiment": "e19_bit_exact_phase",
        "elapsed_seconds": time.time() - started,
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "base_contract_digest": base.digest,
        "n_points": len(all_rows),
        "gates": _phase_gates(by_eta),
    }
    _atomic_json(output / "phase_summary.json", summary)
    _atomic_text(output / "COMPLETE", "complete\n")
    return summary


def _reference_gate_comparison(ours: Path, reference: Path) -> dict[str, Any]:
    if not ours.exists() or not reference.exists():
        return {"available": False}
    left = json.loads(ours.read_text(encoding="utf-8"))
    right = json.loads(reference.read_text(encoding="utf-8"))
    return {"available": True, "ours": left.get("gates", left), "reference": right.get("gates", right)}


def _render_report(output: Path, manifest: dict[str, Any], stages: dict[str, Any]) -> None:
    golden = stages.get("golden_replay", {})
    atlas = stages.get("atlas_comparison", {})
    phase = stages.get("phase_comparison", {})
    particle = stages.get("particle_observer", {})
    particle_110 = next(
        (row for row in particle.get("rows", []) if int(row.get("rule", -1)) == 110),
        {},
    )
    life = stages.get("life_round5", {})
    named = life.get("named", {})
    evolution = stages.get("rulial_evolution", {}).get("evolution", {})
    lines = [
        "# Golden-trace reconciliation report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Headline",
        "",
        (
            f"Golden replay: **{'PASS' if golden.get('passed') else 'FAIL'}** "
            f"({golden.get('sweep_checks', 0)}/907 sweeps, {golden.get('spectrum_checks', 0)}/15 spectra)."
        ),
        (
            f"Raw 88-rule atlas exact reproduction: **{'YES' if atlas.get('exact_reproduction') else 'NO'}**."
        ),
        (
            f"Five-point, 440-cell phase exact reproduction: **{'YES' if phase.get('exact_reproduction') else 'NO'}**."
        ),
        "",
        "## Core contract",
        "",
        f"- Contract digest: `{manifest['contract_digest']}`",
        f"- NumPy: `{manifest['environment']['numpy']}` (pinned `{PINNED_NUMPY}`)",
        "- Launch: 16 shared heterogeneous rows; no preparation.",
        "- Generation: post-rule noise, realized activity, timeout-or-monochrome boundary death.",
        "- Observation: first completed generation, terminal pre-copy final4; copy draws unconditional.",
        "- RNG: one shrinking PCG64 batch stream per rule and seed.",
        "",
        "## Exact endpoint counts",
        "",
        f"- Atlas: `{json.dumps(atlas.get('field_exact_matches', {}), sort_keys=True)}`",
        f"- Atlas libraries: `{atlas.get('library_exact_matches', 0)}/88`",
        f"- Phase: `{json.dumps(phase.get('field_exact_matches', {}), sort_keys=True)}`",
        f"- Phase Spearman: `{json.dumps(phase.get('spearman', {}), sort_keys=True)}`",
        f"- Phase MAE: `{json.dumps(phase.get('mean_absolute_error', {}), sort_keys=True)}`",
        "",
        "## Downstream stages",
        "",
        f"- Particle rule 110 strict: `{particle_110.get('strict')}`; redemption gate: "
        f"`{particle.get('gates', {}).get('gate_redemption_110')}`. Exact particle numerics are not "
        "claimed because four domain launch rows remain undisclosed.",
        "- Life supports: "
        f"glider `{named.get('glider', {}).get('form_support')}`, "
        f"blinker `{named.get('blinker', {}).get('form_support')}`, "
        f"toad `{named.get('toad', {}).get('form_support')}`; all three Life gates pass.",
        "- Evolution: selection-boundary gate "
        f"`{evolution.get('gate_selection_finds_boundary')}`; sticky-walk gate "
        f"`{evolution.get('gate_sticky_walk')}` (both match the retained decisions).",
        "",
        "The full per-rule, per-cell, gate, and reference comparisons are retained in `RESULTS.json`.",
        "",
    ]
    _atomic_text(output / "REPORT.md", "\n".join(lines))


def run_golden_suite(
    output: Path,
    reference_root: Path,
    *,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    """Run the full ambitious cascade; scientific failures never short-circuit it."""

    output.mkdir(parents=True, exist_ok=True)
    require_pinned_numpy()
    contract = E19Contract()
    supplied_json = reference_root / "golden_traces_for_codex.json"
    supplied_md = reference_root / "GOLDEN_TRACES_FOR_CODEX.md"
    manifest = {
        "started_unix": time.time(),
        "command_pid": os.getpid(),
        "workers": workers,
        "resume": resume,
        "reference_root": str(reference_root.resolve()),
        "contract": contract.to_dict(),
        "contract_digest": contract.digest,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "evidence": {
            "supplied_json_sha256": _sha256(supplied_json),
            "supplied_markdown_sha256": _sha256(supplied_md),
            "vendored_fixture_sha256": _sha256(GOLDEN_FIXTURE),
            "vendored_fixture": str(GOLDEN_FIXTURE),
        },
        "cleanroom_exclusions": ["sibling src directories", "sibling tests", "sibling scripts"],
    }
    _atomic_json(output / "MANIFEST.json", manifest)
    stages: dict[str, Any] = {}
    errors: dict[str, str] = {}

    _status(output, "running", "golden_replay")
    golden = validate_golden_fixture()
    stages["golden_replay"] = golden
    _atomic_json(output / "golden" / "replay.json", golden)
    if not (
        golden["passed"]
        and golden["sweep_checks"] == golden["expected_sweep_checks"] == 907
        and golden["spectrum_checks"] == golden["expected_spectrum_checks"] == 15
    ):
        _status(output, "failed", "golden_replay", errors=golden["errors"][:20])
        raise RuntimeError("hard golden replay prerequisite failed")

    def attempt(name: str, function: Callable[[], Any]) -> Any:
        _status(output, "running", name)
        try:
            value = function()
            stages[name] = value
            _status(output, "completed_stage", name)
            return value
        except BaseException as error:
            errors[name] = repr(error)
            stages[name] = {"error": repr(error)}
            _status(output, "stage_error_continuing", name, error=repr(error))
            return None

    diagnostics = attempt(
        "diagnostic_rules",
        lambda: [
            result.to_dict()
            for result in run_rule_set(
                contract,
                output / "diagnostics" / "checkpoints",
                DIAGNOSTIC_RULES,
                workers=min(workers, len(DIAGNOSTIC_RULES)),
                resume=resume,
            )
        ],
    )
    if diagnostics is not None:
        _atomic_json(output / "diagnostics" / "rules.json", diagnostics)

    atlas_summary = attempt(
        "eca_atlas",
        lambda: run_e19_atlas(
            output / "atlas", contract, workers=workers, resume=resume, status_root=output
        ),
    )
    atlas_csv = output / "atlas" / "eca_rules.csv"
    reference_atlas = reference_root / "eca_atlas" / "results" / "full" / "eca_rules.csv"
    if atlas_csv.exists() and reference_atlas.exists():
        comparison = compare_raw_atlas(atlas_csv, reference_atlas)
        stages["atlas_comparison"] = comparison
        _atomic_json(output / "atlas" / "reference_comparison.json", comparison)

    phase_summary = attempt(
        "eca_phase",
        lambda: run_e19_phase(
            output / "phase", contract, workers=workers, resume=resume, status_root=output
        ),
    )
    phase_csv = output / "phase" / "phase.csv"
    reference_phase = reference_root / "eca_phase" / "results" / "full" / "phase.csv"
    if phase_csv.exists() and reference_phase.exists():
        comparison = compare_phase(phase_csv, reference_phase)
        stages["phase_comparison"] = comparison
        _atomic_json(output / "phase" / "reference_comparison.json", comparison)

    # These three stages remain clean-room reconstructions.  They deliberately
    # execute after endpoint disagreement; only missing upstream artifacts can
    # make an individual stage unavailable.
    particle = attempt(
        "particle_observer",
        lambda: run_e19_particle(
            contract,
            output / "particle",
            workers=workers,
            resume=resume,
        ),
    )
    stages["particle_gate_comparison"] = _reference_gate_comparison(
        output / "particle" / "particle_gates.json",
        reference_root / "particle_observer" / "results" / "full" / "particle_gates.json",
    )

    life = attempt(
        "life_round5",
        lambda: run_life(life_config_for_profile("reference"), output / "life"),
    )
    stages["life_gate_comparison"] = _reference_gate_comparison(
        output / "life" / "observer.json",
        reference_root / "glider_test" / "results" / "full" / "observer.json",
    )

    if atlas_csv.exists():
        evolution = attempt(
            "rulial_evolution",
            lambda: run_evolution(atlas_csv, output / "evolution"),
        )
    else:
        errors["rulial_evolution"] = "atlas CSV unavailable"
        stages["rulial_evolution"] = {"error": errors["rulial_evolution"]}
    stages["evolution_gate_comparison"] = _reference_gate_comparison(
        output / "evolution" / "evolution_summary.json",
        reference_root / "rulial_evolution" / "results" / "full" / "evolution_summary.json",
    )

    result = {
        "completed_unix": time.time(),
        "manifest": manifest,
        "stages": stages,
        "software_errors": errors,
        "complete": not errors,
        "scientific_stages_all_attempted": True,
    }
    _atomic_json(output / "RESULTS.json", result)
    _render_report(output, manifest, stages)
    if errors:
        _atomic_text(output / "COMPLETE_WITH_ERRORS", "complete with stage errors\n")
        _status(output, "complete_with_errors", "suite", errors=errors)
    else:
        _atomic_text(output / "COMPLETE", "complete\n")
        _status(output, "complete", "suite")
    return result
