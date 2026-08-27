"""Command-line interface and resumable detached suite runner."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

from .config import PROFILES, config_for_profile
from .ca_campaign import ALL_CONDITIONS, CAMPAIGN_PROFILES, run_ca_campaign
from .ca_carrier_v3 import V3_PROFILES, run_ca_carrier_v3
from .causal_heredity import CAUSAL_PROFILES, launch_detached, run_causal_campaign
from .eca import canonical_rules
from .evolution import run_evolution
from .evolution_gps import run_evolution_gps
from .experiments import compare_atlas, run_atlas, run_phase
from .life import LIFE_PROFILES, life_config_for_profile, run_life
from .life_carrier import CARRIER_PROFILES, run_life_carrier_campaign
from .lineage_field import PUBLIC_PROFILES as LINEAGE_FIELD_PROFILES, run_lineage_field_campaign
from .motif_generalization import run_motif_generalization
from .motif_lineage import PUBLIC_PROFILES as MOTIF_LINEAGE_PROFILES, run_motif_lineage_stage1
from .motif_lineage_stage3 import run_motif_lineage_stage3
from .motif_compression import PHASES as MOTIF_COMPRESSION_PHASES, run_motif_compression
from .motif_localization import PHASES as MOTIF_LOCALIZATION_PHASES, run_motif_localization
from .motif_minimality import ROUNDS as MOTIF_MINIMALITY_ROUNDS, run_motif_minimality
from .motif_minimality_repair import (
    PHASES as MOTIF_MINIMALITY_REPAIR_PHASES,
    run_motif_minimality_repair,
)
from .motif_renewal_repair import (
    PHASES as MOTIF_RENEWAL_REPAIR_PHASES,
    run_motif_renewal_repair,
)
from .motif_regeneration import PHASES as MOTIF_REGENERATION_PHASES, run_motif_regeneration
from .motif_repair import PHASES as MOTIF_REPAIR_PHASES, run_motif_repair
from .life_family import contract_for_condition, run_life_family_condition
from .particle import PARTICLE_GATE_RULES, run_particle
from .reconciliation import run_golden_suite
from .sensitivity import run_sensitivity


def _rules(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _status(root: Path, state: str, stage: str, **extra) -> None:
    payload = {
        "state": state,
        "stage": stage,
        "pid": os.getpid(),
        "updated_unix": time.time(),
        **extra,
    }
    _write_json(root / "STATUS.json", payload)
    print(f"[{state}] {stage}", flush=True)


def run_suite(profile: str, output: Path, workers: int, reference_root: Path | None, resume: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    stages: list[tuple[str, Path, object]] = []
    config = config_for_profile(profile)
    life_config = life_config_for_profile(profile)

    def execute(name: str, directory: Path, function) -> None:
        if resume and (directory / "COMPLETE").exists():
            _status(output, "skipped", name, reason="COMPLETE marker exists")
            return
        _status(output, "running", name)
        function()
        _status(output, "completed_stage", name)

    try:
        execute("eca_atlas", output / "atlas", lambda: run_atlas(config, output / "atlas", workers=workers))
        if reference_root is not None:
            reference = reference_root / "eca_atlas/results/full/eca_rules.csv"
            if reference.exists():
                _status(output, "running", "atlas_reference_comparison")
                comparison = compare_atlas(output / "atlas/eca_rules.csv", reference)
                _write_json(output / "atlas/reference_comparison.json", comparison)
                _status(output, "completed_stage", "atlas_reference_comparison")
        execute("eca_phase", output / "phase", lambda: run_phase(config, output / "phase", workers=workers))
        execute(
            "particle_observer",
            output / "particle",
            lambda: run_particle(config, output / "particle", workers=workers),
        )
        execute("life_objects", output / "life", lambda: run_life(life_config, output / "life"))
        execute(
            "rulial_evolution",
            output / "evolution",
            lambda: run_evolution(output / "atlas/eca_rules.csv", output / "evolution"),
        )
    except BaseException as error:
        _status(output, "failed", "suite", error=repr(error))
        raise
    _status(output, "complete", "suite")
    (output / "COMPLETE").write_text("complete\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plastic-ca")
    sub = parser.add_subparsers(dest="command", required=True)

    atlas = sub.add_parser("atlas", help="run the 88-orbit raw-texture ECA atlas")
    atlas.add_argument("--profile", choices=PROFILES, default="standard")
    atlas.add_argument("--output", type=Path, required=True)
    atlas.add_argument("--workers", type=int, default=1)
    atlas.add_argument("--rules", help="comma-separated canonical representatives")

    phase = sub.add_parser("phase", help="run the five-point noise phase grid")
    phase.add_argument("--profile", choices=PROFILES, default="standard")
    phase.add_argument("--output", type=Path, required=True)
    phase.add_argument("--workers", type=int, default=1)

    particle = sub.add_parser("particle", help="run the figure/ground observer gates")
    particle.add_argument("--profile", choices=PROFILES, default="standard")
    particle.add_argument("--output", type=Path, required=True)
    particle.add_argument("--workers", type=int, default=1)
    particle.add_argument("--rules", help="comma-separated canonical representatives")

    life = sub.add_parser("life", help="run the named Life-object comparison")
    life.add_argument("--profile", choices=LIFE_PROFILES, default="standard")
    life.add_argument("--output", type=Path, required=True)

    compare = sub.add_parser("compare", help="compare a clean-room atlas CSV to permitted reference data")
    compare.add_argument("--ours", type=Path, required=True)
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    evolution = sub.add_parser("evolution", help="run evolutionary search on a completed atlas")
    evolution.add_argument("--atlas", type=Path, required=True)
    evolution.add_argument("--output", type=Path, required=True)

    gps = sub.add_parser("evolution-gps", help="score ECA navigation against a fresh truth atlas")
    gps.add_argument("--dev-atlas", type=Path, required=True)
    gps.add_argument("--truth-atlas", type=Path, required=True)
    gps.add_argument("--output", type=Path, required=True)

    family = sub.add_parser("life-family", help="run one generalized Life-family atlas condition")
    family.add_argument("--condition", choices=ALL_CONDITIONS, default="frozen-b48")
    family.add_argument("--profile", choices=CAMPAIGN_PROFILES, default="reference")
    family.add_argument("--output", type=Path, required=True)
    family.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    family.add_argument("--rules", help="comma-separated 17-bit Life-like rule integers")
    family.add_argument("--resume", action="store_true")

    campaign = sub.add_parser("ca-campaign", help="run/resume the full E23/E24 CA campaign")
    campaign.add_argument("--profile", choices=CAMPAIGN_PROFILES, default="reference")
    campaign.add_argument("--output", type=Path, required=True)
    campaign.add_argument("--reference-root", type=Path, required=True)
    campaign.add_argument(
        "--dev-atlas",
        type=Path,
        default=Path("results/golden-reconciliation/atlas/eca_rules.csv"),
    )
    campaign.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    campaign.add_argument("--resume", action="store_true")
    campaign.add_argument(
        "--stage",
        action="append",
        choices=("truth", "gps", "life"),
        help="run selected stage(s); omit for the complete campaign",
    )

    causal = sub.add_parser(
        "causal-heredity",
        help="run/resume causal common-garden and pedigree tests of Plastic Heredity",
    )
    causal.add_argument("--profile", choices=CAUSAL_PROFILES, default="reference")
    causal.add_argument("--output", type=Path, required=True)
    causal.add_argument(
        "--life-atlas",
        type=Path,
        default=Path("results/ca-campaign-round-1/life-family/frozen-b48/family.csv"),
    )
    causal.add_argument("--workers", type=int, default=max(1, min(20, os.cpu_count() or 1)))
    causal.add_argument("--max-hours", type=float, default=24.0)
    causal.add_argument("--resume", action="store_true")
    causal.add_argument("--detach", action="store_true")
    causal.add_argument(
        "--stage",
        action="append",
        choices=("donors", "common_garden", "pedigrees", "noise", "memory", "transplant"),
        help="run selected stage(s); omitted stages must already have compatible checkpoints",
    )

    carrier = sub.add_parser(
        "life-carrier",
        help="run/resume the Life saturation audit and form-specific carrier campaign",
    )
    carrier.add_argument("--profile", choices=CARRIER_PROFILES, default="reference")
    carrier.add_argument("--output", type=Path, required=True)
    carrier.add_argument(
        "--life-atlas",
        type=Path,
        default=Path("results/ca-campaign-round-1/life-family/frozen-b48/family.csv"),
    )
    carrier.add_argument("--workers", type=int, default=max(1, min(20, os.cpu_count() or 1)))
    carrier.add_argument("--max-hours", type=float, default=48.0)
    carrier.add_argument("--resume", action="store_true")
    carrier.add_argument("--detach", action="store_true")
    carrier.add_argument(
        "--stage",
        action="append",
        choices=("audit", "acquire", "screen", "seal", "holdout", "mapping", "adjudication"),
        help="run selected stage(s); omitted stages must already have compatible checkpoints",
    )

    carrier_v3 = sub.add_parser(
        "ca-carrier-v3",
        help="run/resume the continuous-form narrow and all-CA carrier campaign",
    )
    carrier_v3.add_argument("--profile", choices=V3_PROFILES, default="reference")
    carrier_v3.add_argument("--output", type=Path, required=True)
    carrier_v3.add_argument(
        "--life-atlas",
        type=Path,
        default=Path("results/ca-campaign-round-1/life-family/frozen-b48/family.csv"),
    )
    carrier_v3.add_argument("--workers", type=int, default=max(1, min(20, os.cpu_count() or 1)))
    carrier_v3.add_argument("--max-hours", type=float, default=48.0)
    carrier_v3.add_argument("--resume", action="store_true")
    carrier_v3.add_argument("--detach", action="store_true")
    carrier_v3.add_argument(
        "--stage",
        action="append",
        choices=(
            "calibrate",
            "narrow_acquire",
            "narrow_replay",
            "narrow_confirm",
            "narrow_pedigree",
            "wide_discover",
            "wide_extend",
            "wide_screen",
            "wide_holdout_acquire",
            "wide_holdout",
            "narrow_mechanism",
            "narrow_map_discover",
            "narrow_map_validate",
            "adjudication",
        ),
        help="run selected stage(s); other stages load compatible checkpoints only",
    )

    lineage_field = sub.add_parser(
        "ca-lineage-field",
        help="run/resume the clean-room two-timescale CA lineage-field campaign",
    )
    lineage_field.add_argument("--profile", choices=LINEAGE_FIELD_PROFILES, default="reference")
    lineage_field.add_argument("--output", type=Path, required=True)
    lineage_field.add_argument("--workers", type=int, default=max(1, min(20, os.cpu_count() or 1)))
    lineage_field.add_argument("--max-hours", type=float, default=8.0)
    lineage_field.add_argument("--resume", action="store_true")
    lineage_field.add_argument("--detach", action="store_true")
    lineage_field.add_argument(
        "--stage",
        action="append",
        choices=("calibrate", "seal", "core", "diagnostics", "holdouts", "adjudication"),
        help="run selected stages; omitted trajectory stages load compatible checkpoints",
    )

    motif_lineage = sub.add_parser(
        "ca-motif-lineage",
        help="run/resume a gated clean-room motif-carrier programme stage",
    )
    motif_lineage.add_argument(
        "--stage",
        choices=(
            "upper-bound",
            "generalize",
            "lineage",
            "repair",
            "compression",
            "localization",
            "regeneration",
            "minimality",
            "minimality-repair",
            "renewal-repair",
        ),
        default="upper-bound",
    )
    motif_lineage.add_argument("--profile", choices=MOTIF_LINEAGE_PROFILES, default="reference")
    motif_lineage.add_argument("--output", type=Path, required=True)
    motif_lineage.add_argument(
        "--stage1-root", type=Path, default=Path("results/ca-motif-lineage-stage-1")
    )
    motif_lineage.add_argument(
        "--stage2-root", type=Path, default=Path("results/ca-motif-lineage-stage-2")
    )
    motif_lineage.add_argument(
        "--stage3-root", type=Path, default=Path("results/ca-motif-lineage-stage-3")
    )
    motif_lineage.add_argument(
        "--stage3r-root", type=Path, default=Path("results/ca-motif-lineage-stage-3r")
    )
    motif_lineage.add_argument(
        "--stage4-root", type=Path, default=Path("results/ca-motif-lineage-stage-4")
    )
    motif_lineage.add_argument(
        "--stage5-root", type=Path, default=Path("results/ca-motif-lineage-stage-5")
    )
    motif_lineage.add_argument(
        "--stage5r-root", type=Path, default=Path("results/ca-motif-lineage-stage-5r")
    )
    motif_lineage.add_argument(
        "--stage6-root", type=Path, default=Path("results/ca-motif-lineage-stage-6")
    )
    motif_lineage.add_argument(
        "--stage6ar-root",
        type=Path,
        default=Path("results/ca-motif-lineage-stage-6ar"),
    )
    motif_lineage.add_argument(
        "--round",
        choices=MOTIF_MINIMALITY_ROUNDS,
        default="locality",
        help="run one separately gated Stage-6 round",
    )
    motif_lineage.add_argument(
        "--phase",
        action="append",
        choices=tuple(
            sorted(
                set(MOTIF_REPAIR_PHASES)
                | set(MOTIF_COMPRESSION_PHASES)
                | set(MOTIF_LOCALIZATION_PHASES)
                | set(MOTIF_REGENERATION_PHASES)
                | set(MOTIF_MINIMALITY_REPAIR_PHASES)
                | set(MOTIF_RENEWAL_REPAIR_PHASES)
                | {"all"}
            )
        ),
        help="run selected Stage-3R/4/5 phases; omit for preconfirmation through adjudication",
    )
    motif_lineage.add_argument("--authorize-confirmation", action="store_true")
    motif_lineage.add_argument("--authorize-gate-override", action="store_true")
    motif_lineage.add_argument("--authorize-final-audit", action="store_true")
    motif_lineage.add_argument("--auto-final-audit", action="store_true")
    motif_lineage.add_argument("--workers", type=int, default=max(1, min(20, os.cpu_count() or 1)))
    motif_lineage.add_argument("--max-hours", type=float, default=8.0)
    motif_lineage.add_argument("--resume", action="store_true")
    motif_lineage.add_argument("--detach", action="store_true")

    suite = sub.add_parser("suite", help="run/resume the full CA reconstruction suite")
    suite.add_argument("--profile", choices=PROFILES, default="reference")
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    suite.add_argument("--reference-root", type=Path)
    suite.add_argument("--resume", action="store_true")

    sensitivity = sub.add_parser(
        "sensitivity",
        help="run/resume the clean-room ECA semantic reconciliation campaign",
    )
    sensitivity.add_argument("--design", choices=("overnight", "tiny"), default="overnight")
    sensitivity.add_argument("--output", type=Path, required=True)
    sensitivity.add_argument("--reference-root", type=Path, required=True)
    sensitivity.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    sensitivity.add_argument("--max-hours", type=float, default=12.0)
    sensitivity.add_argument("--resume", action="store_true")

    golden = sub.add_parser(
        "golden-suite",
        help="run/resume the bit-exact golden-trace reconciliation cascade",
    )
    golden.add_argument("--output", type=Path, required=True)
    golden.add_argument("--reference-root", type=Path, required=True)
    golden.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    golden.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "atlas":
        selected = _rules(args.rules)
        run_atlas(config_for_profile(args.profile), args.output, rules=selected, workers=args.workers)
    elif args.command == "phase":
        run_phase(config_for_profile(args.profile), args.output, workers=args.workers)
    elif args.command == "particle":
        selected = _rules(args.rules) or PARTICLE_GATE_RULES
        run_particle(config_for_profile(args.profile), args.output, rules=selected, workers=args.workers)
    elif args.command == "life":
        run_life(life_config_for_profile(args.profile), args.output)
    elif args.command == "compare":
        _write_json(args.output, compare_atlas(args.ours, args.reference))
    elif args.command == "evolution":
        run_evolution(args.atlas, args.output)
    elif args.command == "evolution-gps":
        run_evolution_gps(args.dev_atlas, args.truth_atlas, args.output)
    elif args.command == "life-family":
        settings = CAMPAIGN_PROFILES[args.profile]
        futures = (
            settings["primary_futures"]
            if args.condition in ("frozen-b48", "budget-b256", "area-b1024", "horizon-b1024-t128", "horizon-b1024-t256")
            else settings["stress_futures"]
        )
        run_life_family_condition(
            contract_for_condition(args.condition, futures=futures),
            args.output,
            rules=_rules(args.rules),
            workers=args.workers,
            resume=args.resume,
        )
    elif args.command == "ca-campaign":
        run_ca_campaign(
            args.output,
            args.reference_root,
            dev_atlas=args.dev_atlas,
            workers=args.workers,
            resume=args.resume,
            profile=args.profile,
            stages=args.stage,
        )
    elif args.command == "causal-heredity":
        if args.detach:
            command = [
                sys.executable,
                "-m",
                "plastic_ca",
                "causal-heredity",
                "--profile",
                args.profile,
                "--output",
                str(args.output.resolve()),
                "--life-atlas",
                str(args.life_atlas.resolve()),
                "--workers",
                str(args.workers),
                "--max-hours",
                str(args.max_hours),
            ]
            if args.resume:
                command.append("--resume")
            for stage in args.stage or ():
                command.extend(("--stage", stage))
            pid = launch_detached(command, args.output)
            print(json.dumps({"state": "detached", "pid": pid, "output": str(args.output)}))
        else:
            run_causal_campaign(
                args.output,
                life_atlas=args.life_atlas,
                profile_name=args.profile,
                workers=args.workers,
                max_hours=args.max_hours,
                resume=args.resume,
                selected_stages=args.stage,
            )
    elif args.command == "life-carrier":
        if args.detach:
            command = [
                sys.executable,
                "-m",
                "plastic_ca",
                "life-carrier",
                "--profile",
                args.profile,
                "--output",
                str(args.output.resolve()),
                "--life-atlas",
                str(args.life_atlas.resolve()),
                "--workers",
                str(args.workers),
                "--max-hours",
                str(args.max_hours),
            ]
            if args.resume:
                command.append("--resume")
            for stage in args.stage or ():
                command.extend(("--stage", stage))
            pid = launch_detached(command, args.output)
            print(json.dumps({"state": "detached", "pid": pid, "output": str(args.output)}))
        else:
            run_life_carrier_campaign(
                args.output,
                life_atlas=args.life_atlas,
                profile_name=args.profile,
                workers=args.workers,
                max_hours=args.max_hours,
                resume=args.resume,
                selected_stages=args.stage,
            )
    elif args.command == "ca-carrier-v3":
        if args.detach:
            command = [
                sys.executable,
                "-m",
                "plastic_ca",
                "ca-carrier-v3",
                "--profile",
                args.profile,
                "--output",
                str(args.output.resolve()),
                "--life-atlas",
                str(args.life_atlas.resolve()),
                "--workers",
                str(args.workers),
                "--max-hours",
                str(args.max_hours),
            ]
            if args.resume:
                command.append("--resume")
            for stage in args.stage or ():
                command.extend(("--stage", stage))
            pid = launch_detached(command, args.output)
            print(json.dumps({"state": "detached", "pid": pid, "output": str(args.output)}))
        else:
            run_ca_carrier_v3(
                args.output,
                life_atlas=args.life_atlas,
                profile_name=args.profile,
                workers=args.workers,
                max_hours=args.max_hours,
                resume=args.resume,
                selected_stages=args.stage,
            )
    elif args.command == "ca-lineage-field":
        if args.detach:
            command = [
                sys.executable,
                "-m",
                "plastic_ca",
                "ca-lineage-field",
                "--profile",
                args.profile,
                "--output",
                str(args.output.resolve()),
                "--workers",
                str(args.workers),
                "--max-hours",
                str(args.max_hours),
            ]
            if args.resume:
                command.append("--resume")
            for stage in args.stage or ():
                command.extend(("--stage", stage))
            pid = launch_detached(command, args.output)
            print(json.dumps({"state": "detached", "pid": pid, "output": str(args.output)}))
        else:
            run_lineage_field_campaign(
                args.output,
                profile_name=args.profile,
                workers=args.workers,
                max_hours=args.max_hours,
                resume=args.resume,
                selected_stages=args.stage,
            )
    elif args.command == "ca-motif-lineage":
        if args.detach:
            command = [
                sys.executable,
                "-m",
                "plastic_ca",
                "ca-motif-lineage",
                "--stage",
                args.stage,
                "--profile",
                args.profile,
                "--output",
                str(args.output.resolve()),
                "--stage1-root",
                str(args.stage1_root.resolve()),
                "--stage2-root",
                str(args.stage2_root.resolve()),
                "--stage3-root",
                str(args.stage3_root.resolve()),
                "--stage3r-root",
                str(args.stage3r_root.resolve()),
                "--stage4-root",
                str(args.stage4_root.resolve()),
                "--stage5-root",
                str(args.stage5_root.resolve()),
                "--stage5r-root",
                str(args.stage5r_root.resolve()),
                "--stage6-root",
                str(args.stage6_root.resolve()),
                "--stage6ar-root",
                str(args.stage6ar_root.resolve()),
                "--round",
                args.round,
                "--workers",
                str(args.workers),
                "--max-hours",
                str(args.max_hours),
            ]
            if args.resume:
                command.append("--resume")
            for phase in args.phase or ():
                command.extend(("--phase", phase))
            if args.authorize_confirmation:
                command.append("--authorize-confirmation")
            if args.authorize_gate_override:
                command.append("--authorize-gate-override")
            if args.authorize_final_audit:
                command.append("--authorize-final-audit")
            if args.auto_final_audit:
                command.append("--auto-final-audit")
            pid = launch_detached(command, args.output)
            print(json.dumps({"state": "detached", "pid": pid, "output": str(args.output)}))
        else:
            if args.stage == "upper-bound":
                run_motif_lineage_stage1(
                    args.output,
                    profile_name=args.profile,
                    workers=args.workers,
                    max_hours=args.max_hours,
                    resume=args.resume,
                )
            elif args.stage == "generalize":
                run_motif_generalization(
                    args.output,
                    stage1_root=args.stage1_root,
                    profile_name=args.profile,
                    workers=args.workers,
                    max_hours=args.max_hours,
                    resume=args.resume,
                )
            elif args.stage == "lineage":
                run_motif_lineage_stage3(
                    args.output,
                    stage2_root=args.stage2_root,
                    stage1_root=args.stage1_root,
                    profile_name=args.profile,
                    workers=args.workers,
                    max_hours=args.max_hours,
                    resume=args.resume,
                )
            elif args.stage == "repair":
                run_motif_repair(
                    args.output,
                    stage3_root=args.stage3_root,
                    stage2_root=args.stage2_root,
                    stage1_root=args.stage1_root,
                    profile_name=args.profile,
                    workers=args.workers,
                    max_hours=args.max_hours,
                    resume=args.resume,
                    phases=args.phase,
                    authorize_confirmation=args.authorize_confirmation,
                )
            elif args.stage == "compression":
                run_motif_compression(
                    args.output,
                    stage3r_root=args.stage3r_root,
                    stage3_root=args.stage3_root,
                    stage2_root=args.stage2_root,
                    stage1_root=args.stage1_root,
                    profile_name=args.profile,
                    workers=args.workers,
                    max_hours=args.max_hours,
                    resume=args.resume,
                    phases=args.phase,
                    authorize_confirmation=args.authorize_confirmation,
                )
            elif args.stage == "localization":
                run_motif_localization(
                    args.output,
                    stage4_root=args.stage4_root,
                    stage3r_root=args.stage3r_root,
                    stage3_root=args.stage3_root,
                    stage2_root=args.stage2_root,
                    stage1_root=args.stage1_root,
                    profile_name=args.profile,
                    workers=args.workers,
                    max_hours=args.max_hours,
                    resume=args.resume,
                    phases=args.phase,
                    authorize_confirmation=args.authorize_confirmation,
                )
            elif args.stage == "regeneration":
                run_motif_regeneration(
                    args.output,
                    stage5_root=args.stage5_root,
                    stage4_root=args.stage4_root,
                    stage3r_root=args.stage3r_root,
                    stage3_root=args.stage3_root,
                    stage2_root=args.stage2_root,
                    stage1_root=args.stage1_root,
                    profile_name=args.profile,
                    workers=args.workers,
                    max_hours=args.max_hours,
                    resume=args.resume,
                    phases=args.phase,
                    authorize_confirmation=args.authorize_confirmation,
                )
            elif args.stage == "minimality-repair":
                selected_phases = tuple(args.phase or ("audit",))
                if len(selected_phases) != 1:
                    raise ValueError("Stage-6A-R runs exactly one --phase per invocation")
                run_motif_minimality_repair(
                    args.output,
                    phase=selected_phases[0],
                    stage6_root=args.stage6_root,
                    stage5r_root=args.stage5r_root,
                    stage5_root=args.stage5_root,
                    stage4_root=args.stage4_root,
                    stage3r_root=args.stage3r_root,
                    stage3_root=args.stage3_root,
                    stage2_root=args.stage2_root,
                    stage1_root=args.stage1_root,
                    profile_name=args.profile,
                    workers=args.workers,
                    max_hours=args.max_hours,
                    resume=args.resume,
                    authorize_confirmation=args.authorize_confirmation,
                )
            elif args.stage == "renewal-repair":
                selected_phases = tuple(args.phase or ("all",))
                if len(selected_phases) != 1:
                    raise ValueError(
                        "Stage-6B-R runs one --phase selection or --phase all"
                    )
                run_motif_renewal_repair(
                    args.output,
                    phase=selected_phases[0],
                    stage6ar_root=args.stage6ar_root,
                    stage6_root=args.stage6_root,
                    stage5r_root=args.stage5r_root,
                    stage5_root=args.stage5_root,
                    stage4_root=args.stage4_root,
                    stage3r_root=args.stage3r_root,
                    stage3_root=args.stage3_root,
                    stage2_root=args.stage2_root,
                    stage1_root=args.stage1_root,
                    profile_name=args.profile,
                    workers=args.workers,
                    max_hours=args.max_hours,
                    resume=args.resume,
                    auto_final_audit=args.auto_final_audit,
                )
            else:
                run_motif_minimality(
                    args.output,
                    round_name=args.round,
                    stage5r_root=args.stage5r_root,
                    stage5_root=args.stage5_root,
                    stage4_root=args.stage4_root,
                    stage3r_root=args.stage3r_root,
                    stage3_root=args.stage3_root,
                    stage2_root=args.stage2_root,
                    stage1_root=args.stage1_root,
                    profile_name=args.profile,
                    workers=args.workers,
                    max_hours=args.max_hours,
                    resume=args.resume,
                    authorize_gate_override=args.authorize_gate_override,
                    authorize_final_audit=args.authorize_final_audit,
                )
    elif args.command == "suite":
        run_suite(args.profile, args.output, args.workers, args.reference_root, args.resume)
    elif args.command == "sensitivity":
        run_sensitivity(
            args.output,
            args.reference_root,
            workers=args.workers,
            resume=args.resume,
            design_name=args.design,
            max_hours=args.max_hours,
        )
    elif args.command == "golden-suite":
        run_golden_suite(
            args.output,
            args.reference_root,
            workers=args.workers,
            resume=args.resume,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
