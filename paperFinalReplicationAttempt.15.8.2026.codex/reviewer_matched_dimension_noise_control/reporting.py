"""Human-readable outputs for the nuisance-PCA control."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


REVIEWER_COMMENT = """> A matched-dimension noise control would fully close the capacity question. The derangement sensitivity shows the 195-block's alignment matters in T1. The cleanest remaining null is a composite with the nine history variables plus twelve frozen components built from label-permuted or synthetic features on the same fitting pipeline — cheap (no new futures, rescoring only) and it would let you say \"matched 21-input capacity without aligned state information does not reproduce the gain\" across implementations rather than one.
"""


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _fmt(value: float) -> str:
    return f"{float(value):+.6f}"


def _candidate(value: object) -> str:
    return f"{int(value):02d}"


def write_reports(output: Path) -> None:
    cells = pd.read_csv(output / "cell_results.csv")
    sensitivity = pd.read_csv(output / "derangement_sensitivity.csv")
    summaries = (
        sensitivity.groupby(["cohort", "candidate", "half"], sort=False)
        .agg(
            nuisance_median=("nuisance_gain", "median"),
            nuisance_min=("nuisance_gain", "min"),
            nuisance_max=("nuisance_gain", "max"),
            aligned_advantage_min=("aligned_minus_nuisance", "min"),
            aligned_advantage_max=("aligned_minus_nuisance", "max"),
            nuisance_positive_fraction=("nuisance_gain", lambda values: float(np.mean(values > 0))),
        )
        .reset_index()
    )
    all_intervals_positive = bool((cells["difference_ci_low"] > 0).all())
    all_holm_significant = bool((cells["holm_p"] <= 0.05).all())
    every_pairing_below_aligned = bool((sensitivity["aligned_minus_nuisance"] > 0).all())
    status = (
        "The matched-dimension nuisance control did not reproduce the aligned composite gain."
        if all_intervals_positive and every_pairing_below_aligned
        else "The matched-dimension result is mixed and does not close the capacity concern."
    )

    cell_rows: list[str] = []
    for record in cells.itertuples(index=False):
        cell_rows.append(
            "| {cohort} | {candidate} | {half} | {inputs} | {aligned} | {noise} | "
            "{difference} | [{low}, {high}] | {p:.6f} |".format(
                cohort=record.cohort,
                candidate=_candidate(record.candidate),
                half=record.half,
                inputs=int(record.fitted_inputs),
                aligned=_fmt(record.aligned_gain),
                noise=_fmt(record.nuisance_gain),
                difference=_fmt(record.aligned_minus_nuisance),
                low=_fmt(record.difference_ci_low),
                high=_fmt(record.difference_ci_high),
                p=float(record.holm_p),
            )
        )

    sensitivity_rows: list[str] = []
    for record in summaries.itertuples(index=False):
        sensitivity_rows.append(
            "| {cohort} | {candidate} | {half} | {median} | [{low}, {high}] | "
            "[{dlo}, {dhi}] | {fraction:.2f} |".format(
                cohort=record.cohort,
                candidate=_candidate(record.candidate),
                half=record.half,
                median=_fmt(record.nuisance_median),
                low=_fmt(record.nuisance_min),
                high=_fmt(record.nuisance_max),
                dlo=_fmt(record.aligned_advantage_min),
                dhi=_fmt(record.aligned_advantage_max),
                fraction=float(record.nuisance_positive_fraction),
            )
        )

    report = f"""# Matched-dimension nuisance-PCA control: results

**Date:** 2026-08-19  
**Status:** Reviewer-prompted post-hoc retained-outcome analysis  
**Outcome:** {status}

## Reviewer comment

{REVIEWER_COMMENT}

## Executive result

This analysis supplies the exact control left open by the earlier frozen-model
derangement and duplicate-column diagnostics. Each nuisance composite has the
same history block, twelve PCA inputs, final ridge class, `C=0.1`, development
sample, and confirmation outcomes as its aligned counterpart. PCA scores are
reassigned across matrices within phase before fitting, and confirmation uses
an independent reassignment.

Because each reassignment is a permutation of complete rows, the state-block
multiset, covariance, scaler, PCA basis, component count, and component
marginals are preserved exactly. Only correct matrix-state alignment is lost.

Positive gains are reductions in log loss relative to the registered direct
history comparator. "Aligned minus nuisance" is the amount by which the
aligned composite improves over the matched nuisance composite.

| Cohort | Candidate | Half | Inputs | Aligned gain | Nuisance gain | Aligned minus nuisance | 95% matrix CI | Holm p |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(cell_rows)}

All primary aligned-versus-nuisance intervals positive: **{str(all_intervals_positive).lower()}**.  
All 16 one-sided matrix-randomization tests significant after Holm adjustment:
**{str(all_holm_significant).lower()}**.

## Pairing sensitivity

Replicate zero above was frozen as primary. The table summarizes all 32
independently frozen development/confirmation pairings.

| Cohort | Candidate | Half | Median nuisance gain | Nuisance range | Aligned advantage range | Fraction nuisance gain > 0 |
|---|---:|:---:|---:|---:|---:|---:|
{chr(10).join(sensitivity_rows)}

The aligned composite beat the matched nuisance model in every pairing and
cell: **{str(every_pairing_below_aligned).lower()}**.

## What this adds

- The earlier Codex frozen derangement showed that a 21-input model loses its
  gain when its state block is wrong at scoring time, but it did not refit the
  final ridge on nuisance components.
- The Fable duplicate-column control showed that twelve redundant directions
  do not manufacture the gain, but it did not reproduce the PCA fitting
  sequence.
- This control refits the final ridge after deranging the development pairing,
  while keeping the PCA representation and fitted dimension exactly matched.
- The two 40-matrix headline cohorts are exact 21-input controls in independent
  codebases. Codex scaled repeats the 21-input result at 200 matrices. Fable v2
  is the larger revised 20-input (12+8) robustness check and must be labelled
  as such rather than called 21-input.

## Interpretation

The results directly test the generic explanation that the composite wins
merely because twelve additional regularized inputs, or the scaler/PCA/ridge
fitting sequence, are available. Correct alignment of the state-derived
components is required for the tested predictive advantage if the result table
above is uniformly positive.

This does not identify the physical content of those components, and it does
not prove that every imaginable matched-capacity nuisance representation would
fail. The appropriate claim is that matched dimension and fitting sequence
without aligned state information did not reproduce the gain—not that model
capacity has been disproved as a concept.

## Scope and provenance

- No confirmation futures were generated and no outcome was changed.
- Fable development main paths were deterministically replayed to recover the
  already-used training rows; confirmation features and outcomes were retained
  or deterministically reattached without shooting new branches.
- Candidate, branch half, and cohort remained separate.
- This is post hoc: the reviewer concern and confirmation results were already
  known before the control was designed.
- The originating L53/L54 workflow remains outside scope because its required
  machine-readable state/model artifacts are absent from this checkout.
"""
    _atomic_text(output / "RESULTS_REPORT.md", report)

    suggested = """# Suggested manuscript and reviewer-response language

## Methods addition

As a reviewer-prompted matched-dimension control, we reassigned the complete
state-block representation across matrices within candidate and launch phase
using fixed-point-free permutations. Development and confirmation used
independent frozen assignments. This preserves the exact state-feature
multiset, covariance, development scaler, PCA-12 basis, component marginals,
history block, 21-input dimension (20 in the deduplicated Fable-v2 model), and
`C=0.1` ridge pipeline, while breaking the pairing between the twelve state
components and the row's own matrix, history, and outcome. We refit the final
ridge on each deranged development pairing and scored the already-observed
confirmation outcomes without recalibration or new futures.

## Results addition

Across both independent codebases, matched-dimension nuisance composites did
not reproduce the aligned composite's log-loss gain over direct history. The
aligned model also outperformed the nuisance model in every candidate and
branch half and across all 32 frozen pairing realizations. Thus the observed
advantage requires correctly aligned state-derived content in these pipelines,
rather than twelve additional regularized inputs alone.

## Limitation

This reviewer-prompted control is post hoc and tests a specific exact-marginal
misalignment null. It establishes dependence on aligned state-derived
information but does not isolate which physical coordinates carry that
information, rule out every possible representation effect, or extend to the
originating workflow whose machine-readable artifacts were unavailable.
"""
    if not (all_intervals_positive and every_pairing_below_aligned):
        suggested = (
            "# Suggested manuscript and reviewer-response language\n\n"
            "The frozen analysis produced a mixed matched-dimension result. Do not use the "
            "strong draft language until the cell-level exceptions in RESULTS_REPORT.md are resolved.\n"
        )
    _atomic_text(output / "SUGGESTED_TEXT.md", suggested)

