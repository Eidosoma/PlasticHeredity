"""Report generation for the nonlinear history-only comparator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(value: float, digits: int = 5) -> str:
    return f"{float(value):.{digits}f}"


def _model_name(value: str) -> str:
    return {
        "spline_interaction_pca12_ridge": "Spline/interaction PCA12 ridge",
        "gradient_boosted_history": "Boosted history tree",
    }.get(value, value)


def _table(rows: list[list[Any]], headers: list[str]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def write_reports(output_dir: Path) -> None:
    selection = pd.read_csv(output_dir / "model_selection.csv")
    cells = pd.read_csv(output_dir / "cell_results.csv")
    comparisons = pd.read_csv(output_dir / "comparisons.csv")
    verification = json.loads((output_dir / "verification.json").read_text(encoding="utf-8"))

    primary = comparisons[
        (comparisons["role"] == "primary")
        & (comparisons["baseline"] == "selected_history")
    ].copy()
    strong_pass = bool(
        (primary["gain_nats"] > 0).all()
        and (primary["ci95_lower"] > 0).all()
        and (primary["holm_p"] < 0.05).all()
    )

    selection_rows = []
    for row in selection.itertuples(index=False):
        selection_rows.append(
            [
                row.cohort,
                f"{int(row.candidate):02d}",
                _model_name(row.selected_family),
                _fmt(row.spline_cv_log_loss),
                int(row.selected_tree_leaves),
                _fmt(row.selected_tree_cv_log_loss),
            ]
        )

    primary_rows = []
    for row in primary.itertuples(index=False):
        primary_rows.append(
            [
                row.cohort,
                f"{int(row.candidate):02d}",
                row.half,
                _model_name(row.selected_family),
                _fmt(row.baseline_log_loss),
                _fmt(row.composite_log_loss),
                ("+" if row.gain_nats >= 0 else "") + _fmt(row.gain_nats),
                f"[{_fmt(row.ci95_lower)}, {_fmt(row.ci95_upper)}]",
                _fmt(row.holm_p, 6),
            ]
        )

    model_rows = []
    primary_cells = cells[cells["role"] == "primary"]
    for row in primary_cells.itertuples(index=False):
        model_rows.append(
            [
                row.cohort,
                f"{int(row.candidate):02d}",
                row.half,
                _fmt(row.direct_log_loss),
                _fmt(row.spline_log_loss),
                _fmt(row.tree_log_loss),
                _fmt(row.composite_log_loss),
            ]
        )

    direct_improvement = comparisons[
        (comparisons["role"] == "primary")
        & (comparisons["baseline"] == "selected_history")
    ]["history_minus_direct_gain"]
    history_better_all = bool((direct_improvement > 0).all())
    min_gain = float(primary["gain_nats"].min())
    max_gain = float(primary["gain_nats"].max())
    min_ci = float(primary["ci95_lower"].min())

    outcome = (
        "The composite retained a statistically supported advantage over the "
        "development-selected expressive history model in every primary cell."
        if strong_pass
        else "The composite did not retain the predeclared all-cell advantage over the selected expressive history model."
    )

    report = f"""# Nonlinear history-only comparator: results

**Date:** 2026-08-19  
**Status:** Reviewer-prompted post-hoc retained-outcome analysis  
**Outcome:** {outcome}

## Question

Does current composition/network context add held-out F12 predictive information
beyond a reasonably expressive model of the observable hereditary past?

## Design

Two candidate-separated history-only challengers were fitted without opening
confirmation outcomes:

1. an exactly input-matched ridge with the registered direct block plus twelve
   development-fitted components of a truncated-cubic-spline and all-pairwise-
   interaction history library; and
2. a shallow gradient-boosted history tree, with `3`, `7`, or `15` leaves per
   boosting stage selected by five-fold development-matrix-grouped
   cross-validation.

For each implementation/candidate, the lower development-CV-loss family was
frozen as the selected expressive comparator. It was then scored without
recalibration on the same retained confirmation outcomes and fixed branch
halves as the registered direct and composite models. Primary inference uses
the two scaled cohorts, whole-matrix bootstrap intervals, paired matrix-sign
randomization, and Holm adjustment across eight candidate/half cells.

## Development-only selection

{_table(selection_rows, ["Cohort", "Candidate", "Selected family", "Spline CV loss", "Tree leaves", "Tree CV loss"])}

## Primary selected-history comparison

Positive gain means lower confirmation log loss for the frozen composite.

{_table(primary_rows, ["Cohort", "Candidate", "Half", "Selected history", "History loss", "Composite loss", "Gain (nats)", "95% matrix CI", "Holm p"])}

All eight gains, interval lower bounds, and adjusted tests passed: **{str(strong_pass).lower()}**.  
Gain range: **{min_gain:.5f} to {max_gain:.5f} nats**.  
Smallest 95% lower bound: **{min_ci:.5f} nats**.

## Model score inventory

{_table(model_rows, ["Cohort", "Candidate", "Half", "Direct", "Spline/interactions", "Boosted tree", "Composite"])}

The development-selected nonlinear history comparator beat the registered
direct comparator in every primary confirmation cell: **{str(history_better_all).lower()}**.
This check matters: the result is not obtained merely by comparing the
composite with a nonlinear model that failed to improve on the original H8/H9
ridge.

## Interpretation

The exact-dimension spline control addresses the narrow fitted-capacity
question while allowing nonlinear univariate history effects and pairwise
interactions. The boosted-tree control gives observable history substantially
more flexible functional form and is not claimed to have exactly matched
effective capacity. Development-only family selection prevents choosing the
better history model after seeing confirmation scores.

If the primary all-cell result is positive, the appropriate claim is:

> Within two retained clean-room implementations and the tested nonlinear
> history families, present state/network context retained held-out F12
> predictive information beyond an expressive model of observable hereditary
> history.

This still does not identify which physical part of the aligned 195-coordinate
block carries that information, prove conditional information in a model-free
sense, or rule out every possible history-only learner.

## Scope and provenance

- No new principal lineages or confirmation futures were generated.
- All model and complexity selection used development matrices only.
- Confirmation outcomes were used only after the protocol and selection rule
  were fixed.
- This is post hoc because the reviewer concern and prior confirmation results
  were already known; it is not described as prospective preregistration.
- The originating L53/L54 workflow is excluded because its row-level machine-
  readable development and confirmation artifacts are unavailable locally.
- Independent test 1 uses its registered H9 representation. The revised large
  independent-test-2 cohort uses its registered deduplicated H8 representation.

## Verification

All verification checks passed: **{str(all(verification.values())).lower()}**.
See `verification.json`, `model_selection.csv`, `cv_fold_scores.csv`,
`cell_results.csv`, and `comparisons.csv` for the complete audit trail.
"""
    (output_dir / "RESULTS_REPORT.md").write_text(report, encoding="utf-8")

    suggested = f"""# Suggested manuscript and reviewer-response language

## Methods

As a reviewer-prompted post-hoc robustness analysis, we fitted two nonlinear
history-only comparators separately by implementation and candidate. The first
added twelve development-fitted principal components of a fixed truncated-
cubic-spline and pairwise-interaction expansion to the registered direct
history variables and used the composite's final ridge setting, thereby
matching its fitted input count. The second was a shallow gradient-boosted tree
using only the registered history variables, with tree size selected by
five-fold development-matrix-grouped cross-validation. The lower-development-
loss family was frozen and scored without recalibration on the retained
confirmation branches. No new futures were generated.

## Results

{outcome} The composite-over-selected-history gain ranged from {min_gain:.4f}
to {max_gain:.4f} nats across the eight scaled implementation-by-candidate-by-
half cells; the smallest whole-matrix 95% lower bound was {min_ci:.4f} nats.
This supports incremental predictive content beyond the tested nonlinear
history families while remaining post-hoc and model-dependent.

## Reviewer response

We agree that a linear nine-variable ridge alone is not sufficient to support
the broader physical interpretation. We therefore added an exactly input-
matched spline/interaction ridge and a development-selected shallow boosted
tree, both restricted to observable pre-launch history and evaluated with
matrix-grouped training and inference. The selected nonlinear comparator was
frozen before retained confirmation scoring. {outcome} We have kept the claim
narrow: this is evidence beyond the tested expressive history families, not a
model-free identification of the physical content of the state/network block.
"""
    (output_dir / "SUGGESTED_TEXT.md").write_text(suggested, encoding="utf-8")
