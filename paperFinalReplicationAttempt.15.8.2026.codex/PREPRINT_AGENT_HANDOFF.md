# Preprint revision handoff

**Prepared:** 2026-08-20  
**Purpose:** authoritative editing guide for the agent revising the plastic-heredity preprint  
**Manuscript snapshot audited:** [current manuscript](../PRE_PRINT_PAPER_DRAFT.md),
1,052 lines, SHA-256
`3ed4a58114c77ffe39b7b2413883cad1aebfd17e8fb628fef9d3f5404cf66daa`,
modified 2026-08-19 11:28 UTC

This document is the entry point for preprint editing. The older
`REPLICATION_REPORT.md` covers the initial white-room Phi-r reconstruction; it
is not a complete summary of the later plastic-heredity and reviewer-response
program.

## Non-negotiable scope

- Use only evidence documented under this replication folder and the current
  manuscript's already cited, verified campaigns.
- Do not use, cite, paraphrase, or import results from `NewIdeas/` or other
  provisional strict-8 work. Those materials were hypothesis-generating only.
- Do not describe a reviewer-prompted post-hoc analysis as preregistered,
  confirmatory, or prospective confirmation. “Frozen before fresh futures” is
  accurate for the rulebook intervention; “preregistered” is not.
- Keep F12 and strict coherent eight separate. F12 is a break followed by a
  three-boundary renewal; strict eight additionally requires a later inherited
  run of eight, all 28 selected-daughter pair comparisons above threshold, and
  separation from the old pre-break anchor.
- At first substantive mention, use **selected-lineage parent-to-daughter
  inheritance**. The endpoint follows one contract-selected daughter at each
  fission and does not measure both-daughter reproductive fidelity.
- Keep nats per stochastic future distinct from bits per transition. Keep
  candidate and frozen branch-half cells visible; do not replace them with a
  pooled significance claim.

## Evidence classes

| Class | Meaning | How it may be written |
|---|---|---|
| Registered/confirmatory | Protocol or gate fixed before the relevant untouched campaign | May support the manuscript's principal claims, subject to its registered gate |
| Prospective occurrence replication | Strict-eight occurrence tested on new matrices in two clean-room implementations | Supports occurrence and matrix breadth, not prediction or mechanism |
| Reviewer-prompted retained-outcome analysis | Protocol fixed after the concern arose; no new futures | Label post hoc; use as robustness or limitation-narrowing evidence |
| Reviewer-prompted fresh-future intervention | Hypothesis and protocol fixed after prior results, before new intervention futures | Label post hoc mechanistic follow-up; causal only for the exact edits and retained states |
| Internal diagnosis | Designed to understand an already observed result | Prefer supplement/discussion; never promote isolated cells to a new headline claim |

## Current manuscript: what is already addressed

Line numbers below refer to the audited manuscript snapshot and will move after
editing.

| Reviewer issue | Current manuscript status | Required action |
|---|---|---|
| Direct-history model class and hyperparameters | Addressed at lines 175–179: the same standardized L2 logistic estimator, `lbfgs`, `C=0.1`, nine listed variables, versus nine plus twelve frozen PCA scores | Keep; do not add a duplicate paragraph |
| Ordered/sequence history comparator | Addressed at lines 373–377 | Keep the post-hoc/model-dependent qualification |
| Expressive nonlinear history comparator | Addressed at line 377 | Keep; source report gives the exact 0.01748–0.03847-nat range |
| Matched 21-input capacity control | Addressed at lines 446, 654, and 852 | Keep the narrow claim: matched dimension and fitting sequence did not reproduce the gain |
| Survival and landmark acquisition | Addressed in Results at line 354 | Add or retain an explicit limitation that the estimand is conditional on surviving, observable landmark states |
| Selected daughter rather than both daughters | Addressed in the strict endpoint definition and limitation 6 | Preserve at abstract/first-mention level as well |
| Threshold and metric sensitivity | **Outdated.** Lines 290–292, 424 onward, and Appendix H describe the earlier smaller grid | Replace rather than append, using the completed 2026-08-19 extension |
| Strict-event non-degeneracy | Missing | Add the concentration/diversity characterization and matched same-state comparison |
| Strict-event target-matched prediction | Current text correctly says prospective prediction was not established, but it predates the post-hoc diagnostics | Preserve the prospective verdict; optionally add the narrower post-hoc result |
| Strict-event causal mechanism | Missing | If included, report the null generic diversity edits and partial rulebook-alignment result together |

Authoritative sources for the controls already incorporated are the
[nonlinear-history report](reviewer_nonlinear_history_comparator/artifacts/output/RESULTS_REPORT.md),
[matched-dimension report](reviewer_matched_dimension_noise_control/artifacts/output/RESULTS_REPORT.md),
and [state-acquisition audit](reviewer_state_acquisition_survivorship/REVIEW_NOTE.md).
Use them to check existing wording; do not paste their suggested passages a
second time.

## Material that must not enter the evidence chain

The current manuscript's lines 644–648 describe an exploratory composome
follow-up, including the 94–97% convergence result, phase-conditioned composome
sets, drift/switching, and a “catalytic rulebook” interpretation. These
statements overlap the quarantined exploratory program and conflict with the
instruction not to quote unconfirmed exploratory work.

Remove those empirical claims unless the preprint team can point to a separate,
sealed, in-scope analysis and provenance record. The surrounding literature
discussion at line 642 can remain. If “rulebook” is retained as a metaphor, it
must be tied only to the verified post-hoc analysis below and must not imply a
permanent attractor, written memory, or completed strict-eight mechanism.

## Required manuscript updates

### 1. Replace the older threshold grid

Use the completed reviewer-prompted grid:

- F12 inheritance thresholds: 0.85, 0.875, 0.90, 0.925, 0.95;
- horizons: 8, 10, 12, 16;
- renewal runs: 2–5;
- strict coherent windows: 6, 8, 10;
- old-anchor source thresholds: 0.80, 0.85, 0.90; and
- cosine plus percentile-matched Bray–Curtis.

Across 640 F12 metric-by-candidate-by-half comparisons, 628 favored the frozen
composite and 502 had wholly positive whole-matrix 95% intervals. All twelve
negative point estimates occurred in candidate 03 at the joint 0.95/F16 stress
corner. At the registered F12 shape, all eight cosine/Bray–Curtis candidate-by-
half gains were positive with positive intervals.

Do not describe strict eight as metric invariant. At its registered shape,
Bray–Curtis prevalence was about 0.20% versus 1.84–2.10% for cosine, and event
Jaccard overlap was only 0.063–0.073. The cosine-trained strict predictor did
not transfer to Bray labels.

Sources:

- [results](reviewer_threshold_metric_sensitivity_extension/artifacts/output/RESULTS_REPORT.md)
- [manuscript-ready language](reviewer_threshold_metric_sensitivity_extension/artifacts/output/SUGGESTED_TEXT.md)
- [figures](reviewer_threshold_metric_sensitivity_extension/artifacts/output/)

### 2. Add the strict-event non-degeneracy result

For registered-cosine confirmation events, the eight selected daughters had
mean effective species numbers 5.36/5.49, occupied-type counts 11.86/12.07,
and largest-species shares 0.583/0.566 in candidates 02/03. Only 0.13%/0.04%
of events were at least 80% one-species dominated in all eight daughters;
1.06%/0.30% met the analogous two-species criterion.

This does **not** mean concentration is irrelevant. Against matched negative
branches from the same starting state that still reached a post-break run of
eight, registered-cosine events had 5.48/5.30 fewer effective species,
4.82/4.61 fewer occupied types, a 0.219/0.215 larger leading-species share,
about 0.109 less adjacent total-variation turnover, and 7.26/6.73 fewer growth
updates. Every listed interval excluded zero. The correct interpretation is:
the events are not trivial one- or two-species copies, but they are strongly
concentrated and low-turnover relative to hard same-state controls.

Sources:

- [geometry and non-degeneracy report](reviewer_strict_event_geometry_audit/RESULTS_REPORT.md)
- [concrete suggested text](reviewer_strict_event_geometry_audit/SUGGESTED_TEXT.md)
- [matched-effect figure](reviewer_strict_event_geometry_audit/artifacts/output/figures/matched_nondegeneracy_effects.png)

### 3. Update the strict-eight prediction/mechanism wording

Do not overturn the registered verdict: reproducible prospective prediction of
strict eight was not established. The later analyses refine why.

The frozen post-hoc stage decomposition found that concentration and the state
block each improved prediction of the first-break gate in all 12 cells. Gains
for coherence conditional on a run of eight were mixed, and the old-anchor gate
was not robustly predicted. Direct concentration/flattening and richness
contraction/expansion edits produced 0/24 passing strict-event cells. Thus
starting concentration is a reliable marker of break propensity but the tested
generic diversity axes are not a demonstrated cause of the complete event.

Source: [strict-eight mechanism diagnosis](reviewer_strict8_prediction_mechanism_diagnosis/DIAGNOSTIC_REPORT.md).

### 4. Optionally add the post-hoc rulebook follow-up

This analysis used no `NewIdeas` input. It derived one deterministic expected-
flow form per beta matrix from the frozen simulator equations, fitted models on
development matrices, scored confirmation once, and generated 896,000 fresh
common-random-stream intervention futures from the 2,000 retained confirmation
states.

For the registered cosine endpoint:

- adding deterministic rulebook features beyond history, concentration, and
  the retained state block improved coherence-given-run-8 log loss by 0.04708
  nats and strict-eight log loss by 0.00909 nats; all four candidate-by-half
  cells passed;
- adding opposite-candidate, opposite-half empirical holding features improved
  the same targets by 0.05744 and 0.01340 nats; all four cells passed; and
- matrix-level opposite-candidate correlations averaged 0.913 for coherence
  and 0.796 for strict eight.

The empirical holding score uses sibling futures and is therefore a mechanistic
world calibration, not a deployable launch-time predictor.

Moving four molecules toward rather than away from the beta-derived form
increased eight-way coherent outcomes by 0.00775–0.00909 across the four cells
(mean +0.00820, approximately 5.2% to 6.0%). All four intervals were positive.
There was no supported effect on first break, eight uninterrupted inherited
boundaries, or the complete strict-eight event. The frozen global causal gate
remained false: candidate 03 achieved the complete four-molecule toward edit in
89.7% of states, narrowly below the 90% feasibility requirement, so only 2/4
cells received a formal pass. Alignment differences decayed strongly by
fission 8 and were unresolved by fission 32.

The allowed interpretation is that beta-conditioned alignment contributes to
the **mutual-coherence component** of strict eight. Do not write that the
rulebook causes strict eight, that the full mechanism is solved, or that a
stable attractor was demonstrated.

Sources:

- [technical report](reviewer_strict8_rulebook_holding_probe/DIAGNOSTIC_REPORT.md)
- [manuscript guidance](reviewer_strict8_rulebook_holding_probe/SUGGESTED_TEXT.md)
- [verification audit](reviewer_strict8_rulebook_holding_probe/artifacts/output/verification_audit.json)

## Recommended insertion map

| Manuscript location | Action |
|---|---|
| Abstract | Keep the confirmed 1.70–2.11% strict-occurrence claim and selected-lineage wording. Do not add a claim that strict eight is predicted or caused by rulebook alignment |
| Methods: endpoint sensitivity | Replace the old grid with the exact completed grid and separately calibrated Bray–Curtis cutoffs |
| Results: strict coherent eight | Add one compact non-degeneracy paragraph and one metric-dependence paragraph after the occurrence table |
| Results: prediction | Keep the original algorithmic F12 claim; keep the expressive-history and matched-capacity controls already present; do not duplicate them |
| Results or supplement: strict-eight mechanism | Add the stage decomposition and null diversity edits; add the rulebook result only with its post-hoc/global-gate qualification |
| Discussion: strict eight | Replace “prediction and intervention remain open” with the distinction between unresolved prospective full-event prediction and partial post-hoc evidence for the coherence component |
| Discussion: composome/rulebook passage | Remove the quarantined exploratory empirical claims at audited lines 644–648; retain literature context only |
| Limitations | Add metric dependence, concentration relative to matched controls, conditional state acquisition, selected-lineage scope, and the fact that no intervention increased complete strict-eight occurrence |
| Supplement | Put complete grids, gate waterfalls, matched non-degeneracy estimates, post-hoc model families, feasibility rules, and verification identifiers here |

## Claims ledger

### Supported with the stated qualification

- Strict coherent-eight occurrence prospectively reproduced across two
  clean-room implementations at 1.70–2.11% under the registered cosine rule.
- Events were spread across 119–143 of 200 matrices per cell.
- Registered-cosine events were not universally one- or two-species dominated.
- Events were nevertheless substantially more concentrated and lower-turnover
  than matched same-state run-8 controls.
- The F12 composite advantage survived the tested sequence, nonlinear-history,
  and matched-dimension nuisance controls.
- Post-hoc beta-derived alignment predicted strict-event components and
  causally altered eight-way coherence under the exact four-molecule edits.

### Not supported

- Metric-invariant strict-event identity or prevalence.
- Full branching reproductive fidelity.
- A reproducibly successful prospective strict-eight predictor.
- A causal increase in complete strict-eight occurrence from concentration,
  richness, or rulebook-alignment edits.
- A permanent compositional attractor, assembly-written memory, or solved
  break–capture–hold mechanism.
- A model-free proof that no history learner can match the composite.
- A claim that matched model capacity is irrelevant in every representation;
  only the tested exact-marginal nuisance null was rejected.

## Verification and editing discipline

Before accepting manuscript changes:

1. Check every number against the linked result report or CSV, not a chat
   summary.
2. Preserve the post-hoc label in Results, Methods, figure captions, and the
   reviewer response—not only in a limitation.
3. Verify that the updated threshold grid replaces all older definitions in
   Methods, Results, captions, and Appendix H.
4. Search the manuscript for `0.88`, `0.92`, `run lengths seven`, `run lengths
   two, three and four`, and the exploratory 94–97% composome claim.
5. Ensure the final abstract does not imply both-daughter replication,
   metric-invariant strict eight, successful strict-event prediction, or a
   completed causal mechanism.
