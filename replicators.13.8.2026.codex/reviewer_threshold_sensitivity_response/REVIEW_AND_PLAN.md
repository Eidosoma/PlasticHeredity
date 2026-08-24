# Review of the threshold-sensitivity comments and proposed response plan

**Date:** 18 August 2026  
**Status:** planning and feasibility review only  
**Isolation:** this folder is independent of the manuscript, sealed result bundles, and other parallel work. No source manuscript or result artifact has been modified.

## Executive recommendation

The objection is valid and predictable. Limitation 3 identifies the operational choices, but the paper does not currently demonstrate that the central findings survive nearby definitions. The existing episode-coherence sensitivity analysis is useful but insufficient: it varies pairwise-coherence and old-anchor cutoffs only among futures already positive under the fixed F12 definition. It therefore cannot answer sensitivity of event prevalence, empirical-probability reliability, predictor advantage, or intervention direction.

The best response is to add a compact, explicitly exploratory, no-refit appendix in v1 and retain a strengthened limitation. The appendix should do four things:

1. empirically contextualize the `H>0.90` inheritance and `H<=0.85` distinctness cutoffs;
2. rescore a fully disclosed union of the reviewer's nearby F12 definitions;
3. apply the already frozen predictors without refitting or choosing a favorable definition;
4. test the direction of any F12 intervention claim on the same endpoint grid.

The strict F32 endpoint should receive a smaller companion sensitivity analysis because its run length, coherence threshold, and old-anchor threshold are separate operational choices. It must remain separate from F12 and cannot be used to rescue the failed strict-event predictor.

If the appendix cannot be completed, add this exact sentence to Limitation 3:

> Robustness of the principal conclusions across nearby endpoint thresholds and horizons has not yet been established.

Merely acknowledging that the choices are operational is not an adequate substitute.

## What was reviewed

- `../PRE_PRINT_PAPER_DRAFT.md`, especially the endpoint definitions, Results 5-8, and Limitations 3, 12, and 13.
- The supplied 18-page Pigozzi-Levin PDF, `../Causal Architecture Dynamics Prior to Arrival of Self-replicators in a Model of Catalytic Networks Relevant to Origin-of-Life.pdf`. The source paper describes recurring compositions and a similarity threshold but does not supply an executable threshold contract that resolves the present choices.
- Existing F12, episode-geometry, F32, and intervention bundles and the code paths that generated them.

## Assessment of each commenter point

| Commenter concern | Current coverage | Remaining gap | Recommended response |
|---|---|---|---|
| Limitation 3 admits operational choices but reports no endpoint sweep | Correct in substance | The draft mentions exploratory geometry thresholds, but not an endpoint-definition sweep | Add the no-refit appendix and revise Limitation 3 |
| Show the parent-to-selected-daughter boundary-H distribution | Not shown | Readers cannot see whether `0.90` lies in a valley, shoulder, or dense part of the distribution | Add a candidate-separated histogram/ECDF and quantiles with threshold lines |
| Contextualize `H<=0.85` with a between-independent-lineage distribution | Not shown | The meaning of `0.85` in this compositional geometry is opaque | Add a precisely defined same-matrix, matched-phase independent-lineage reference distribution |
| Sweep inheritance H and F8/F10/F12/F16 horizons | Not done | F4/F8/F12 were explored during development, but this is not an unchanged-model sensitivity analysis on the final claim | Rescore the complete disclosed grid with frozen models |
| Vary renewal run length 2/3/4 | Only separate descriptive process outcomes exist at the baseline threshold/horizon | No joint sensitivity with threshold and horizon | Include run length in the F12 grid |
| Vary strict run length 7/8/9 and old-anchor cutoff 0.80/0.85/0.90 | Only the baseline prospective F32 endpoint and conditional geometry tables exist | Strict-event prevalence/reliability under nearby definitions is unknown | Add a separate F32 grid; do not mix it with F12 |
| Report prevalence | Baseline only | No surface across definitions | Report every grid cell, candidate, and branch half |
| Report branch-half reliability | Baseline only | Empirical q may become unreliable for rarer definitions | Report ordinary and within-matrix-centered split-half Spearman with matrix-bootstrap intervals |
| Report predictor advantage | Baseline only | A model may work only because its endpoint matches the discovery definition | Apply frozen predictions unchanged; report full-minus-history log-loss and rank contrasts |
| Report intervention direction | No intervention result is included at the manuscript's 15 August cutoff | If later causal results enter the paper, their endpoint robustness is untested | Either keep intervention results outside v1, or update the evidence cutoff and rescore the frozen CR1 arms on the same F12 grid |
| Do not retune and select a favorable combination | Not yet applicable | A poorly designed sweep could become a second adaptive search | Freeze the grid, metrics, plots, and interpretation rules before opening cell results; fit nothing |

## Important conceptual clarifications

### 1. The two endpoints must not be conflated

The F12 endpoint is:

> Within horizon F, observe a parent-to-selected-daughter break at `H<=tau`, followed later by R consecutive parent-to-selected-daughter inheritances at `H>tau`.

Its registered baseline is `(tau, F, R)=(0.90, 12, 3)`. It does not require the R daughters to occupy one mutually coherent neighborhood or be far from the old anchor.

The strict F32 endpoint additionally requires an episode-wide all-pairs coherence condition and separation of every episode daughter from the pre-break parent. Its registered baseline is horizon 32, run length 8, strict adjacent and all-pairs `H>0.90`, and inclusive old-anchor `H<=0.85`.

Sensitivity results for one endpoint do not validate the other.

### 2. An empirical valley cannot rewrite the adaptive history

If the boundary-H distribution is bimodal and has an antimode near `0.90`, the paper may say that the conventional choice *coincides post hoc with an empirical separation in these cohorts*. It should not say that the threshold *was data-motivated* unless the chronology shows that this distribution was used before the threshold was frozen.

### 3. The cosine-floor concern is valid, but the wording needs care

Nonnegative vectors can have cosine similarity near zero when their supports differ, so a high floor is not a mathematical consequence of positivity alone. In this implementation, compositions are also sparse count vectors (`N_g=100`, daughter mass near 40), rather than strictly positive dense vectors. Nevertheless, shared support, closure, mass constraints, and a common catalytic matrix can make empirical similarities high. The right answer is the requested empirical reference distribution and the percentile position of `0.85`, not a generic theoretical assertion about cosine geometry.

### 4. Exact replay is not the same as retaining full trajectories

The core F12 bundles retain branch labels and state predictions, but not every continuous boundary-H value or intermediate daughter composition. Their seeds and state contracts permit exact deterministic regeneration.

- Definitions with `F<=12` can be obtained by exact regeneration followed by rescoring.
- `F16` on an originally F12 branch requires four additional simulated fissions. It uses the same restored state and seed stream, with no new matrix, state, model, or tuning, but it is a deterministic horizon extension rather than pure rescoring of retained F12 records.
- Existing 32-fission cohorts provide a genuinely replayed alternative for horizon sensitivity, but applying the original frozen F12 predictor to those states is a new, explicitly exploratory external-cohort evaluation.

The appendix should use these terms accurately.

## Data and feasibility audit

| Analysis | Best local source | What is already retained | What must be regenerated |
|---|---|---|---|
| Main F12 prediction sensitivity | `../results/scaled5` (and optionally nested `../results/full`) | Frozen full/history predictions, state/matrix IDs, baseline labels, branch halves, exact seed contract | Continuous boundary H through the maximum selected horizon |
| Independent F12 replication | `../results/mechanistic_confirmation` and `../results/beta_complete_confirmation` | Disjoint states, branch labels, several legacy frozen predictions | Continuous boundary H if included in the endpoint grid |
| Existing conditional episode geometry | `../results/episode_coherence_audit` | 145,516 positive-episode measurements and the `0.90/0.95/0.975` by `0.90/0.85/0.80` grid | Nothing for the existing conditional audit; it does not replace endpoint rescoring |
| Strict F32 sensitivity | `../results/regime_confirmation` and/or `../results/regime_ensemble_confirmation` | Baseline labels, branch halves, predictions, first run-8 windows and their geometry, exact replay contracts | Full boundary/composition sequences for alternate inheritance thresholds or run lengths |
| F12 intervention direction | `../results_intervention_replication/cr1_model_guided_confirmation` | Full 12-boundary H arrays, common-random-stream arms, predictions, branch halves | Nothing for `F<=12`; an F16 suffix requires deterministic extension |
| Parent-daughter H reference | Same unmanipulated observational replay used for the F12 grid | Reconstructable from every fission record | One replay pass that saves boundary H |
| True between-independent-lineage H reference | Upstream L36-L37 artifacts named in the manuscript are not present in this compact folder | No directly usable lineage-composition table found here | Retrieve the immutable upstream artifact, or generate a separately labeled matched independent-lineage reference |

Do not use intervention arms to define the natural boundary-H distribution. If a local shortcut is needed for engineering checks, the CR1 `NOOP` arm contains boundary H, but the paper-facing reference should come from unmanipulated observational cohorts.

## Proposed frozen exploratory specification

### A. F12-family grid

Use the union of both reviewer suggestions so no favorable subset is chosen after seeing results:

- inheritance threshold `tau in {0.85, 0.88, 0.90, 0.92, 0.95}`;
- horizon `F in {8, 10, 12, 16}`;
- renewal run length `R in {2, 3, 4}`;
- strict inheritance `H>tau` and break `H<=tau` in every cell;
- registered baseline `(0.90, 12, 3)` visibly marked.

This is 60 fully reported definitions. For a compact figure, identify the local neighborhood `tau={0.88,0.90,0.92}` as the main visual panel and label `0.85` and `0.95` as stress tests, but retain all 60 cells in the numerical table and machine-readable file.

For every definition, candidate, and preassigned branch half, report:

1. branch-level prevalence with a whole-matrix bootstrap 95% interval;
2. number and fraction of states in the empirical transition region;
3. ordinary branch-half Spearman reliability;
4. within-matrix-centered branch-half Spearman reliability;
5. frozen full-versus-history log-loss gain for half A and half B;
6. frozen full-versus-history q-Brier gain and centered-rank difference as secondary diagnostics;
7. whole-matrix bootstrap intervals, with candidates and halves never pooled to rescue disagreement.

The full and history models, transforms, coefficients, and probabilities remain exactly as archived. Do not recompute threshold-dependent history features and do not recalibrate either model. This deliberately asks whether the baseline predictor carries information about nearby endpoint definitions. Because altered prevalence can create calibration mismatch, interpret raw log-loss together with the threshold-insensitive ranking diagnostics.

### B. Intervention-direction grid

If CR1 is brought inside the manuscript's evidence cutoff, rescore the same F12-family endpoint for each archived arm. At minimum report, separately by candidate and branch half:

- `MODEL_UP - MODEL_DOWN`;
- `MODEL_UP - NOOP`;
- `NOOP - MODEL_DOWN`;
- `RANDOM - NOOP`.

Preserve the common-random-stream pairing and whole-matrix inference. Plot estimates and intervals for every definition; summarize sign stability rather than selecting definitions that pass. The random/no-op equivalence margin remains the registered margin and is not changed per endpoint.

The current CR1 arrays directly support thresholds and run lengths through F12. F16 needs a registered deterministic extension. If that extension is not completed, label the intervention grid as `F8/F10/F12 only` and do not imply F16 robustness.

If CR1 remains outside the manuscript cutoff, say explicitly that intervention sensitivity is not applicable to the results reported in this version and retain the sentence that no Codex causal intervention is claimed. Do not use the unsuccessful paper-facing Phi intervention reconstruction as a substitute for F12 causal sensitivity.

### C. Strict F32 grid

Keep horizon 32 fixed and vary the three strict-event choices in a compact coupled grid:

- adjacent-inheritance and all-pairs-coherence threshold jointly `tau_strict in {0.88, 0.90, 0.92}`;
- strict run length `L in {7, 8, 9}`;
- inclusive maximum old-anchor similarity `delta in {0.80, 0.85, 0.90}`;
- registered baseline `(0.90, 8, 0.85)` visibly marked.

Report prevalence, ordinary and centered branch-half reliability, and unchanged predictor log-loss gains for the registered h10-plus-state comparison. If the later fixed ensemble is shown, report it as a separately registered failed predictor and do not let a favorable alternate endpoint relabel either baseline failure.

The coupled threshold grid is compact and directly tests the operational package. A secondary one-factor-at-a-time table may separate adjacent-inheritance from episode-wide coherence if space permits, but it must be declared before inspecting those cells.

No current intervention campaign establishes control of the strict F32 event. Do not include F12 intervention effects in the strict grid or imply strict-event control.

## Reference-distribution figure

Create one appendix figure with shared x-axis and candidate-specific facets.

### Panel A: parent-to-selected-daughter boundary H

- Source: all unmanipulated fission records from the chosen observational replay.
- Show a full-range ECDF plus a zoomed histogram or density over the upper tail.
- Draw vertical lines at `0.85`, `0.88`, `0.90`, `0.92`, and `0.95`.
- Print candidate-specific quantiles and the fraction on each side of every proposed threshold.
- Avoid treating millions of boundaries as independent. Intervals and any modality summary must resample whole catalytic matrices.
- If discussing bimodality, report a fixed modality diagnostic and a bootstrap interval for any estimated antimode. Do not infer a privileged cutoff from a visually tuned KDE bandwidth.

### Panel B: between-independent-lineage H

Before computing, freeze the reference-pair contract:

1. same catalytic matrix and simulator candidate;
2. genuinely independent lineages or seeds, not two observations from the same lineage;
3. matched generation/fission index and comparable mass/phase;
4. a fixed number of pairs per matrix so matrices with more possible pairs do not dominate;
5. whole-matrix bootstrap uncertainty;
6. no cross-candidate pooling.

Show the ECDF/distribution with `0.80`, `0.85`, and `0.90` marked. Report the empirical percentile of `0.85` and quantities such as `P(H<=0.85)` and `P(H>0.85)`. These numbers make "distinct" interpretable. If only same-origin branch futures are available, label the panel "between independent futures from a common restored state" rather than "between independent lineages."

## Analysis workflow

### Step 1: freeze the exploratory protocol

Write a short protocol before examining grid outcomes. Seal:

- source bundles and SHA-256 identities;
- endpoint inequalities and onset rules;
- the complete grids above;
- cohort priority and whether F16 uses extension or an existing F32 cohort;
- branch-half split;
- bootstrap/permutation units and seeds;
- models and saved prediction columns;
- plots, tables, and interpretation language;
- failure/stop conditions.

Classify the work as `post_hoc_exploratory_sensitivity`, not confirmation.

### Step 2: regenerate once, then rescore many times

Create a new immutable analysis bundle in this separate folder. Regenerate each selected future once to the maximum required horizon and retain, at minimum:

- state, cohort, candidate, matrix, landmark, branch, and half IDs;
- parent-to-daughter boundary H sequence;
- intermediate daughter compositions where strict geometry is needed;
- completion/extinction status;
- exact source seed and record digest.

All grid cells must be vectorized rescoring of this single retained replay table. Do not rerun the simulator separately for each threshold.

### Step 3: require baseline round-trip equivalence

Before opening alternate cells, reproduce the registered baseline exactly:

- every F12 branch label and process value;
- q estimates and branch-half allocations;
- baseline prevalence and reliability;
- archived log-loss gain to numerical tolerance;
- F32 baseline labels and continuous geometry;
- CR1 baseline arm effects if intervention sensitivity is included.

Any mismatch stops the analysis until resolved. Save an independent metric-recomputation audit.

### Step 4: compute the complete grids

Evaluate all declared cells, including unfavorable and undefined ones. Preserve reasons for undefined reliability (for example, no between-state variation at an extreme definition). Report event counts and event-positive matrices so apparently stable point estimates are not detached from effective support.

### Step 5: create compact figures and complete tables

Recommended appendix package:

- **Figure S1:** boundary-H and between-independent-lineage H reference distributions;
- **Figure S2:** F12 prevalence and centered branch-half reliability surfaces;
- **Figure S3:** frozen-model log-loss gain and, if in scope, CR1 intervention-direction surfaces;
- **Figure S4:** strict F32 prevalence/reliability/predictor sensitivity;
- **Table S1:** every F12 cell with estimates, intervals, counts, and undefined reasons;
- **Table S2:** every strict F32 cell;
- **CSV/JSON:** exact machine-readable grids, protocol, manifest, replay audit, and checksums.

Use the baseline marker consistently. Show candidates and branch halves rather than averaging them into a favorable headline.

### Step 6: interpret continuity, not winners

The appendix should answer four qualitative questions:

1. Does event prevalence change smoothly or collapse near the registered definition?
2. Does empirical q remain measurable with independent branch halves?
3. Does the unchanged predictor retain an advantage across the local neighborhood?
4. If causal results are in scope, do intervention contrasts keep their expected direction?

Do not promote the best alternate definition, refit models, or describe an isolated favorable cell as robustness. A sign reversal, near-zero reliability, or support collapse is a finding to report, not a reason to narrow the displayed grid.

### Step 7: revise the manuscript and response letter

Add:

- a Methods subsection titled **Exploratory endpoint-definition sensitivity**;
- a short Results subsection that states the whole pattern, including failures;
- explicit Appendix figure/table calls;
- a revised Limitation 3 noting that the sweep is post hoc and does not prospectively validate the cutoffs;
- an updated evidence cutoff if post-15-August intervention results are incorporated;
- a response-to-reviewer paragraph mapping each requested metric and value to its appendix location.

Suggested limitation after a completed appendix:

> The inheritance, horizon, run-length, coherence, and old-anchor choices remain operational rather than uniquely validated. A post-hoc, no-refit replay sensitivity across nearby definitions is reported in Appendix X; it tests local qualitative stability but does not convert any alternate definition into a confirmatory endpoint.

## Acceptance criteria

The response is complete only if all of the following are true:

- every reviewer-suggested value appears in the declared union or is explicitly justified as out of scope;
- the registered baseline is reproduced exactly before alternate results are used;
- no model, feature transform, threshold-dependent input, or calibration is refit;
- all cells are reported, not only favorable ones;
- candidates, branch halves, and materially different cohorts are not pooled to rescue a result;
- F16 is accurately labeled as an extension unless based on an already archived 16+-fission future;
- the `0.85` cutoff is placed in an empirical lineage-reference distribution;
- intervention direction is shown only for the endpoint actually intervened upon;
- the appendix is labeled exploratory throughout;
- the manuscript retains a direct limitation sentence even if the local pattern looks robust.

## Recommended order of work

1. Decide whether the manuscript evidence cutoff will remain 15 August or include the completed intervention program.
2. Retrieve or define the independent-lineage reference before opening distribution results.
3. Seal the exploratory protocol and source hashes.
4. Build and verify the maximum-horizon replay table.
5. Produce the F12 grid and reference figure first; these answer the strongest objection.
6. Add CR1 direction only if intervention evidence enters this manuscript version.
7. Produce the strict F32 companion grid.
8. Revise manuscript language only after the complete, nonselected grid is visible.

## Bottom line

The commenter is not asking for another confirmatory campaign. They are asking whether the reported phenomenon is a stable neighborhood in definition space or a point result at one operational coordinate. The repository is well positioned to answer that without new model fitting or new matrices. It is not entirely zero-compute table rescoring, because continuous trajectories were not retained in every core bundle and F16 exceeds the original F12 horizon, but deterministic replay, strict baseline round-trip checks, and a frozen complete grid can answer the objection cleanly.
