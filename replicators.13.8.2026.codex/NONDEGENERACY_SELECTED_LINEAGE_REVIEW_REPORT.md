# Non-degeneracy and selected-lineage review report

**Date:** 2026-08-18  
**Status:** Reviewer-prompted diagnostic analysis and revision recommendation only. No change to the preprint has been made.  
**Connected item:** [`COMPOSITE_HISTORY_CAPACITY_MATCHING_REPORT.md`](COMPOSITE_HISTORY_CAPACITY_MATCHING_REPORT.md)  
**Scope:** The strict coherent-eight endpoint and the long-running feedback experiments.

## Executive finding

The reviewer has identified a real and important interpretive gap. The retained data are favourable to the preprint, but they support a narrower and more informative claim than the current narrative.

1. The strongest degeneracy explanation is not supported. Strict coherent-eight episodes are generally not one- or two-type assemblies. Across all 9,703 positives in the two independent tests, the four local-candidate means are 11.86–12.54 occupied types and 5.36–5.73 effective species per daughter. No strict window has all eight daughters reduced to two occupied types.
2. Concentration is nevertheless strongly associated with the strict endpoint. Mean leading-species share is 0.561–0.583 in strict windows versus 0.338–0.346 in same-state matched inherited-eight non-events. The same leading species persists through all eight daughters in 98.7–99.1% of strict events, versus 20.0–23.8% of matched non-events.
3. The strict daughters remain compositionally dynamic. Mean normalized successive-daughter turnover is 0.193–0.199, first-to-last turnover is 0.384–0.394, and 31.7–33.3 distinct types appear somewhere across a typical eight-daughter event window.
4. The feedback result shows the same stability–diversity trade-off without general collapse. In independent test 1, inherited-boundary-raising control ends at about 10–11 occupied types and about 6 effective species at both fission 60 and fission 120, compared with about 17 occupied types and 11–12 effective species under matched no-op. No endpoint is a one-type assembly; only 0.35–0.69% of learned-controller endpoints have at most two occupied types.
5. Independent test 2 already reproduces the fission-60 direction: the learned stabilizer has Shannon entropy 1.585–1.589, 10.35–10.44 occupied types, and leading-species share 0.495–0.500, versus entropy 2.243–2.256, 16.61–16.90 occupied types, and leading share 0.337–0.338 under no-op. It also retained survival and approximately 0.995 inheritance over fissions 61–120, but its retained summary does not include diversity at fission 120.
6. Growth and fission do not stop. Strict positives almost always complete F32, and every retained feedback lineage completes fission 60 and fission 120. Growth-update counts remain nonzero, although their direction relative to controls is implementation-dependent and their units differ across simulator contracts.
7. An exact immediate complementary-daughter sensitivity is highly positive: at all eight divisions, the unselected complementary daughter also has parent similarity above 0.9 in 98.3–99.2% of strict events across the four local candidates. This is stronger than selected-daughter-only partition fidelity, but it is not a full branching-lineage experiment because the unselected daughters were not propagated.
8. The reviewer is correct about terminology. The abstract and limitations should say **selected-lineage parent-to-daughter inheritance**. The methods already explain that one daughter continues the lineage, but the present abstract can be read as a claim about both daughters or reproductive fidelity in a population.

The appropriate interpretation is therefore:

> Strict coherent-eight episodes are compositionally multitype and continue to turn over, but they are markedly concentration-associated and are usually organized around one persistent leading species. Active feedback likewise maintains selected-lineage inheritance while narrowing diversity, without generally collapsing the assembly to one or two occupied types.

This strengthens credibility because it addresses the geometric concern directly. It also changes the mechanistic emphasis: concentration is not merely a nuisance alternative to dismiss; it is a plausible part of how high cosine continuity is achieved.

## Reviewer comment, verbatim

> Add a non-degeneracy characterization
>
> This is the most important substantive analysis that appears to be missing from the narrative.
>
> Cosine similarity above 0.9 can become easier to maintain when compositions become highly concentrated or dominated by one molecule type. The feedback experiments already show that stabilized lineages end with fewer occupied types and a larger leading-species share. The discussion appropriately identifies a possible stability–diversity trade-off.
>
> A skeptical reviewer can therefore argue that “near-continuous heredity” is partly achieved by collapsing the assembly into a low-diversity, easily copied composition.
>
> For both the strict coherent-eight events and long-running stabilized feedback trajectories, report:
>
> effective species number or Shannon diversity;  
> occupied-type count;  
> largest-species fraction;  
> compositional turnover;  
> growth updates and fission continuity;  
> comparison against matched non-event or no-op trajectories;  
> the fraction of events dominated by one or two types.
>
> For the strict event in particular, show whether the eight mutually coherent daughters are compositionally nontrivial, rather than merely mutually similar because one species dominates. This does not necessarily weaken the result—concentration may be the actual mechanism—but it materially changes its interpretation.
>
> The endpoint also follows one selected daughter per fission, not both daughters. The paper states this clearly in the methods, but the distinction should appear earlier because some reviewers will regard it as selected-lineage continuity rather than full reproductive fidelity. A both-daughters sensitivity analysis would be valuable, but at minimum the abstract and limitations should say “selected-lineage parent-to-daughter inheritance.”

## 1. Audit of the current preprint

### 1.1 What is already clear

The manuscript is not silent about daughter selection in the technical sections:

- the Introduction says that one selected daughter continues the lineage;
- the reader's guide defines inheritance as cosine similarity between the pre-fission parent and selected daughter;
- the endpoint methods state that the eight daughters occur at successive fissions of one continuing lineage and are not siblings; and
- the figure captions repeatedly use “parent-to-selected-daughter.”

Relevant source locations in [`../PRE_PRINT_PAPER_DRAFT.md`](../PRE_PRINT_PAPER_DRAFT.md):

- Abstract: line 12;
- In brief: line 20;
- Introduction: line 24;
- reader's guide: line 42;
- strict endpoint methods: lines 188–199;
- strict results: lines 374–403;
- feedback results: lines 520–530;
- synthesis: lines 609–619 and 641–647; and
- limitations: lines 671–680.

The typeset PDF was also inspected directly. The relevant material appears on pages 1, 19–20, 30, and 41 of [`../output/pdf/plastic-heredity-biorxiv-v1.pdf`](../output/pdf/plastic-heredity-biorxiv-v1.pdf).

### 1.2 What remains missing

The reviewer is right on four points.

First, the Abstract and In brief use unqualified “parent-to-daughter” language. A reader does not learn until the Introduction or methods that only one contract-selected daughter continues.

Second, the strict-event results establish pairwise cosine coherence and old-anchor distinctness but do not report diversity, dominance, or turnover within the eight daughters. Figure 5 is a similarity-geometry figure, not a non-degeneracy figure.

Third, the feedback narrative says that stabilized lineages end with fewer occupied types and a larger leading-species share, but gives no numerical diversity or dominance characterization in the main text. It therefore acknowledges the trade-off without bounding its magnitude.

Fourth, neither the strict-event nor feedback narrative reports an explicit “one- or two-type dominance” frequency. Without a stated threshold, “dominated” is also underspecified.

### 1.3 Existing limitation language is insufficient

The current strict-event limitation says that prediction, recurrence, intervention response, and basin behaviour remain open. The feedback limitation discusses external selection, exploratory compression, policy-ranking differences, and the missing shared restoring basin. Neither says:

- all endpoints follow one selected daughter;
- unselected descendant lineages were not propagated;
- high cosine continuity may be facilitated by composition concentration; or
- the strict event is strongly associated with a persistent leading species.

Those qualifications belong in the main limitations, not only in an appendix.

## 2. Reviewer-prompted diagnostic protocol used for this report

These calculations were performed to evaluate the comment. They are not yet a sealed, publication-ready result bundle and should not be copied into the manuscript until a permanent script, readback audit, and checksums are created.

### 2.1 Composition metrics

For composition counts \(n_i\), relative abundance is

\[
p_i = \frac{n_i}{\sum_j n_j}.
\]

The diagnostic uses:

- **Shannon diversity:** \(H_S=-\sum_{i:p_i>0}p_i\log p_i\), in nats;
- **effective species number:** \(N_{\mathrm{eff}}=\exp(H_S)\);
- **occupied types:** \(N_{\mathrm{occ}}=\sum_i 1[p_i>0]\);
- **largest-species fraction:** \(p_{(1)}=\max_i p_i\);
- **top-two share:** \(p_{(1)}+p_{(2)}\); and
- **normalized turnover:** \(T(p,q)=\tfrac12\sum_i|p_i-q_i|\), ranging from 0 to 1.

For a strict event, daughter-level metrics are averaged across the eight daughters. Turnover is reported both as the mean across the seven successive daughter pairs and between the first and eighth daughters.

### 2.2 Strict positives

Independent test 1 retained labels and exact seed-addressable state compositions in:

- [`results/regime_confirmation/confirmation_arrays.npz`](results/regime_confirmation/confirmation_arrays.npz); and
- [`plastic_heredity/regime_confirmation.py`](plastic_heredity/regime_confirmation.py).

All 5,041 labelled positives were replayed from the original state, matrix, candidate, landmark, and branch seed. The endpoint and first qualifying onset were checked against the retained arrays before metrics were calculated.

Independent test 2 retained all 4,662 positive strict windows, including their eight daughter compositions, in:

- [`../replicators.13.8.2026.fable/replication/results_strict8_occurrence/strict8_units.pkl`](../replicators.13.8.2026.fable/replication/results_strict8_occurrence/strict8_units.pkl).

The archived daughters supplied the composition calculations. Exact seed replay supplied growth updates, parent compositions, complementary daughters, and continuity checks.

Together these cover the same 9,703 positives reported in the manuscript.

### 2.3 Matched strict non-events

The comparison was designed to avoid contrasting strict positives with arbitrary futures that never had an opportunity to form an inherited eight-run.

For each positive, the diagnostic searched within the same:

- independent implementation;
- local candidate;
- catalytic matrix;
- restored state and landmark; and
- preassigned branch half.

Eligible controls had:

- a completed F32 future;
- a first inheritance break;
- a later run of eight consecutive inherited selected-daughter boundaries; and
- no qualifying strict event.

Within that set, an unused negative branch was selected by minimum distance in first-break timing and inherited-eight-window timing, with deterministic branch-index tie breaking. The positive uses its first qualifying strict window; the negative uses its first eligible inherited-eight window.

This produced:

| Independent test | Candidate | Strict positives | Matched non-events | Unmatched positives |
|---|---|---:|---:|---:|
| 1 | 02 | 2,354 | 2,330 | 24 |
| 1 | 03 | 2,687 | 2,672 | 15 |
| 2 | 02 | 2,177 | 2,149 | 28 |
| 2 | 03 | 2,485 | 2,472 | 13 |

The matching is a reviewer-prompted descriptive control. A final protocol should freeze it before a permanent rerun and should use catalytic matrix as the uncertainty unit.

### 2.4 Dominance characterizations

Because the reviewer does not define “dominated,” the diagnostic avoids a single post hoc threshold and reports several transparent properties:

- structural collapse: any or all daughters have \(N_{\mathrm{occ}}\leq2\);
- sustained one-type majority: all eight daughters have \(p_{(1)}\geq0.5\);
- average one-type majority: the eight-daughter mean of \(p_{(1)}\) is at least 0.5;
- strong two-type concentration: all eight daughters have \(p_{(1)}+p_{(2)}\geq0.8\); and
- dominant-identity persistence: the same type is largest in all eight daughters.

The final analysis should preregister or freeze these definitions before generating the publication table. Continuous distributions should remain primary.

### 2.5 Immediate both-daughters sensitivity

All relevant fissions use complementary partitioning. If \(P\) is the pre-fission parent and \(D\) is the selected daughter, the unselected daughter is exactly

\[
D_{other}=P-D.
\]

The diagnostic asks whether \(H(P,D_{other})>0.9\) at all eight divisions in the selected-lineage strict window.

This is an exact rescore of already-observed partitions. It does not generate new futures. It tests immediate two-daughter partition fidelity, not the later histories of both daughters.

## 3. Strict coherent-eight results

### 3.1 Composition and turnover

| Test/candidate | Event/control \(n\) | Effective species, event/control | Occupied types, event/control | Top-1 share, event/control | Successive turnover, event/control | First-to-eighth turnover, event/control |
|---|---:|---:|---:|---:|---:|---:|
| Independent test 1 / 02 | 2,354 / 2,330 | 5.360 / 11.174 | 11.86 / 17.26 | 0.583 / 0.338 | 0.193 / 0.309 | 0.384 / 0.661 |
| Independent test 1 / 03 | 2,687 / 2,672 | 5.494 / 10.939 | 12.07 / 17.01 | 0.566 / 0.340 | 0.196 / 0.309 | 0.385 / 0.655 |
| Independent test 2 / 02 | 2,177 / 2,149 | 5.728 / 11.017 | 12.54 / 17.25 | 0.561 / 0.346 | 0.195 / 0.301 | 0.394 / 0.658 |
| Independent test 2 / 03 | 2,485 / 2,472 | 5.639 / 11.141 | 12.47 / 17.41 | 0.565 / 0.340 | 0.199 / 0.309 | 0.390 / 0.653 |

The contrast is consistent across all four local candidates:

- strict windows retain about 12 occupied types per daughter;
- their effective diversity is about half that of matched inherited-eight non-events;
- the leading type holds about 56–58% of the composition rather than about 34%;
- successive turnover is lower but clearly nonzero; and
- the first and eighth daughters still differ by about 0.39 on the normalized L1 scale.

Across an event window, 31.7–33.3 distinct types appear at least once, compared with 43.6–44.8 in controls. Thus the events contain a persistent core plus turnover in lower-abundance types. They are neither static eightfold copies nor generally one- or two-type objects.

### 3.2 Concentration and dominance

| Test/candidate | Same leading type in all eight | Any daughter has at most 2 occupied types | All eight have at most 2 occupied types | Mean top-1 share at least 0.5 | All eight top-1 shares at least 0.5 | All eight top-2 shares at least 0.8 |
|---|---:|---:|---:|---:|---:|---:|
| Independent test 1 / 02 | 99.11% | 3.40% | 0% | 81.69% | 22.60% | 1.06% |
| Independent test 1 / 03 | 98.70% | 2.08% | 0% | 76.52% | 17.64% | 0.30% |
| Independent test 2 / 02 | 98.90% | 0.69% | 0% | 74.78% | 16.90% | 0.18% |
| Independent test 2 / 03 | 98.79% | 1.17% | 0% | 73.76% | 17.42% | 0.28% |

This is the central interpretive result.

- **No general structural collapse:** no event consists of eight daughters with two or fewer occupied types.
- **Substantial one-type concentration:** about three quarters to four fifths of events have an average leading share above 0.5.
- **Persistent dominant identity:** approximately 99% retain the same leading type through all eight daughters.
- **Not generally a two-type object:** only 0.18–1.06% have top-two share at least 0.8 in every daughter.

Matched inherited-eight non-events retain the same leading type in only 20.0–23.8% of windows. The strict event is therefore not merely an arbitrary multitype sequence; its coherence is usually organized around one persistent dominant species with a changing lower-abundance tail.

### 3.3 Growth and fission continuity

| Test/candidate | Mean growth updates per event fission | Matched-control mean | Positive F32 completion |
|---|---:|---:|---:|
| Independent test 1 / 02 | 29.85 | 40.34 | 100% |
| Independent test 1 / 03 | 23.89 | 31.88 | 100% |
| Independent test 2 / 02 | 111.23 | 64.79 | 100% |
| Independent test 2 / 03 | 29.16 | 16.33 | 99.92% |

The assemblies continue to grow and fission. Growth-update direction does not reproduce across implementations: strict events require fewer registered updates than controls in independent test 1 and more in independent test 2. This is not surprising because the update kernels differ, and independent-test-2 candidate 02 counts single-molecule categorical events whereas its candidate 03 counts vector-Poisson steps. The manuscript should report the values by contract and avoid a shared directional growth claim.

Two independent-test-2 positives certified the strict event before later extinction, explaining the 99.92% F32 completion in that cell. Certification-before-extinction is part of its frozen endpoint contract.

### 3.4 Immediate complementary-daughter sensitivity

| Test/candidate | Strict events where both immediate daughters pass \(H>0.9\) at all eight divisions | Matched inherited-eight non-events |
|---|---:|---:|
| Independent test 1 / 02 | 99.15% | 73.48% |
| Independent test 1 / 03 | 98.51% | 63.17% |
| Independent test 2 / 02 | 98.94% | 72.41% |
| Independent test 2 / 03 | 98.31% | 65.45% |

This diagnostic substantially reduces the concern that the strict window exists only because a favourable daughter happened to be selected at each partition. In nearly every strict event, the complementary daughter also resembles the parent at every one of the eight observed divisions.

It does **not** establish full reproductive fidelity. The complementary daughters were not grown and divided forward, so the analysis does not show that both descendant lineages would maintain the strict episode. The manuscript must preserve that boundary.

## 4. Long-running feedback results

### 4.1 Independent test 1: exact matched no-op endpoint comparison

The dense-feedback bundle retains 288 lineages per candidate and arm at fission 60, plus 288 per candidate and retained arm through an additional 60 active fissions:

- [`results_intervention_replication/cr7_closed_loop_steering/lineages.csv.gz`](results_intervention_replication/cr7_closed_loop_steering/lineages.csv.gz);
- [`results_intervention_replication/cr7_closed_loop_steering/lineage_arrays.npz`](results_intervention_replication/cr7_closed_loop_steering/lineage_arrays.npz); and
- [`results_intervention_replication/cr7_closed_loop_steering/conditional_active_extension/lineages.csv.gz`](results_intervention_replication/cr7_closed_loop_steering/conditional_active_extension/lineages.csv.gz).

The following comparison uses the learned inherited-boundary-raising controller and its matched no-op arm.

| Candidate/horizon | Inherited fraction, control/no-op | Shannon H, control/no-op | Effective species, control/no-op | Occupied types, control/no-op | Top-1 share, control/no-op | Mean growth updates, control/no-op | Completion |
|---|---:|---:|---:|---:|---:|---:|---:|
| 02 / fission 60 | 0.9926 / 0.9042 | 1.643 / 2.297 | 6.08 / 11.77 | 10.67 / 17.15 | 0.490 / 0.328 | 27.22 / 40.36 | 100% / 100% |
| 03 / fission 60 | 0.9901 / 0.8943 | 1.614 / 2.271 | 5.96 / 11.33 | 10.46 / 16.74 | 0.501 / 0.328 | 21.57 / 32.58 | 100% / 100% |
| 02 / fission 120 | 0.9942 / 0.9103 | 1.654 / 2.279 | 6.09 / 11.46 | 10.75 / 16.66 | 0.491 / 0.321 | 26.88 / 39.90 | 100% / 100% |
| 03 / fission 120 | 0.9912 / 0.9020 | 1.625 / 2.287 | 5.97 / 11.55 | 10.39 / 16.69 | 0.488 / 0.318 | 21.77 / 32.27 | 100% / 100% |

For fission 120, inherited fraction and mean growth updates refer to the additional fissions 61–120; composition metrics are the fission-120 endpoint.

The compact outgoing-influence rule gives the same qualitative result. Across candidates and horizons it ends at 10.68–11.18 occupied types, 5.86–6.23 effective species, and top-1 share 0.510–0.516 while retaining inherited fractions of 0.987–0.993.

### 4.2 Endpoint dominance frequencies

For the learned controller in independent test 1:

| Candidate/horizon | One occupied type | At most two occupied types | Top-1 share at least 0.5 | Top-1 share at least 0.9 |
|---|---:|---:|---:|---:|
| 02 / fission 60 | 0% | 0.69% | 48.26% | 2.08% |
| 03 / fission 60 | 0% | 0.35% | 48.96% | 0.69% |
| 02 / fission 120 | 0% | 0.69% | 48.26% | 1.74% |
| 03 / fission 120 | 0% | 0.35% | 48.26% | 1.04% |

The learned controller therefore produces a broad shift toward one leading species, not wholesale reduction to one or two occupied species. About half the endpoints have a majority species, while almost all retain more than two occupied types.

### 4.3 Independent test 2 reproduces the fission-60 trade-off

Independent test 2's retained D2/D3 audit reports final-ten-fission means:

- [`../replicators.13.8.2026.fable/replication/results_d2d3/d2d3_results.json`](../replicators.13.8.2026.fable/replication/results_d2d3/d2d3_results.json).

| Candidate | Shannon H, control/no-op | Occupied types, control/no-op | Top-1 share, control/no-op | Growth updates, control/no-op | Extinctions |
|---|---:|---:|---:|---:|---:|
| 02 | 1.589 / 2.243 | 10.35 / 16.61 | 0.495 / 0.338 | 106.97 / 78.88 | 0 / 0 |
| 03 | 1.585 / 2.256 | 10.44 / 16.90 | 0.500 / 0.337 | 26.96 / 19.47 | 0 / 0 |

Its selected-lineage inheritance over fissions 61–120 is 0.9958 and 0.9948 under the learned stabilizer, and every extension lineage survives. This independently reproduces concentration with continued growth and fission at fission 60. The retained extension summary does not preserve fission-120 entropy, occupied count, dominance, or turnover; those require exact replay before a shared fission-120 diversity statement is made.

### 4.4 Feedback turnover remains the main uncompleted item

Independent test 1 retains boundary \(H\), growth-update sequences, final compositions, and initial compositions, but not the full composition after every fission. Independent test 2 can regenerate the same histories from its frozen seeds but likewise did not retain a publication table of per-boundary turnover.

Initial-to-endpoint turnover is not an adequate substitute: it mixes 60 or 120 transitions and cannot distinguish a stable concentrated trajectory from repeated compositional replacement. The reviewer asked for trajectory turnover, so the proper measure is the normalized L1 distance between consecutive post-fission selected-daughter compositions, reported through time and against the matched no-op lineage.

This is an exact replay task, not a new-futures campaign.

## 5. Scientific interpretation

### 5.1 What the results rule out

The strict event cannot generally be dismissed as eight copies of a one- or two-type assembly:

- every local candidate averages about 12 occupied types per daughter;
- zero events keep all eight daughters at two or fewer occupied types;
- about 32–33 types appear somewhere across an event; and
- first-to-eighth turnover is about 0.39.

The feedback result is also not usually a terminal one- or two-type collapse. Controlled lineages keep approximately 10–11 types, continue growing, and complete every retained long horizon.

### 5.2 What the results do not rule out

Concentration is not a minor side effect. Strict events have approximately half the effective diversity of matched inherited-eight non-events and almost always keep one type in the leading position. Feedback drives the same direction relative to no-op.

The most defensible mechanistic reading is that high-fidelity cosine continuity is facilitated by a persistent dominant coordinate while a multitype lower-abundance tail continues to turn over. That is compositionally nontrivial, but it is not diversity-neutral.

### 5.3 Why this is positive for the preprint

The underlying occurrence and feedback results survive. The new characterization:

- answers a predictable geometric objection;
- shows that the assemblies are still multitype and dynamically changing;
- identifies a reproducible concentration signature across independent implementations;
- gives the stability–diversity discussion quantitative substance; and
- sharply distinguishes selected-lineage continuity from population reproduction.

The cost is a narrower headline. “Near-continuous heredity” should not stand alone; it should be “near-continuous selected-lineage parent-to-daughter inheritance under concentration-associated active control.”

## 6. What should be added to the preprint, and where

No edits have been made. The following is the recommended insertion plan after a permanent analysis bundle reproduces the diagnostics.

### 6.1 Abstract — required

At the first definition of plastic heredity, replace unqualified parent-to-daughter wording with:

> The alternative approach revealed plastic heredity: state-dependent probability that selected-lineage parent-to-daughter compositional inheritance will break and later renew.

Add one compact strict-event qualifier:

> Coherent eight-fission episodes occurred in 1.70–2.11% of futures; their daughters remained multitype and compositionally dynamic but were markedly more concentrated than matched inherited-eight non-events.

Qualify feedback:

> Active feedback maintained near-continuous selected-lineage inheritance while reducing diversity and increasing compositional concentration.

The Abstract should not imply full two-daughter reproductive fidelity.

### 6.2 In brief — required

Change “parent-to-daughter resemblance” to “parent-to-selected-daughter resemblance” or “selected-lineage parent-to-daughter resemblance.” Add one sentence that the strict event and feedback are concentration-associated rather than one- or two-type collapses.

### 6.3 Methods, after Strict coherent eight-fission endpoint — required

Add a subsection titled **Non-degeneracy and daughter-scope characterization** containing:

1. the exact definitions of Shannon H, effective species, occupied types, top-1 and top-2 share, and normalized L1 turnover;
2. the event window as the first qualifying strict window;
3. the matched inherited-eight non-event rule;
4. the dominance thresholds and a statement that continuous distributions are primary;
5. growth-update units by simulator contract;
6. whole-matrix cluster bootstrap inference;
7. the exact complementary-daughter calculation \(P-D\); and
8. the distinction between immediate two-daughter fidelity and propagating both descendant lineages.

### 6.4 Strict-event Results, immediately after the occurrence table — required

Add a short subsection titled **Strict episodes were multitype but concentration-associated**.

It should report, by independent test and candidate:

- effective species;
- occupied types;
- top-1 and top-2 shares;
- successive and first-to-eighth turnover;
- growth updates and F32 completion;
- one- and two-type dominance fractions;
- dominant-identity persistence;
- matched non-event contrasts; and
- immediate complementary-daughter sensitivity.

A compact table is sufficient. A supplementary ECDF figure would make the event/control separation in effective diversity, top-1 share, and turnover easy to audit.

### 6.5 Feedback Results, after the current trade-off sentence — required

Replace the qualitative sentence with a quantitative paragraph or table covering:

- learned stabilizer, compact rule, and matched no-op;
- fissions 60 and 120;
- both candidates and both independent tests where retained;
- Shannon or effective diversity, occupied types, top-1/top-2 share;
- per-boundary turnover trajectories;
- growth updates and horizon completion; and
- structural-collapse and dominance frequencies.

The text should separate shared results from single-implementation fission-120 diversity results.

### 6.6 Discussion — required

Expand the stability–diversity discussion to say:

> High inherited-boundary frequency was not generally achieved by collapse to one or two occupied types. It was nevertheless consistently associated with lower effective diversity and a larger, usually persistent leading species. Concentration may therefore be part of the mechanism by which cosine-defined continuity is maintained, while lower-abundance types continue to turn over.

For the strict event, add that one persistent dominant type appears to anchor most coherent windows. Avoid calling the event diversity-neutral or treating concentration only as a limitation.

### 6.7 Limitations — required

Add a limitation substantially equivalent to:

> **Selected-lineage and concentration scope.** Every lineage endpoint follows one contract-selected daughter after fission. It therefore measures selected-lineage parent-to-daughter continuity, not population-level or fully branching reproductive fidelity. The immediate complementary-daughter rescore assesses both products of the observed partitions but does not propagate unselected descendants. Strict events and actively stabilized trajectories are also concentration-associated: they remain multitype and compositionally dynamic, but reduced effective diversity and a persistent leading species make high cosine continuity easier to sustain.

### 6.8 Conclusion — recommended

Where the conclusion says that dense feedback maintained near-continuous inheritance, add “selected-lineage” and mention the concentration cost. Where it describes strict episodes, add “multitype but concentration-associated.”

### 6.9 Appendix B — recommended

Add a row to the endpoint-comparison table:

| Property | F12 break-and-renewal | Strict F32 eight-fission episode |
|---|---|---|
| Lineage topology | One selected daughter continues after each fission | One selected daughter continues after each fission; immediate complementary daughters can be rescored, but their descendant lineages are not propagated |

## 7. Required publication-grade analysis bundle

Before inserting numerical results, create a reviewer-response bundle under a new directory such as `reviewer_nondegeneracy_response/` with:

- `PROTOCOL.md` fixing metrics, thresholds, matching, inference, and language before the final rerun;
- one deterministic strict replay/rescore script for each implementation;
- one deterministic feedback replay/rescore script for each implementation;
- per-window and per-lineage CSV or Parquet outputs;
- `summary.json` with all four strict cells and every feedback arm/horizon;
- whole-matrix bootstrap intervals;
- endpoint and seed-replay audits;
- source and input hashes;
- `SHA256SUMS`; and
- a concise scientific report.

The bundle should explicitly distinguish:

- **existing-future rescore:** strict compositions, complementary daughters, and retained feedback endpoints;
- **exact same-seed replay:** missing parents, growth updates, and per-fission feedback compositions; and
- **new experiment:** propagation of both daughters into a branching descendant tree.

The first two require no new stochastic futures. The third would require a new endpoint and random-stream contract and should not be presented as a simple sensitivity rescore.

## 8. Recommended inferential conventions

1. Use catalytic matrix, not daughter, event, boundary, or lineage, as the resampling unit.
2. Stratify all summaries by independent implementation and local candidate before discussing shared ranges.
3. Treat the analysis as reviewer-prompted post hoc characterization.
4. Prefer intervals and full distributions over a large family of p-values.
5. Keep the matched inherited-eight comparison separate from an optional H-matched sensitivity; matching on H can overcontrol a pathway through which concentration affects the endpoint.
6. For feedback, use matched matrix/state/replicate no-op lineages and show trajectories rather than endpoint-only bars.
7. Do not pool growth-update counts across contracts with different update semantics.
8. Do not call immediate complementary-daughter fidelity “full reproductive fidelity.”

## 9. Bottom-line recommendation

This item should be addressed before submission. It is more than wording, but the retained evidence suggests that addressing it will improve rather than undermine the paper.

The manuscript should make three claims together:

1. strict coherent-eight selected-lineage episodes are real and independently reproduced;
2. they remain multitype and compositionally dynamic; and
3. they are strongly concentration-associated, usually around one persistent leading species.

For feedback, the corresponding claim is that external control maintains near-continuous selected-lineage inheritance while narrowing—but generally not eliminating—compositional diversity.

The minimal wording change is “selected-lineage parent-to-daughter inheritance” in the Abstract and limitations. The scientifically complete response is the permanent non-degeneracy rescore, matched controls, turnover trajectories, and immediate complementary-daughter sensitivity described above.
