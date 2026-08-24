# Reviewer note: state acquisition, survival, and landmark observability

**Date:** 2026-08-19  
**Status:** Evidence audit and analysis recommendation; no manuscript or canonical result files changed.

## Reviewer comment

> **State acquisition is conditioned on survival and available landmarks**
>
> The three implementations do not sample arbitrary GARD states in exactly the same way. The originating workflow requires complete 100-fission lineages; independent test 1 uses deterministic retries to construct complete landmark cohorts; independent test 2 uses no retry and omits unavailable later landmarks.
>
> This does not invalidate the results, but the estimand is closer to:
>
> Risk among surviving, observable post-fission states from lineages able to reach the registered landmarks.
>
> It is not necessarily risk among all assemblies generated from the initial distribution.
>
> Add this explicitly to the limitations and report:
>
> - completion/extinction rates by matrix and candidate;
> - the fraction of initial lineages contributing each landmark;
> - whether predictor performance or F12 prevalence changes at earlier versus later landmarks;
> - a sensitivity analysis restricted to a common acquisition rule across implementations, if practical.
>
> This will head off a survivorship-bias criticism.

## Bottom line

The comment is valid, but most of the requested response can be produced from retained data without generating new branch futures.

The most reassuring fact is that **all matched headline confirmation lineages reached all five registered landmarks in all three implementations**. Each implementation contributed 40 matrices x 2 candidates = 80 candidate lineages and 400 landmark states. Across O, T1, and T2, all 240 candidate lineages in the matched confirmation comparison reached generations 20, 35, 50, 65, and 80. Consequently, observed confirmation-cohort attrition cannot explain the headline differences among implementations.

Acquisition conditioning nevertheless remains part of the design and should be stated explicitly. It matters most for the development cohorts and for interpreting the target population: the analysis estimates risk among observable post-fission states under each implementation's acquisition contract, not necessarily among every assembly initially generated.

## What the manuscript already says

Appendix F already documents the implementation contracts:

- T1 used deterministic, domain-separated retries to construct complete 100-fission landmark cohorts. Restored futures were not retried.
- T2 did not retry principal lineages or restored futures and omitted unavailable later landmarks.
- The originating workflow retained principal cohorts requiring complete 100-fission lineages.

Source: `../../PRE_PRINT_PAPER_DRAFT.md`, especially lines 955-957.

What is currently missing is:

1. an explicit statement of the conditional estimand;
2. completion/extinction and landmark-contribution tables;
3. landmark-stratified predictor results;
4. a common-acquisition sensitivity or a precise statement of what can and cannot be harmonized from retained artifacts.

## Data availability by implementation

| Implementation | Completion and landmark availability | Landmark-stratified predictor data | Main limitation |
|---|---|---|---|
| Originating workflow (O) | Complete counts are available for retained development/validation and confirmation cohorts. L54 records 80 primary trajectories and 400 landmark states with no replacement. | L50 reports F12 prevalence by landmark. The checked-in L54 report does not include predictions and outcomes by landmark. | The denominator of all upstream lineages considered before complete-lineage selection is not retained locally. L54 landmark-specific predictor scores cannot be reconstructed from the checked-in report alone. |
| Clean-room test 1 (T1) | Accepted trajectories and landmarks are retained. Deterministic replay of the sealed main-path seeds reveals how often retries were actually used. | State identifiers, targets, direct-history predictions, and composite predictions are retained and can be rescored by landmark. | Canonical outputs do not record the failed-attempt index, so retry rates require deterministic replay. Discarded failed attempts do not have branch futures. |
| Clean-room test 2 (T2) | Main-path death, completed-fission count, and omitted landmarks are directly recorded or deterministically replayable. | Outcomes and direct/composite predictions are retained and can be rescored by landmark. | Existing development summaries undercount deaths occurring before a trajectory produces any training row; replay-derived counts should be used. |

## Completion and landmark audit

### Originating workflow

- L54 confirmation records 40 new shared matrices, 80 primary candidate trajectories, and 400 preregistered landmark states.
- Its validation report states exact replay and **no replacement**. Therefore, within the realized L54 confirmation cohort, 80/80 trajectories completed and every candidate/landmark cell contains 40/40 initial lineages.
- L50 development/validation selection retained 80 matrices, two candidates, and five landmarks, yielding 800 states. These cohorts are complete by construction.
- The checked-in artifacts do not preserve the number of upstream lineages that failed before complete-lineage eligibility. The unconditional completion rate from the original initial distribution therefore cannot be reported honestly without recovering the original trajectory manifests or rerunning the upstream acquisition.

Relevant retained reports:

- `../../original.1.8.2026.eidosoma-ai-scientist.stepReports/cache/e01_s19_l54/build/S19_L54_FULL_RESULTS.md`
- `../../original.1.8.2026.eidosoma-ai-scientist.stepReports/cache/e01_s19_l50/build/S19_L50_FULL_RESULTS.md`

### Clean-room test 1

The cohort builder permits as many as 100 deterministic attempts per matrix/candidate until a complete trajectory is obtained. A read-only deterministic replay of the sealed main-path seeds found:

| Cohort | Candidate | Accepted lineages | Accepted on first attempt | Required a retry |
|---|---:|---:|---:|---:|
| Headline development | c02 | 40 | 40 | 0 |
| Headline development | c03 | 40 | 40 | 0 |
| Headline confirmation | c02 | 40 | 40 | 0 |
| Headline confirmation | c03 | 40 | 40 | 0 |
| Scaled development | c02 | 200 | 200 | 0 |
| Scaled development | c03 | 200 | 200 | 0 |
| Scaled confirmation | c02 | 200 | 199 | 1 |
| Scaled confirmation | c03 | 200 | 200 | 0 |

Thus the retry mechanism was dormant in the matched headline analysis. At scaled confirmation it affected only 1/200 c02 lineages, or 1/400 candidate lineages in that confirmation cohort.

All accepted T1 trajectories are complete by construction and contribute all five landmarks. The implementation is in `../../replicators.13.8.2026.codex/plastic_heredity/experiment.py`, lines 101-152.

### Clean-room test 2

T2 uses no retry and omits a landmark when the main trajectory terminates before it. The implementation is in `../../replicators.13.8.2026.fable/replication/cohort.py`, lines 62-84 and 96-129.

Headline cohorts:

| Cohort | Candidate | Complete/reached registered landmarks | Fraction |
|---|---:|---:|---:|
| Development | c02 | 39/40 | 97.5% |
| Development | c03 | 40/40 | 100% |
| Confirmation | c02 | 40/40 | 100% |
| Confirmation | c03 | 40/40 | 100% |

The missing headline c02 development trajectory terminated at fission 16 and therefore contributed none of the registered landmarks.

At 25x development scale, deterministic replay finds:

- c02: 2/1,000 deaths; 998/1,000 contribute each registered landmark;
- c03: 4/1,000 deaths; landmark contributions are 999/1,000 at generation 20, 998/1,000 at generations 35, 50, and 65, and 997/1,000 at generation 80.

At 25x confirmation scale:

- c02 contributes 1,000/1,000 lineages at generations 20-65 and 999/1,000 at generation 80;
- c03 contributes 1,000/1,000 at every landmark.

The current T2 development-report path filters out trajectories with zero training rows before counting `died`. It therefore undercounts very early extinctions. For example, the existing 25x report says 3/2,000 trajectories died, while deterministic replay finds 6/2,000: two c02 and four c03. This is a bookkeeping issue, not evidence of a material result change, but the manuscript table should use replay-derived counts.

## Preliminary landmark-stratified predictor results

The following values are descriptive rescoring of already-observed outcomes. Gain is direct-history log loss minus full-composite log loss, in nats; positive values favor the composite. No model was refit for these summaries.

### T1 headline confirmation

| Candidate | Landmark | F12 prevalence | Composite gain (nats) |
|---|---:|---:|---:|
| c02 | 20 | 0.3754 | 0.0445 |
| c02 | 35 | 0.2816 | 0.0258 |
| c02 | 50 | 0.2891 | 0.0399 |
| c02 | 65 | 0.3590 | 0.0590 |
| c02 | 80 | 0.3434 | 0.0160 |
| c03 | 20 | 0.3902 | 0.0282 |
| c03 | 35 | 0.3789 | 0.0552 |
| c03 | 50 | 0.3988 | 0.0419 |
| c03 | 65 | 0.3422 | 0.0436 |
| c03 | 80 | 0.3875 | 0.0059 |

### T2 headline confirmation

| Candidate | Landmark | F12 prevalence | Composite gain (nats) |
|---|---:|---:|---:|
| c02 | 20 | 0.3457 | 0.0592 |
| c02 | 35 | 0.3180 | 0.0397 |
| c02 | 50 | 0.3812 | 0.0361 |
| c02 | 65 | 0.3371 | 0.0543 |
| c02 | 80 | 0.3543 | 0.0220 |
| c03 | 20 | 0.4137 | 0.0763 |
| c03 | 35 | 0.3895 | 0.0296 |
| c03 | 50 | 0.3727 | 0.0366 |
| c03 | 65 | 0.3770 | 0.0291 |
| c03 | 80 | 0.3887 | 0.0664 |

The gain is positive at every landmark for both candidates in both headline clean-room confirmations. F12 prevalence varies moderately by phase but does not collapse at later landmarks. Larger retained cohorts show the same qualitative pattern: every T1/T2 candidate-by-landmark gain is positive, although some datasets show a smaller gain at generation 80. The paper should therefore report phase heterogeneity rather than claim landmark invariance.

The originating L50 report already provides validation F12 prevalence by landmark. At F12, mean prevalence is:

| Candidate | g20 | g35 | g50 | g65 | g80 |
|---|---:|---:|---:|---:|---:|
| c02 | 0.4196 | 0.3950 | 0.3850 | 0.4342 | 0.4008 |
| c03 | 0.4231 | 0.4365 | 0.4481 | 0.3846 | 0.3923 |

Originating L54 predictor performance by landmark is not recoverable from the locally retained markdown report because its row-level prediction tables are absent.

## Recommended analysis

No new branch-future campaign is needed. The reviewer can be answered with a compact post-hoc acquisition audit:

1. Produce a completion/extinction table by implementation, cohort, candidate, and matrix, plus candidate-level totals.
2. Produce the fraction of initial lineages contributing each landmark.
3. Rescore F12 prevalence and direct-versus-composite log loss by landmark from retained T1/T2 outcomes and predictions.
4. Add whole-matrix bootstrap intervals to the landmark-specific gains. Treat this as descriptive/post-hoc unless formally registered before calculation.
5. Refit T2 after excluding all incomplete development trajectories, then score only complete confirmation trajectories. No new futures are required.
6. State that the complete-lineage-only confirmation sensitivity is exactly identical to the headline matched-scale analysis, because all headline confirmation trajectories completed.
7. Disclose that an unconditional originating-workflow completion denominator and an exact shared no-retry analysis cannot be reconstructed from the checked-in artifacts.

## What a common-acquisition sensitivity can establish

A practical shared rule is **complete principal lineages only**:

- O already retained complete 100-fission lineages.
- T1 headline development and confirmation are unchanged because every lineage completed on its first attempt.
- T2 can be refit after removing the one incomplete headline development lineage and scored on its already-complete confirmation cohort.

This is a strong and inexpensive sensitivity. It does not fully reconstruct risk among every assembly drawn initially, because failed/rejected O and T1 lineages lack the corresponding branch outcomes. A genuinely common, unconditional no-retry acquisition campaign would require recovered upstream artifacts or new main-path simulation and, for prediction at missing landmarks, a newly defined handling of extinction. It should be presented as optional rather than necessary for the present reviewer response.

## Suggested limitations language

> Predictor results estimate risk among surviving post-fission states observable at the registered landmarks. They do not estimate risk among all assemblies drawn from the initial distribution. The originating workflow retained complete 100-fission lineages, clean-room test 1 allowed deterministic retries when constructing landmark cohorts, and clean-room test 2 used no retry and omitted unavailable later landmarks. In the matched 40-matrix confirmation cohorts, however, every candidate lineage in all three implementations reached all five landmarks, so observed confirmation-cohort attrition cannot explain the reported predictor differences. Development-cohort acquisition remained implementation-specific, and unconditional completion rates for the originating workflow's upstream pool were not retained.

## Suggested results language

> All 80 candidate lineages in each matched confirmation implementation reached generations 20, 35, 50, 65, and 80; the complete-lineage-only confirmation sensitivity was therefore identical to the headline analysis. In clean-room test 1, every headline development and confirmation lineage completed on its first deterministic attempt. In clean-room test 2, 39/40 c02 and 40/40 c03 development lineages reached the registered landmarks, while all confirmation lineages completed. Landmark-stratified rescoring found positive composite-over-history log-loss gains for both candidates at every landmark in both clean-room confirmations, with moderate phase variation but no loss of direction at later landmarks.

## Interpretation

The reviewer has identified a real estimand limitation, but not an observed explanation for the headline confirmation result. The correct response is to narrow the population claim, report acquisition transparently, and add the inexpensive retained-data sensitivity. The matched confirmation evidence is favorable because its observed state coverage is complete across implementations.
