# E01/S18 — Final Dual Verdict and E01 Closeout

## Concise top summary

- **Research step ID:** S18 (`E01-S18-FINAL-DUAL-VERDICT-CLOSEOUT-v1.0.0`)
- **Completion status:** Complete; E01 is closed and control is returned for Chief Scientist and human review.
- **Artifacts written:** Canonical 59-claim Matrix A; seven-question Matrix B; Figure 2–6/Table 1 reconstruction map; final classification registry; claim-evidence crosswalk; status/count/caveat/decision tables; V2 replication artifact and report inputs; immutable-prior, provenance, validation, and closeout hash manifests.
- **Validation result:** PASS — complete 59-claim coverage, dual-matrix separation, directional/exact separation, evidence traceability, required classifications, S01–S17 and legacy-bundle immutability, S17 waiver preservation, and artifact hash replay all passed.
- **Outcome classification:** **CONSTRAINING_CONTRADICTORY**. Paper-facing result: **PARTIAL_DIRECTIONAL_RETROSPECTIVE_RECONSTRUCTION**. Prospective prediction: **not supported within tested scope**. Prospective causal control: **not supported within tested scope**. Full-plan E01 gate: **HOLD**.
- **Caveats or blockers:** Exact author code and several implementation details are unavailable; completed-fit values use the future suffix; `Y=I(H>0.9)` is exactly determined by H; ordinary stability is coupled; 16 claims were not evaluated; two remain underdetermined; and S17's immutable CPU-allowance waiver is operational only.
- **Lay summary:** E01 did make progress: it reconstructed the simulator family, information branch, descriptive spikes, and several association directions. It did not reproduce the paper's aggregate no-trend result, MLP advantage, or max/control/min causal ordering. The closest matches are retrospective and label-coupled, so they cannot support early warning or causal control.
- **Recommended next action:** Human review should choose either the stronger planned E02 or a separately authorized, versioned E01 reopening for S19–S20. S18 starts neither.

## Frozen question

Can the complete E01 evidence be closed with a paper-facing forensic-reproduction matrix that remains strictly separate from prospective-prediction and causal-control adjudication?

**Answer:** Yes. The forensic layer is a partial directional retrospective reconstruction; the prospective and causal layers remain unsupported within the frozen tested scope. A favorable retrospective result does not rescue either failed layer.

## Lay summary

The result is neither “nothing replicated” nor “the paper replicated.” Of the 59 ledgered claims, 3 meet the frozen paper-facing criterion and 17 more point in the same qualitative direction. At the same time, 21 are not supported, 2 cannot be uniquely resolved from the paper, and 16 were not run. The strongest resemblance is descriptive and retrospective: spikes and positive emergence/replicator associations recur. The most consequential claims do not: a genuinely first-quarter-only PhiRL signal does not beat the controls, and maximizing the online score does not outperform control.

This distinction is why E02 remains scientifically worthwhile. E01 has narrowed the plausible explanations to label construction, ordinary stability/attractor geometry, future-fitted parameters, estimator behavior, and weakly separated action scores. E02 can test those explanations without rewriting E01.

## Inputs and immutable evidence

- `/workspace/AGENTS.md`, `/workspace/FULL_PLAN.md`, and `/workspace/RESEARCH_PLAN.md`
- `/workspace/input-attachments/MANIFEST.json`, every `_metadata/ATTACHMENT.md` sidecar, the extracted paper Markdown, and the official arXiv v1 PDF (`77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`)
- the authoritative 59-claim ledger and legacy E01 forensic bundle
- every completed S01–S17 report, status, compact result, and manifest
- frozen S12FR, S13Y, and S14–S17 evidence, with no regenerated trajectory, refit, prediction, intervention, or new estimator

The immutable-prior baseline contains **1,183 files** and **925,229,045 bytes**. The legacy V1 bundle is not modified; the V2 artifact is a compact indexed closeout layer.

## Methods

S18 is deterministic synthesis only. The committed contract froze the five-status vocabulary, candidate-separation rule, exact-versus-directional distinction, all 59 claim adjudications, Matrix B questions, Figure/Table map, final classifications, and post-closeout human-review options before outputs were written.

For each paper claim, S18 copied the original claim text, target, expected direction, and reproduction criterion from the S01 ledger; attached the frozen S14–S17 evidence; and adjudicated four distinct axes:

1. **Final claim status** using the directed five-term vocabulary.
2. **Directional assessment** independent of exact point-estimate agreement.
3. **Quantitative assessment** independent of qualitative direction.
4. **Dependency flags** for completed-fit values, exact-H label scope, and intervention-scoring semantics.

`Directionally supported` does not require an exact paper number. It does require the relevant qualitative direction in the mandatory branches. Candidate 2 and candidate 3 remain separate; pooling is secondary; one favorable candidate cannot rescue a disagreement. Level and change analyses likewise remain separate.

Matrix B was adjudicated independently. Operational success (suffix isolation, online scoring, exact replay) is not equated with predictive or causal success. The status of author implementation is carried separately as `UNDERDETERMINED_AUTHOR_IMPLEMENTATION`.

## Commands

```bash
python -m pytest -q tests/e01/test_s18_final_dual_verdict.py
python -m compileall -q scripts/e01/run_s18_final_dual_verdict.py
python scripts/e01/run_s18_final_dual_verdict.py --stage freeze
python scripts/e01/run_s18_final_dual_verdict.py --stage synthesize
python scripts/e01/run_s18_final_dual_verdict.py --stage validate
```

No scientific compute, GPU work, trajectory generation, model training, estimator fit, or intervention rollout occurred in S18.

## Matrix A — paper-facing forensic reproduction

### Status totals

| Status | Count |
| --- | --- |
| Supported | 3 |
| Directionally supported | 17 |
| Not supported within tested scope | 21 |
| Underdetermined | 2 |
| Not evaluated | 16 |

### Family totals

| Claim family | Supported | Directional | Not supported | Undetermined | Not evaluated |
| --- | --- | --- | --- | --- | --- |
| aggregate_trend | 0 | 0 | 1 | 0 | 0 |
| intervention_absolute | 0 | 2 | 10 | 0 | 0 |
| intervention_contrast | 0 | 7 | 3 | 1 | 0 |
| intervention_time_trend | 0 | 1 | 2 | 0 | 0 |
| metric_distinctiveness | 0 | 0 | 0 | 0 | 12 |
| prediction | 0 | 0 | 5 | 0 | 1 |
| run_level_association | 1 | 3 | 0 | 0 | 0 |
| spike_timing | 0 | 0 | 0 | 0 | 3 |
| spiking | 0 | 1 | 0 | 0 | 0 |
| state_comparison | 1 | 1 | 0 | 1 | 0 |
| temporal_structure | 1 | 2 | 0 | 0 | 0 |

### All 59 claims

| Claim | Final status | Direction | Quantitative fit | Frozen evidence summary |
| --- | --- | --- | --- | --- |
| E01-C001 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | No catalytic-node distinctiveness estimand was executed in the locked S14-S17 sequence. |
| E01-C002 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | No catalytic-edge distinctiveness estimand was executed in the locked S14-S17 sequence. |
| E01-C003 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | No catalytic in-degree distinctiveness estimand was executed in the locked S14-S17 sequence. |
| E01-C004 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | No catalytic out-degree distinctiveness estimand was executed in the locked S14-S17 sequence. |
| E01-C005 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | No betweenness-centrality distinctiveness estimand was executed in the locked S14-S17 sequence. |
| E01-C006 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | No PageRank distinctiveness estimand was executed in the locked S14-S17 sequence. |
| E01-C007 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | No HITS-score distinctiveness estimand was executed in the locked S14-S17 sequence. |
| E01-C008 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | No sample-entropy distinctiveness estimand was executed in the locked S14-S17 sequence. |
| E01-C009 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | No correlation-dimension distinctiveness estimand was executed in the locked S14-S17 sequence. |
| E01-C010 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | No Lyapunov-exponent distinctiveness estimand was executed in the locked S14-S17 sequence. |
| E01-C011 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | No detrended-fluctuation distinctiveness estimand was executed in the locked S14-S17 sequence. |
| E01-C012 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | No generalized-Hurst distinctiveness estimand was executed in the locked S14-S17 sequence. |
| E01-C013 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Candidate 2 and 3 primary aggregate slopes were positive with p=3.1857e-13 and p=4.2835e-4 rather than nonsignificant. |
| E01-C014 | Directionally supported | ALIGNED | CLOSE_OR_WITHIN_REPORTED_DISPERSION | Both candidates had positive three-standard-deviation excursions in 90 of 100 runs. |
| E01-C015 | Directionally supported | ALIGNED | MIXED | Positive runwise Spearman counts were 78 and 76 for levels and 75 and 81 for changes versus 73 reported. |
| E01-C016 | Directionally supported | ALIGNED | MIXED | Positive significant counts were 45 and 44 for levels and 55 and 50 for changes versus 54 reported. |
| E01-C017 | Directionally supported | ALIGNED | DIFFERENT | Mean runwise Spearman values were 0.0611-0.0749 across candidates and analyses versus 0.139 reported. |
| E01-C018 | Supported | ALIGNED | TARGET_MET | One-sample mean-correlation diagnostics were significantly positive for both candidates in both level and change analyses and dependence-aware controls agreed. |
| E01-C019 | Directionally supported | ALIGNED | MIXED | Replicator-state means were higher in 74-86 runs across candidates and level/change analyses versus 57 reported. |
| E01-C020 | Underdetermined | ALIGNED | TARGET_MET | All retained point-pooled and run-summary Mann-Whitney reconstructions gave higher replicator emergence with p<0.001. |
| E01-C021 | Supported | ALIGNED | TARGET_MET | Fisher-combined diagnostics gave p far below 0.001 for both candidates and both analyses. |
| E01-C022 | Directionally supported | ALIGNED | MIXED | Raw Ljung-Box rejection counts were 82 and 79 versus 86 reported. |
| E01-C023 | Directionally supported | ALIGNED | DIFFERENT | Median raw Ljung-Box p-values were 2.66e-8 and 7.01e-6 and therefore strongly indicated dependence though not the reported 2.07e-51 magnitude. |
| E01-C024 | Supported | ALIGNED | TARGET_MET | Differenced trajectories rejected temporal independence in 100 of 100 runs for each candidate. |
| E01-C025 | Not supported within tested scope | CONTRADICTED | DIFFERENT | PhiRL did not outperform composition change in either completed-fit candidate and no p<0.01 advantage occurred. |
| E01-C026 | Not supported within tested scope | MIXED_ACROSS_REQUIRED_BRANCHES | DIFFERENT | Candidate 2 had a tiny directional completed-fit accuracy edge over raw counts but candidate 3 did not and neither comparison had p<0.01. |
| E01-C027 | Not supported within tested scope | CONTRADICTED | DIFFERENT | PhiRL did not outperform molecular flux in either candidate at the paper-like significance criterion. |
| E01-C028 | Not supported within tested scope | CONTRADICTED | DIFFERENT | The majority dummy slightly exceeded PhiRL median accuracy in both candidates. |
| E01-C029 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | Alternative input-output proportions were not evaluated because Option 2 locked exactly the 25/75 split. |
| E01-C030 | Not supported within tested scope | CONTRADICTED | DIFFERENT | All six directed cutoff-causal prospective gates failed in both candidates and all matrices were already positive by cutoff. |
| E01-C031 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | Spike time versus replication-probability association was not estimated in the locked sequence. |
| E01-C032 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | Inter-spike spacing versus replication-probability association was not estimated in the locked sequence. |
| E01-C033 | Not evaluated | NOT_ASSESSED | UNAVAILABLE | Spike height versus replication-probability association was not estimated in the locked sequence. |
| E01-C034 | Directionally supported | ALIGNED | CLOSE_OR_WITHIN_REPORTED_DISPERSION | Max persistence means were 751.3 and 786.0 versus 874 and fell within one paper-reported dispersion for both candidates. |
| E01-C035 | Directionally supported | ALIGNED | CLOSE_OR_WITHIN_REPORTED_DISPERSION | Control persistence means were 771.3 and 789.2 versus 716 and fell within one paper-reported dispersion for both candidates. |
| E01-C036 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Min persistence means were 721.8 and 769.0 versus 559 and were outside one paper-reported dispersion. |
| E01-C037 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Max replication probabilities were 0.9786 and 0.9807 versus 0.88. |
| E01-C038 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Control replication probabilities were 0.9800 and 0.9833 versus 0.88. |
| E01-C039 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Min replication probabilities were 0.9752 and 0.9772 versus 0.80. |
| E01-C040 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Max consistency means were 0.1758 and 0.1244 versus 0.52. |
| E01-C041 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Control consistency means were 0.1237 and 0.1450 versus 0.38. |
| E01-C042 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Min consistency means were 0.1355 and 0.1120 versus 0.42. |
| E01-C043 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Max normalized first-replicator times were 0.00219 and 0.00141 versus a reported 0.36. |
| E01-C044 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Control normalized first-replicator times were 0.00224 and 0.00147 versus a reported 0.37. |
| E01-C045 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Min normalized first-replicator times were 0.00279 and 0.00145 versus a reported 0.40. |
| E01-C046 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Max-minus-control mean persistence was -19.9 and -3.2 and both confidence intervals included zero. |
| E01-C047 | Not supported within tested scope | MIXED_ACROSS_REQUIRED_BRANCHES | MIXED | Max-minus-control consistency was positive for candidate 2 but negative for candidate 3 and neither Wilcoxon comparison was significant. |
| E01-C048 | Directionally supported | ALIGNED | CLOSE_OR_WITHIN_REPORTED_DISPERSION | Max-minus-control probability confidence intervals included zero and paired p-values exceeded 0.05 in both candidates. |
| E01-C049 | Directionally supported | ALIGNED | CLOSE_OR_WITHIN_REPORTED_DISPERSION | Raw first-replicator times were identical for max and control in every matrix and normalized paired intervals included zero in both candidates. |
| E01-C050 | Directionally supported | ALIGNED | DIFFERENT | Max probability slopes were positive in both candidates but not significant at 0.05 with p=0.286 and p=0.052. |
| E01-C051 | Not supported within tested scope | MIXED_ACROSS_REQUIRED_BRANCHES | MIXED | Control was significantly positive in candidate 2 at p=0.0237 and nonsignificant in candidate 3 at p=0.0954. |
| E01-C052 | Not supported within tested scope | CONTRADICTED | DIFFERENT | Min probability slopes were positive and nonsignificant in both candidates rather than negative and significant. |
| E01-C053 | Directionally supported | ALIGNED | UNAVAILABLE | Last-generation mean probability was higher under max than control in both candidates. |
| E01-C054 | Directionally supported | ALIGNED | MIXED | Min persistence was below control in both candidates; the paired interval excluded zero for candidate 2 but not candidate 3. |
| E01-C055 | Directionally supported | ALIGNED | DIFFERENT | Min probability was below control with positive control-minus-min confidence intervals and p<0.05 in both candidates. |
| E01-C056 | Underdetermined | MIXED_ACROSS_REQUIRED_BRANCHES | MIXED | Min consistency was above control in candidate 2 and below control in candidate 3. |
| E01-C057 | Not supported within tested scope | MIXED_ACROSS_REQUIRED_BRANCHES | DIFFERENT | Min was slightly later only in candidate 2 and identical in raw first-replicator steps in candidate 3 with no significant paired effect. |
| E01-C058 | Directionally supported | ALIGNED | MIXED | Max persistence exceeded min in both candidates; the paired interval excluded zero for candidate 2 but not candidate 3. |
| E01-C059 | Directionally supported | ALIGNED | MIXED | Control persistence exceeded min in both candidates; the paired interval excluded zero for candidate 2 but not candidate 3. |

Interpretation boundary: C018, C021, and C024 meet their paper-facing criteria in the locked reconstruction. They do not establish author-code identity, prospective information, or causal control. The 17 directional findings explicitly preserve qualitative resemblance despite numerical differences; they are not silently upgraded to exact reproduction.

## Figure 2–6 and Table 1 reconstruction map

| Component | Reconstruction degree | Summary | Completed-fit dependency | Label dependency | Scoring dependency |
| --- | --- | --- | --- | --- | --- |
| FIGURE_2 | DIRECTIONALLY_SIMILAR_WITH_MAJOR_DISCREPANCY | Punctuated positive excursions and temporal dependence resemble the paper, but the primary aggregate trend is significantly positive rather than trendless. | YES_PRIMARY_VALUES_ARE_COMPLETED_FIT_AND_FUTURE_DEPENDENT | NO_FOR_EMERGENCE_SHAPE | NO |
| FIGURE_3 | DIRECTIONALLY_SIMILAR_RETROSPECTIVE_ONLY | Positive runwise association counts, means, and one-sample diagnostics recur, but are label-coupled and reverse under past-only refitting. | YES | YES_Y_EQUALS_I_H_GT_0_9 | NO |
| FIGURE_4 | DIRECTIONALLY_SIMILAR_RETROSPECTIVE_ONLY | Replicator-state emergence contrasts and Fisher diagnostics point in the paper's direction, with paper scope unresolved and exact-H/stability coupling intact. | YES | YES_Y_EQUALS_I_H_GT_0_9 | NO |
| FIGURE_5 | DIFFERENT_NOT_SUPPORTED | Neither completed-fit nor cutoff-causal PhiRL achieved the reported prediction advantage; accuracy largely reflected approximately 98% prevalence and balanced accuracy remained near 0.5. | PAPER_LIKE_MODE_YES_BUT_FAILED;CUTOFF_MODE_NO_AND_FAILED | YES_TARGET_IS_EXACT_H_THRESHOLD | NO |
| FIGURE_6 | PROCEDURE_RECONSTRUCTED_OUTCOMES_DIFFERENT | The prospective prefix-only max/control/min procedure executed and replayed exactly, but max was below control and the reported bidirectional ordering did not recur. | NO | YES_OUTCOME_USES_Y_EQUALS_I_H_GT_0_9 | YES_APPEND_AND_REFIT_CURRENT_PREFIX_IS_ONE_LITERAL_RECONSTRUCTION |
| TABLE_1 | MIXED_PERSISTENCE_SCALE_SIMILAR_OTHER_VALUES_DIFFERENT | Max and control persistence means fall within one paper-reported but undefined dispersion; min persistence and all probability, consistency, and timing targets differ materially. | NO | YES_OUTCOME_USES_Y_EQUALS_I_H_GT_0_9 | YES |

### What resembles the paper

- Figure 2-like punctuated positive excursions occur in 90/100 runs for each candidate; differenced temporal dependence is 100/100.
- Figures 3–4-like positive association and replicator-state contrasts recur across both candidates and both level/change analyses.
- Max and control Table 1 persistence means are on a broadly similar absolute scale, within one paper-reported but undefined dispersion.
- Several intervention contrasts point in the reported direction: max and control do not materially differ in overall occupancy; last-generation max occupancy exceeds control; and min tends to reduce persistence/occupancy.

### What differs

- Figure 2's aggregate no-trend result does not recur: the primary slopes are significantly positive.
- Figure 5's PhiRL advantage does not recur in either completed-fit or cutoff-causal mode; prevalence drives near-98% raw accuracy while balanced accuracy is about 0.5.
- Figure 6's max >= control >= min ordering fails because max mean persistence and occupancy are below control in both candidates.
- Most Table 1 probability, consistency, and timing values differ materially; min persistence also differs.

## Matrix B — prospective and causal interpretation

| Question | Status | Eligible mode | Finding |
| --- | --- | --- | --- |
| Past-only early warning | Not supported within tested scope | PAST_ONLY_OR_CUTOFF_CAUSAL_ONLY | Past-only S13Y/S15 association directions reversed, and the cutoff-causal S16 prediction gates failed in both candidates. |
| Incremental prediction beyond exact H and ordinary stability | Not supported within tested scope | CUTOFF_CAUSAL_ONLY | Y is exactly I(H>0.9), unrestricted incremental information beyond contemporaneous exact H is zero, ordinary stability is coupled, and S16 gate 3 failed in both candidates. |
| Future-suffix independence | Directionally supported | CUTOFF_CAUSAL_ONLY | All cutoff-causal suffix-invariance checks passed, while the paper-like completed-fit branch is explicitly future-dependent and therefore ineligible for prospective interpretation. |
| Online action scoring | Supported | APPEND_AND_REFIT_CURRENT_PREFIX_RECONSTRUCTION | The locked scorer used only the current prefix, applied every selected action, and replayed exactly; exact author scorer identity remains unavailable. |
| Intervention reproducibility | Supported | LOCKED_LITERAL_RECONSTRUCTION | All 72 intervention trajectories, 4,800 applied treated actions, source fits, and trajectory replays were exact under the locked implementation. |
| Action separability | Not supported within tested scope | LOCKED_LITERAL_RECONSTRUCTION | Only 0.9258-0.9400 of selected action gaps exceeded the frozen 0.95 separability scale; the required separability gate failed in both candidates. |
| Bidirectional causal control | Not supported within tested scope | PROSPECTIVE_PAIRED_INTERVENTION | Max emergence was below control for mean persistence and occupancy in both candidates, max-minus-control uncertainty included zero, and matched-random outcome exclusion was unavailable. |

The completed-fit branch fails future-suffix independence by construction. The cutoff-causal branch passes suffix-invariance checks, but that operational success does not rescue prediction because its comparative, uncertainty, incremental-value, and calibration gates fail. Likewise, S17 proves that a literal online scorer can be executed and replayed, not that the score causally controls replication.

## Directed final classifications

| Classification | Status | Meaning |
| --- | --- | --- |
| LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE | RETAINED_ESTABLISHED | Figures 3-4-like association and state contrasts recur only as retrospective evidence under a target exactly determined by H. |
| RETROSPECTIVE_TEMPORAL_FITTING_DEPENDENCE | RETAINED_ESTABLISHED | Completed-fit partitions and Gaussian parameters use the future suffix, while past-only associations reverse. |
| RETROSPECTIVE_PREDICTION_RESEMBLANCE | NOT_SUPPORTED_WITHIN_TESTED_SCOPE | The completed-fit PhiRL MLP did not beat all paper baselines or the majority dummy under the locked Figure 5 reconstruction. |
| PROSPECTIVE_PREDICTION_SUPPORTED | NOT_SUPPORTED_WITHIN_TESTED_SCOPE | All six directed cutoff-causal gates failed in both candidates. |
| LITERAL_INTERVENTION_ORDERING_RESEMBLANCE | NOT_SUPPORTED_WITHIN_TESTED_SCOPE | The literal max >= control >= min persistence or occupancy ordering failed in both candidates. |
| PROSPECTIVE_CAUSAL_CONTROL_SUPPORTED | NOT_SUPPORTED_WITHIN_TESTED_SCOPE | Bidirectional paired effects, separability, and matched-random outcome exclusion did not jointly pass. |
| NOT_SUPPORTED_WITHIN_TESTED_SCOPE | RETAINED_SCOPE_BOUNDARY | Non-support is limited to the frozen candidates, branches, labels, estimators, prediction design, and intervention scorer tested in E01. |
| UNDERDETERMINED_AUTHOR_IMPLEMENTATION | RETAINED_UNRESOLVED | Exact author simulator, aggregation, MLP layout, uncertainty definitions, and intervention scoring details remain unavailable. |
| PUNCTUATED_EXCURSIONS_WITH_AGGREGATE_TREND_DISCREPANCY | RETAINED_ESTABLISHED | Figure 2-like spikes recur, but the aggregate no-trend target does not. |

## E01 verdict

- **Paper-facing forensic reproduction:** partial and directional, concentrated in completed-fit descriptive and association results.
- **Past-only early warning:** not supported within tested scope; association directions reverse under past-only fitting.
- **Incremental prediction beyond H/stability:** not supported; exact contemporaneous H defines Y and ordinary stability is coupled.
- **Retrospective completed-fit prediction resemblance:** not supported within tested scope.
- **Prospective cutoff-causal prediction:** not supported within tested scope in either candidate.
- **Literal intervention ordering resemblance:** not supported within tested scope in either candidate.
- **Prospective bidirectional causal control:** not supported within tested scope.
- **Exact author implementation:** underdetermined.
- **Full-plan gate:** `HOLD` because the result is a partial retrospective forensic reconstruction, fewer than two central evidence layers reproduce robustly, and author implementation remains materially unresolved.

## Validation

Validation passed for:

- exact 59-claim coverage and uniqueness;
- frozen status counts and vocabulary;
- distinct directional and quantitative assessments;
- complete Matrix A/Matrix B separation;
- seven Matrix B questions and all Figure 2–6/Table 1 entries;
- required directed classifications;
- every claim-to-evidence path and SHA-256 digest;
- byte-for-byte S01–S17 preservation;
- byte-for-byte legacy V1 bundle preservation;
- unchanged S17 CPU-waiver evidence;
- identical canonical and V2 report-input copies;
- S18/V2 artifact manifests and terminal closeout manifest;
- inactive S19/S20 and E02 options.

## Provenance

- Repository branch: `eidosoma/groups/42`
- Frozen pushed commit: `b92eb97e2b2e961e3aea4657561f49cab0c4848f`
- Original paper: `/cache/e01_s03/downloads/paper-2607.28250v1.pdf`
- Original paper SHA-256: `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`
- S18 contract: `configs/e01/s18_final_dual_verdict_contract.json`
- Claim adjudication lock: `configs/e01/s18_claim_adjudications.csv`
- Canonical output: `/artifacts/research_steps/S18`
- V2 replication artifact: `/artifacts/E01_forensic_replication_artifact_v2`
- Legacy V1 bundle: `/artifacts/E01_forensic_replication_bundle` (unchanged)

Python standard-library synthesis was used. No new dependency was installed.

## Caveats, blockers, failed assumptions, and limitations

- `Y=I(H>0.9)`: exact H fully determines the binary target, so unrestricted incremental target information beyond exact H is zero.
- Completed-fit PhiRL values depend on full-trajectory partitions/Gaussian parameters; their resemblance is retrospective.
- The historical post-fission label points negatively, and past-only refitting reverses the association direction.
- Exact author simulator, aggregation, MLP layout, uncertainty meaning, and intervention scoring remain unavailable.
- Sixteen paper claims were not evaluated; this is not evidence against them.
- The paper's Mann–Whitney scope, Ljung–Box lag, spike threshold scope, Table 1 dispersion type, and first-replicator unit remain unresolved.
- S17's human CPU-allowance waiver remains byte-for-byte preserved and changes no scientific conclusion.
- S17 did not include a random-action rollout because the exact 72-trajectory scope was locked; matched-random outcome exclusion is therefore underdetermined.
- These are simulation results in a reconstructed GARD model, not evidence about living systems or prebiotic chemistry.

## Human-review options and recommended next action

| Option | Title | Scope | Active |
| --- | --- | --- | --- |
| OPTION_A_STRONGER_E02 | Authorize the planned stronger E02 | Use the frozen E01 bundle to adjudicate leakage, labels, estimator stability, attractor geometry, incremental value, and matched intervention controls in a separate workspace. | False |
| OPTION_B_VERSIONED_E01_REOPEN_S19_S20 | Reopen E01 under a new version for S19-S20 | S19 would prospectively lock and evaluate the 16 Matrix-A claims left not evaluated (C001-C012, C029, and C031-C033), using frozen E01 data wherever valid. S20 would be a separately locked confirmation and author-ambiguity resolution step defined before any new outcome access. | False |
| OPTION_C_CLOSE_WITHOUT_CONTINUATION | Archive after E01 | Retain the V2 artifact and take no immediate follow-up action. | False |

The scientific recommendation is **Option A: a stronger separate E02**, focused on prospective onset-risk targets, incremental value beyond H and attractor stability, estimator/partition robustness, and matched intervention controls. This follows the existing group plan and gives negative E01 results productive use.

**Option B is retained because the human requested it:** E01 may be explicitly reopened under a new version for S19–S20. A sensible S19 would cover the 16 not-evaluated claims (C001–C012, C029, C031–C033) under a prospectively frozen contract. S20 would be a separately locked confirmation/author-ambiguity step defined before outcome access. This option is not queued, does not amend the present completed-step record, and cannot mutate S01–S18.

## Terminal boundary

E01 is closed at S18 under the current authorization. Control returns to the Chief Scientist and human reviewer. E02 is not started, S19–S20 are not queued, no report bundle is automatically generated beyond the user-requested compact V2 replication artifact/report-input layer, and no estimator, simulator, threshold, label, or scorer is added.
