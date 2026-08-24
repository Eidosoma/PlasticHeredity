# Paper Replication Issues V1

## Document status and decision boundary

- **Research-step context:** Post-S13Y documentation-only human review; this is not a new research step and does not execute S14 or any continuation option.
- **Completion status:** `COMPLETE`.
- **Artifacts written:** `PAPER_REPLICATION_ISSUES_V1.md`, `PAPER_REPLICATION_ISSUES_V1.sha256`, and `PAPER_REPLICATION_ISSUES_V1.provenance.json`.
- **Validation result:** `PASS`—three and only three options are present; all requested evidence boundaries and option fields are covered; quoted S13Y anchor values agree with the frozen machine-readable results; and the pre-existing artifact corpus hash is unchanged.
- **Outcome classification:** `DOCUMENTATION_AND_DECISION_SUPPORT_COMPLETE` (not a scientific-result reclassification).
- **Caveats or blockers:** The author implementation remains unavailable; working assumptions about the authors and the separate gene-regulatory-network paper are human supplied; the strongest positive result is retrospective and label coupled; no option has been selected.
- **Recommended next action:** Human selection among exactly one of the three options below, followed—if authorized—by a separately versioned plan amendment.
- **Purpose:** Explain, in plain language but with technical detail, what E01 has and has not reproduced from *Causal Architecture Dynamics Prior to Arrival of Self-replicators in a Model of Catalytic Networks Relevant to Origin-of-Life*, and present exactly three ways to finish or formally dispose of S14–S18 before E02+.
- **Evidence cutoff:** Completed S13Y, `E01-S13Y-CLEAN-DIRECTIONAL-CONFIRMATION-v1.0.0`, on 2026-08-07.
- **Action taken here:** Documentation and decision support only.
- **Action not taken:** No author contact, simulation, estimator search, source fit, prediction, intervention, S14–S18 execution, E02 execution, or report-bundle generation.
- **Prior evidence:** S01–S13Y and their historical classifications remain unchanged. This document summarizes them; it does not reclassify them.
- **Integrity:** The document SHA-256, source provenance, completeness checks, and pre/post hash audit of the prior artifact corpus are recorded in `PAPER_REPLICATION_ISSUES_V1.provenance.json` and `PAPER_REPLICATION_ISSUES_V1.sha256`.

## Executive summary

We did not obtain an exact reproduction of the paper. The main reason is not that one program produced a clean contradictory answer. The paper's exact GARD program, self-replicator labelling code, local causal-emergence calculation, prediction layout, and intervention scorer were not available to us. Public code supplied valuable clues, but it was not the unpublished GARD-paper implementation. Many choices that look minor—what counts as a molecular step, whether a daughter state is recorded, which composition is labelled, whether a statistical model sees the completed trajectory, and which PhiID atoms are called causal emergence—materially changed the result.

The strongest result we found is narrow but real. An adaptive search in S13X found that completed-trajectory **PhiRL source-defined emergence** was positively associated with a molecular-state label defined by adjacent composition similarity, `Y=I{H>0.9}`. S13Y then froze that exact branch and tested it on 100 genuinely new catalytic matrices under each of two independently confirmed time-base candidates. Both candidates passed the preregistered retrospective association and replicator-minus-drift gates.

That confirmation does **not** establish early warning or causal control:

1. The causal-emergence model was fitted to the completed trajectory. Its partition, means, and covariance matrices could therefore depend on observations later than the time being scored.
2. The binary replication target was exactly obtained by thresholding the same incoming similarity coordinate: `Y=I{H>0.9}`. Across 180,435 molecular rows there were zero identity mismatches. Given exact H, the label has zero remaining uncertainty: `H(Y|H)=0`. Consequently, unrestricted conditional information from emergence about this same label is also zero: `I(E;Y|H)=0`.
3. When the identical PhiRL source pipeline was refitted using only observations available up to each endpoint, the association reversed sign. Median trajectory Spearman correlations were `-0.0741` and `-0.0693`, with both positive-direction circular-shift p-values equal to `1.0`.
4. A small S13X intervention pilot reproduced the paper-directed max/control/min ordering in only 1 of 8 persistence-or-occupancy comparisons. The earlier strict intervention study allowed only 1 distinguishable action among 1,090 treated opportunities.

The defensible conclusion is therefore:

> We found a cleanly confirmed, paper-like **retrospective and label-coupled resemblance**, but not a recovered prospective predictor, independent information contribution beyond H, intervention effect, author-code implementation, fixed-window Phi-r trajectory, or exact paper replication.

Under the human-supplied working assumption that the paper's reported claims are correct, our nonreplications point to missing or unresolved implementation details. They are not evidence of misconduct, and this document makes no such suggestion.

## Respectful working context

The human direction supplies three pieces of context:

- The authors are highly respected.
- We should work on the assumption that the reported claims are correct.
- A separate effort also failed to reproduce the paper *Associative conditioning in gene regulatory network models increases integrative causal emergence*.

We adopt those points as **working context**, not as independently verified findings of this E01 workspace. These human-supplied assumptions are not independently verified by this artifact. The second paper is peer-reviewed according to the human direction, but peer review is not itself proof that an independent replication has occurred. We have not established here whether that work has been independently replicated. The respectful and scientifically useful interpretation is that our implementations may still be missing shared conventions, preprocessing, source behavior, indexing, or analysis choices used across the authors' research program.

The decision not to contact the authors is preserved. It means that remaining ambiguities must be handled through source archaeology, explicit branches, sensitivity analysis, or formal underdetermination—not by guessing what the authors did.

## What the paper reports

The paper describes 100 independent GARD assemblies with 100 molecular types, initial mass 40, fission threshold 80, 100 generations, maximum 1,000 growth steps per generation, lognormal catalytic parameters `A=-4` and `sigma=4`, distinct-type initialization without replacement, Poisson updates, and binomial fission with probability 0.5.

Its principal reported results are:

| Claim family | Paper-reported target |
| --- | --- |
| Aggregate trajectory | No significant aggregate causal-emergence trend over molecular time, linear-regression `p=0.1995`. |
| Punctuated behavior | Most individual trajectories contain positive excursions above the overall mean plus three standard deviations. |
| Replication association | 73/100 runwise correlations are positive; 54 runs are positive and significant; reported mean rho is approximately 0.139. |
| Replicator versus drift | 57/100 runs have higher mean causal emergence during replication; Mann–Whitney and Fisher-combined evidence are reported as `p<0.001`. |
| Temporal structure | 86/100 raw trajectories and 100/100 differenced trajectories reject temporal independence. |
| Prediction | An MLP using the first 25% of the causal-emergence trajectory reportedly predicts the remaining replication trajectory or initial appearance better than composition-change, raw-composition, flux, and majority-class baselines. |
| Spike timing | Replication probability reportedly correlates with spike time (`rho=0.66`) and inter-spike distance (`rho=0.71`), but not spike height. |
| Intervention | Adding or deleting the molecule that maximizes causal emergence after every fission reportedly improves self-replicator persistence and consistency; minimizing it worsens outcomes. |
| Table 1 control | Persistence `716±198`, probability `88±3%`, consistency `0.38±0.06`, and time to first replicator `37±27` with a unit ambiguity. |
| Table 1 max/min | Max persistence `874±233`, probability `88±3%`, consistency `0.52±0.04`; min persistence `559±99`, probability `80±3%`, consistency `0.42±0.04`. |

The paper says code will be made available upon publication and supplies no supplementary materials in the version available to this workspace.

## Why this was not a push-button replication

### The claim and ambiguity ledgers

S01 separated the paper into **59 independently adjudicable claims**:

- 12 metric-distinctiveness claims;
- 1 aggregate-trend claim;
- 1 spiking claim;
- 4 run-level association claims;
- 3 replicator-versus-drift claims;
- 3 temporal-structure claims;
- 6 prediction claims;
- 3 spike-timing claims;
- 26 intervention claims.

Only one claim was fully specified from the paper, four were testable with declared ambiguity, and 54 were initially underdetermined. S01 also retained 12 paper-internal discrepancies instead of smoothing them over. Material examples include:

- the Results describe correlation with Phi-r levels while the Figure 3 caption describes changes in Phi-r;
- “54 of 73 positive runs” and “54% of 100 runs” use different denominators;
- the prediction text alternates between predicting the remaining state trajectory and predicting initial appearance;
- Table 1 prints percent signs for time to first replicator while its note says molecular steps;
- the text says minimization worsened all four outcomes, while Table 1 gives min consistency `0.42`, above control `0.38`, where higher is defined as better;
- overall max probability is described as not different from control in one place but higher in the Discussion;
- spike threshold scope, Mann–Whitney scope, Table 1 dispersion, and Figure 6 coefficient identity are not specified.

S02 expanded these into **105 traceable ambiguity items** across source provenance, GARD dynamics, preprocessing, labels, Phi estimation, descriptive statistics, prediction, and intervention. The v0.3.0 registry has 120 parameters and remains intentionally non-executable as a universal paper specification: 64 parameters remain unresolved, conflicting, or evidence-deferred, and 21 branch sets require separate immutable specification identities. This is why apparently reasonable reconstructions can disagree without either implementation containing an elementary coding error.

### Source and implementation uncertainty

The source archaeology established three distinct evidence layers:

1. **Historical public GARD** at commit `86dff6320d5ae91b4e831471079ff46749b14df9` uses one categorical join/loss event per loop, initialization with replacement, a fixed-size split, first-child continuation, and no paper-style `max_steps` condition.
2. **Paper prose** describes vector-valued Poisson join/loss updates, initialization without replacement, binomial fission, `maxsteps=1000`, and the 100/40/80/100/-4/4 parameter tuple.
3. **Public information-theory lineage** supplies IIGR at `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7` and PhiRL at `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`, but neither repository supplies the paper's GARD pipeline.

The public information code distinguishes:

- `integrated = local_phi_r(...)`; and
- `emergence = synergy + downward causation`.

S12D verified the latter identity against source behavior in 40/40 fixture cases, to maximum component error `1.78e-15`. This establishes what the public source computes. It does not establish which scalar, source revision, preprocessing, partition, or temporal fitting mode generated the paper's figures.

The public source family fits a lagged-MI/Fiedler bipartition and local Gaussian distributions to the complete array supplied to it. If the complete trajectory is supplied, a value plotted near the beginning can change when later observations are added or removed. That is valid as a retrospective descriptive calculation, but it is not a past-only early-warning calculation.

## Chronology of what we tested

### Foundation and software validation: S01–S10

- **S01–S03:** Built the 59-claim ledger, 105-item ambiguity ledger, 12-discrepancy taxonomy, source archive, environment lock, dependency hashes, source commits, license notes, and runtime identities.
- **S04:** Reconstructed source-traceable historical GARD behavior, including its one-event update, first-child lineage, and historical non-drift routines. It was explicitly not called the author implementation.
- **S05–S07:** Built an independent integer-state engine and validated it against hand calculations and matched historical distributions. S07 passed 26/26 stochastic tests, 54/54 deterministic invariants, and 7/7 failure injections. This gives high confidence in the software for the specifications it actually implements.
- **S08:** Reconstructed historical H-based labels plus cosine, Euclidean, and Aitchison families. The families disagreed materially, and retrospective labels flipped 21 fixture labels relative to their past-only versions. No author label implementation was identified.
- **S09:** Tested 13 zero treatments and multiple log-ratio representations. All observations were retained with explicit eligibility. No author pseudocount or dropped-component rule was identified.
- **S10:** Validated the Gaussian PhiID reference on known systems, source comparisons, invariance tests, and CPU/GPU cross-checks. The strict validated branch requires at least 512 effective observations. One optional OmegaID discrete branch failed relabelling tests and was excluded.

### Fixed-window and strict analyses: S11–S12

- **S11:** Tried a separately validated small-window/high-dimensional estimator for all 16 planned window/lag pairs. It passed 11/16 gate families but failed truth, shuffle, partition, and relabelling requirements. Result: 0/576 branches and 0/16 fixed pairs were eligible; all 33,984 fixed estimate rows remained explicitly nonnumeric.
- **S11R:** Tested one bounded repair with development/confirmation separation. It passed 16/19 gate families but again failed known-truth and dimension-8 partition-agreement gates. No fixed-window GARD estimate was released.
- **S12:** Used only the strict expanding branch after 512 effective observations on 12 baselines. Each branch was numeric for 94.6352% of molecular observations; first eligibility occurred around generation 10. All six preregistered prospective association summaries were negative, with median trajectory rho from `-0.1341` to `-0.0387`. The intervention branch found only one separable action among 1,090 treated post-origin opportunities; 1,089 were suppressed as indistinguishable. The 59-claim matrix therefore contained 0 `SUPPORTED`, 0 `DIRECTIONALLY_SUPPORTED`, 7 `NOT_SUPPORTED_WITHIN_STRICT_SCOPE`, 40 `UNDERDETERMINED`, and 12 `NOT_EVALUATED` claims. That matrix remains historically unchanged.

### Public-source audit: S12B–S12D

- **S12B:** Stopped before GARD outcomes because the IIGR wrapper disagreed with source status on a singular fixture.
- **S12C:** A narrowly authorized wrapper correction passed 14/14 development and 14/14 untouched confirmation rows. On the 12 S12 trajectories, neither IIGR nor PhiRL `local_phi_r` produced a coherent positive full-trajectory result. Prefix medians were `-0.0193` and `-0.0230`. Full-versus-prefix correlations were only `0.198` and `0.257`; spike Jaccards were `0.0615` and `0.0228`; median partition adjusted Rand indices were `0.0157` and `0.0464`. The public-source classification was `SOURCE_FAMILY_NOT_SUPPORTED` for that scalar and dataset.
- **S12D:** Verified the distinct source-defined emergence scalar exactly, but one of 24 preregistered historical-GARD lineages became extinct after four fissions. The all-24 firewall stopped the scientific test. This was an operational failure, not a negative emergence result.

### Recovering a plausible paper time base: S12E–S12FR

- **S12E:** Tested five source-grounded engines before labels or information metrics. Historical eventwise dynamics were much too long (median 5,095.5 steps); unscaled paper-Poisson candidates were too short (medians 434.5–443.5); reservoir-one dynamics were extreme. None passed the frozen 500–1,500-step gate.
- **S12F:** Treated the paper figures as time-base data and introduced a bounded Poisson exposure inference. The first development run stopped on an exact-replay comparison even though 16/16 benchmark pairs replayed.
- **S12FR:** Diagnosed that failure as paired, same-cause NaNs in undefined summaries rather than finite or discrete trajectory divergence. A new untouched 2,048-pair suite passed unanimously. Approximate Bayesian computation then identified three nonunique paper-compatible time-base candidates. Their median confirmation clocks were 812.0, 889.5, and 872.0 molecular steps. This was classified `NONIDENTIFIABLE_TIMEBASE_ENSEMBLE`: the paper-visible clock could be matched, but daughter, overshoot, and recording conventions could not be uniquely identified.

The two candidates retained in the later clean work were:

- **Candidate 2:** exposure `h=0.6031526490073492`, first-daughter continuation, trimming only excess newly joined molecules, and a clock that records every selected daughter boundary.
- **Candidate 3:** exposure `h=0.5613315384859516`, random-nonempty daughter continuation, the same trimming rule, and the same boundary-inclusive clock.

These are confirmed paper-time-base reconstructions, not probabilities of author identity.

### Ensemble attempts, operational failures, and the first held-out result: S12G–S13RRR

Several steps stopped on operational rules before scientific inference. These failures are relevant because repeated post-failure repairs weaken confirmatory credibility even when the repairs preserve values.

- **S12G:** One retained-overshoot candidate had nine zero-update generations, so its C0 clock lacked a distinct endpoint. Ninety-five of 96 tasks completed, but none were promoted.
- **S12H:** Reinterpreted that candidate with a uniform daughter-boundary clock. It passed 13/14 upstream gates but narrowly failed aggregate support because 2/32 trajectories exceeded the paper-axis ceiling, where at most 1/32 was allowed.
- **S12I:** A human waiver allowed that near-envelope candidate as exploratory. All 96 source tasks passed, but statistics failed because the generic interface expected `rawObservationIndex` while prefix data exposed `endpointRawObservationIndex`.
- **S12J:** A one-column, value-preserving adapter was confirmed. On 32 trajectories per candidate, all three candidates failed the frozen association and drift gates. IIGR full median rho values were `-0.0168`, `0.0155`, and `-0.00264`; prefix medians were small positive values but did not pass.
- **S13:** Generated 100 new shared matrices and 200 complete trajectories for candidates 2 and 3. All simulations and source tasks passed. Global aggregation stopped because five all-null optional label tables had Arrow `null` types rather than the canonical string/double types.
- **S13R and S13RR:** Two explicit schema overrides normalized label, prefix, suffix, and seed views without changing rows or values. New fail-closed checks then exposed a fixed replay-count mismatch and two reporting-column orders.
- **S13RRR:** A third human override made replay availability-aware: 3,552/3,552 executable sentinels passed and 48 unavailable slots were accounted for. The original analysis then ran twice identically. Both confirmed candidates failed the retrospective, drift, prospective, coherent, and punctuated gates under the frozen historical post-fission label. Candidate-2 retrospective IIGR median rho was `-0.00615`; candidate 3 was `-0.00841`. Prefix medians were `-0.00300` and `-0.03088`. Classification: `HELD_OUT_TWO_CANDIDATE_SCALEUP_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE`.

S13RRR is valid as a transparent, value-preserving analysis, but it is explicitly a third post-outcome override. It is weaker procedurally than a clean first-pass analysis and cannot erase the earlier failures.

### The adaptive lead: S13X

S13X was explicitly authorized to search creatively rather than obey prior scientific gates. It retained every attempted specification in an 8,498-row chronological ledger. The campaign screened 7,744 first-pass pipelines and 360 focused-neighbourhood pipelines across label scope, source implementation, metric atoms, level versus difference, alignment, and related choices.

The highest-leverage gap was **where and how replication was labelled**. Historical post-fission labels remained near zero or negative. Labelling every molecular state by similarity to its immediately preceding composition changed the direction.

The lead branch was:

`PhiRL regularized source-defined emergence` + `completed-trajectory level values` + `molecular adjacent-incoming H>0.9 label` + `same-state alignment`.

Across the development and diagnostic sets, candidate 2 had 73/100 positive correlations and candidate 3 had 79/99. The disjoint diagnostic medians were `0.0298` and `0.0466`, with positive bootstrap intervals and circular-shift p-values `1/4097`.

An outcome-guided `H>0.97` sensitivity produced occupancy around 0.873–0.887, close to the paper's 0.88, and stronger association fractions. It did not reproduce onset (near 3 rather than 37) or consistency (near 0.09 rather than 0.38), and it cannot replace the paper-stated `H>0.9` threshold.

The same exact PhiRL source pipeline refitted past-only reversed the sign: median rho `-0.0621/-0.0720` at `H>0.9`. A four-triplet retrospective intervention pilot reproduced the paper-directed `max >= control >= min` persistence/occupancy ordering in only 1/8 comparisons despite exact scorer, trajectory, and action replay.

S13X therefore produced an adaptive retrospective explanation, not confirmation.

### The clean test of that lead: S13Y

S13Y froze the exact S13X `H>0.9` lead before generating data. It used 100 genuinely new catalytic matrices and matched initial states, shared across candidates 2 and 3, for 200 complete 100-fission trajectories. The preregistration and implementation were pushed at commit `41b235c882effe03accf86256e586c2530eadf66` before outcome access.

All 200 trajectories and 200 source tasks passed exact replay. All 41,115 structural suffix checks and all 1,773 executed deletion/shuffle/replacement sentinels passed. Source coverage, component identity, shared pairing, seed separation, schema, statistics replay, prior-artifact immutability, runtime, storage, and artifact hashes all passed.

The retrospective primary result passed for both candidates:

| Candidate | Defined correlations | Positive | Median rho | 95% trajectory bootstrap | Circular-shift p | Higher during labelled replication | Median emergence difference |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Candidate 2 | 99 | 78 (78.8%) | 0.0580 | [0.0390, 0.0751] | 0.000244 | 75/99 (75.8%) | 0.3958 |
| Candidate 3 | 99 | 76 (76.8%) | 0.0554 | [0.0305, 0.0761] | 0.000244 | 74/99 (74.7%) | 0.5162 |

This is close in direction and positive-run fraction to the paper's 73/100 and 57/100 summaries. It is not a match to the paper's full causal story.

#### Exact label determinism and incremental information

For this branch, incoming H was computed first and the binary target was defined as:

`Y = 1 if H > 0.9, else 0`.

There were:

- 0 mismatches in 87,487 candidate-2 rows;
- 0 mismatches in 92,948 candidate-3 rows;
- baseline-H classification accuracy of 1.0 for both candidates.

Therefore exact H already completely determines Y. In information terms:

- `H(Y|H)=0`; and
- unrestricted `I(E;Y|H)=0`.

PhiRL emergence correlated with continuous H (median rho approximately `0.270`) and with negative ordinary L2 composition change (approximately `0.276–0.278`). A prespecified smooth H/L2 regression did not establish a separate threshold-specific effect: its emergence coefficients had bootstrap intervals crossing zero for both candidates. This does not mean emergence and H are numerically identical. It means emergence cannot add unrestricted information about a binary variable that is already a deterministic function of exact H.

#### Past-only sign reversal

At each eligible post-fission endpoint after 256 transitions, S13Y refitted the same PhiRL source pipeline using only the prefix available at that time. It reused no future-fitted partition, mean, or covariance.

| Candidate | Defined trajectories | Positive | Median rho | 95% trajectory bootstrap | Positive-direction shift p |
| --- | ---: | ---: | ---: | --- | ---: |
| Candidate 2 | 85 | 29 (34.1%) | -0.0741 | [-0.1095, -0.0522] | 1.0 |
| Candidate 3 | 84 | 30 (35.7%) | -0.0693 | [-0.1071, -0.0296] | 1.0 |

This is not merely loss of significance. It is a replicated direction reversal. The completed-fit association must therefore remain `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL`; it cannot be used as evidence that causal emergence forecasts replication from the past.

#### Spikes and temporal behavior

S13Y did find punctuated completed-fit trajectories:

- positive three-standard-deviation excursions occurred in 90% of runs for each candidate;
- robust MAD excursions occurred in 100%;
- raw Ljung–Box significance occurred in 82% and 79%;
- differenced Ljung–Box significance occurred in 100% for both.

But the aggregate trend did **not** match the paper's non-significant `p=0.1995`: S13Y's aggregate slope p-values were `3.19e-13` and `0.000428`. Thus even the clean retrospective lead does not reproduce the full Figure 2 fingerprint.

The frozen historical post-fission label also pointed negatively in S13Y: median correlations were `-0.0112` and `-0.0208`, and fewer than half the runs had higher emergence during that labelled state. This is further evidence that the positive result is label-scope dependent.

S13Y's classification is exactly `LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE`.

## Findings by scientific question

| Question | Best current E01 answer |
| --- | --- |
| Can we simulate validated GARD-like dynamics? | Yes. Historical and independent engines are well tested for their explicit specifications. |
| Did we identify the authors' exact GARD implementation? | No. Paper prose and historical source differ, and three paper-time-compatible parameter/clock combinations remained nonidentifiable. |
| Did we reconstruct the paper's fixed-window/local Phi-r? | No. S11 and S11R failed validation at all 16 planned small-window/lag pairs. |
| Did strict past-only Phi-r become numerically usable? | Yes, after 512 effective samples, usually around generation 10 in S12. |
| Did strict past-only Phi-r show the reported positive relationship? | No. All six restricted S12 summaries were negative. |
| Did public `local_phi_r` reproduce the reported association? | No on the 12-run S12C audit; neither IIGR nor PhiRL passed its coherence gate. |
| Is public `emergence = synergy + downward causation` implemented correctly by our wrapper? | Yes. Source-metric identity passed 40/40 fixtures. |
| Did the frozen historical post-fission label support the source-emergence relationship? | No in the S12J sensitivity set, S13RRR held-out analysis, and S13Y historical-label comparator. |
| Is there any paper-like positive association? | Yes, for completed-fit PhiRL emergence aligned to a molecular adjacent-incoming `H>0.9` label. S13Y confirmed it on new matrices under both time-base candidates. |
| Does that association add unrestricted information beyond H? | No. The binary label is exactly `I{H>0.9}`; exact H completely determines it. |
| Does the association survive past-only fitting? | No. It reverses sign under independently refitted prefixes in S13X and S13Y. |
| Did we reproduce the MLP early-warning claim? | No. The exact layout remains unresolved, the paper-like first quarter is not supported by the strict estimator, and the source-prefix evidence is negative. No valid E01 result supports prospective prediction. |
| Did we reproduce intervention efficacy? | No. Strict scoring suppressed 1,089/1,090 opportunities; the adaptive retrospective pilot gave paper-directed ordering in only 1/8 comparisons. |
| Did we reproduce exact Figures 2–6 or Table 1? | No. Some retrospective directions and spike patterns resemble parts of Figures 2–4, but the aggregate trend, label fingerprints, prediction, intervention, and exact table semantics do not jointly match. |
| Do these failures prove the paper wrong? | No. They constrain the tested reconstructions. Under the working assumption that the paper is correct, they show that material implementation details remain missing. |

## Most likely missing or unresolved details

Ranked by leverage, the remaining gaps are:

1. **Replication-label scope and reference.** Molecular adjacent similarity, historical post-fission non-drift, dominant compotype, and retrospective cluster membership are not equivalent. This was the largest observed directional switch.
2. **Completed-fit versus time-local fitting.** Public code uses the complete supplied trajectory. The paper discusses molecular-time trajectories and early prediction but does not state whether partitions and Gaussian parameters are recomputed locally, fitted once globally, or leaked from the completed run.
3. **Exact GARD exposure and clock.** Candidate 2 and candidate 3 both match paper-visible time scales but differ in daughter selection and exposure. No downstream result can identify which, if either, is the author implementation.
4. **Metric identity.** Public code names `local_phi_r` as integrated and `synergy + downward causation` as emergence. The paper's extracted equations were initially unavailable, and no released GARD script maps its plotted scalar to a pinned implementation.
5. **Preprocessing and partition construction.** The paper explicitly states relative composition, CLR, and removal of the last component. Public GRN code adds z-scoring, global-signal regression, lag-one residualization, low-variance filtering, Fiedler partitioning, and in PhiRL regularization.
6. **Level versus difference and time alignment.** Results discuss levels; the Figure 3 caption discusses changes. Same-state, next-state, incoming, outgoing, and averaged labels can change direction.
7. **Prediction tensor and training protocol.** Sequence shape, padding, scaling, target definition, class weighting, split reuse, architecture, and whether the first-quarter input was computed using the completed trajectory are not fully specified.
8. **Intervention scoring.** It is unclear whether each candidate action is scored by a prefix refit, a model fitted to the completed control, a replacement within the completed control, or another state/window construction. Tie handling and stochastic pairing are also unresolved.
9. **Statistical and reporting conventions.** Spike scope, variable-length alignment, Mann–Whitney unit, Ljung–Box lags, Table 1 dispersion, consistency formula, and time-to-first units remain partly unresolved.
10. **Exact code and RNG semantics.** Public historical MATLAB behavior, modern Python source behavior, and the unavailable GARD-paper code are separate lineages. Exact cross-language trajectories are not expected without the original code and seeds.

## What is solid enough to carry forward

The following are reusable E01 assets regardless of which continuation is selected:

- two tested GARD engines and an explicit simulator-contract system;
- two paper-time-compatible confirmed candidate pipelines, with an honest nonidentifiability label;
- domain-separated seed and replay contracts;
- validated compositional transforms and lossless status-bearing schemas;
- validated source wrappers for IIGR and PhiRL, including safe lattice handling;
- a strict PhiID branch valid at 512 or more effective observations;
- proof that the attempted 32–256 fixed-window reconstructions are not eligible;
- proof that public completed-fit values can depend strongly on future data;
- replicated negative prospective-prefix evidence;
- a clean retrospective association for one fixed branch, together with proof of its exact H-label coupling;
- negative strict and exploratory intervention evidence;
- complete claim, ambiguity, discrepancy, failure, source, seed, and artifact ledgers.

These assets are enough to close E01 honestly or to mount one bounded continuation. They are not enough to treat Phi-r as an established prospective or causal control variable.

## Three decision-ready continuation options

No option below has been executed. Selecting one would require a new explicit human authorization and a separately versioned plan amendment.

### Option 1 — Conservative closeout of E01

**When to choose it:** Choose this when the priority is to finish E01 rigorously and move to E02+ without spending more compute trying to infer unpublished details.

**Scope**

- Complete S14–S18 as an evidence-synthesis and formal-disposition sequence using only frozen S01–S13Y evidence.
- Generate no new GARD trajectory, source fit, prediction model, or intervention.
- Preserve the S13Y result as `LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE` and the S13RRR result as held-out non-support under the historical post-fission label.

**Methods and frozen evidence**

- **S14:** Reconstruct descriptive trend/spike panels from already frozen S13Y full-fit values, while showing that punctuations resemble the paper but aggregate trends do not. Keep S12/S12C/S13RRR comparators visible.
- **S15:** Finalize association tables by label and temporal mode. Put the S13Y molecular-H result beside historical-label and past-only results. Make exact H determinism and the smooth H/L2 control central rather than footnotes.
- **S16:** Formally classify prediction and spike-timing claims as `UNDERDETERMINED` or `NOT_EVALUATED`, with the negative prefix evidence recorded as a constraint. Do not train an MLP merely to fill the slot.
- **S17:** Formally classify paper intervention claims as `UNDERDETERMINED` for exact author semantics and `NOT_SUPPORTED_WITHIN_STRICT_SCOPE` for the strict and S13X pilot estimands. Do not promote near-universal action suppression to an outcome null.
- **S18:** Produce the final forensic claim matrix, reconstructed/failed figure panels, discrepancy table, caveat ledger, and E02 handoff.

**Treatment of prediction and intervention claims**

- Prediction is not claimed. Full-fit values are explicitly ineligible for prospective prediction, and prefix results point negative.
- Intervention efficacy is not claimed. The exact paper scorer remains unknown, strict actions were almost entirely nonseparable, and the small retrospective pilot did not reproduce ordering.

**Validation and compute implications**

- Expected new compute: approximately 5–15 CPU-hours, primarily table generation, plotting, cross-file checks, and independent regeneration samples; no GPU is needed.
- Validate claim-to-evidence traceability for all 59 claims, exact preservation of S01–S13Y hashes, figure/table reproducibility, status-vocabulary consistency, and E02 input manifests.

**Benefits**

- Fastest and strongest procedural closeout.
- Does not add another post-result branch.
- Gives E02 a clean record of what is validated, negative, and unresolved.

**Risks**

- Leaves the paper's prediction and intervention claims unreconstructed.
- May feel incomplete if the objective is to find a paper-like end-to-end pipeline rather than adjudicate available evidence.

**Evidentiary limits**

- The final E01 verdict remains partial forensic replication, not exact replication.
- Retrospective association is reported only as label-coupled resemblance.

**Stop or escalation conditions**

- Stop if any prior artifact hash changes or if a figure cannot be regenerated from frozen tables.
- Escalate rather than invent a value whenever a claim cannot be mapped to an admissible estimand.

**Completion criteria**

- Every S14–S18 deliverable exists.
- Every one of the 59 claims has a source, evidence link, and final status.
- Prediction, intervention, prospective, retrospective, and author-identity statements are separately classified.
- E01 is closed at human review with no active scientific branch.

**Expected E01 artifacts**

- S14–S18 canonical reports;
- final 59-claim matrix and discrepancy crosswalk;
- paper-versus-reconstruction figures and Table 1 audit;
- definitive caveat/failure ledger;
- immutable E01 bundle manifest and E02 handoff manifest.

**Resulting E02+ handoff**

- E02 receives both confirmed time-base candidates, the strict estimator boundary, source wrappers, historical and molecular labels, continuous H/L2 baselines, and all negative evidence.
- E02 begins with the question “Is PhiRL emergence useful beyond ordinary attractor stability?” rather than assuming that E01 established a causal knob.

### Option 2 — One bounded paper-directed reconstruction through S18

**When to choose it:** Choose this when the priority is to make one last source- and paper-directed attempt at the missing prediction and intervention pieces under the working assumption that the reported pipeline is coherent.

**Scope**

- Freeze one final E01 protocol before outcomes and run S14–S18 sequentially.
- Use only confirmed candidates 2 and 3; candidate 1 remains excluded.
- Use the exact S13Y primary scalar and label for the paper-facing retrospective lane. Preserve the paper's level-versus-change ambiguity as two named analyses, never as outcome-based model selection.
- Introduce no estimator, threshold, preprocessing, partition, or simulator search.

**Methods and frozen evidence**

- **S14:** Use S13Y data to produce exact, prespecified trend, positive/negative 3-sigma, MAD, width, spacing, and temporal-dependence panels under the completed-fit retrospective label.
- **S15:** Reproduce paper-like Spearman, positive/significant counts, replicator-minus-drift summaries, Mann–Whitney/Fisher diagnostics, and block-aware controls for both candidates. Report S13Y's circularity result as a hard interpretation boundary.
- **S16:** Reconstruct the paper-like 25%/75% MLP in two locked modes: a completed-fit retrospective-input mode and a true past-only prefix mode. Use run-grouped train/test splits, the paper's composition-change/raw-count/flux/dummy baselines, plus exact H and ordinary composition change. The retrospective mode may test resemblance only; the prefix mode is the only prospective test.
- **S17:** Before any intervention outcome, freeze the three already source-grounded scoring semantics: prefix refit, frozen completed-control model, and replacement in completed control. Run a bounded MRR under both candidates with shared matrices and exact action logs. Retrospective scorers can explain a possible author implementation but cannot establish online causal control; only prefix scoring can do that.
- **S18:** Produce separate paper-facing and causality-facing verdicts so a favorable retrospective scorer cannot rescue a failed prospective one.

**Treatment of prediction and intervention claims**

- A completed-fit MLP result is labelled retrospective and leakage-sensitive, never early warning.
- A prediction claim requires both candidates to show incremental held-out value beyond exact H/composition baselines using prefix-only features.
- An intervention claim requires bidirectional max/control/min movement under a prospectively scoreable rule, exact replay, adequate action frequency, and no reliance on completed-control future data.

**Validation and compute implications**

- Preflight against the remaining E01 envelope: after S13Y, approximately 109 CPU-hours remain under the 250-hour ceiling. The full protocol must be benchmarked before new computation and must fit inside that remainder without reducing a declared scope after outcomes.
- Expected range: roughly 60–105 CPU-hours depending on candidate-action scoring. GPU use is optional and must be cross-checked against CPU float64.
- Freeze MLP tensors, splits, architecture, scaling, scoring semantics, action ties, seeds, and all claim gates before access.

**Benefits**

- Best chance of identifying a coherent end-to-end paper-like pipeline without contacting the authors.
- Directly addresses the two large remaining paper claims rather than only marking them unresolved.
- Separates retrospective resemblance from prospective validity within the same protocol.

**Risks**

- Continues a long E01 branch after substantial negative evidence and several human overrides.
- Three intervention semantics increase multiplicity and may remain nonidentifiable.
- Completed-fit analyses can look persuasive while still depending on future data and exact label coupling.
- The remaining compute envelope may be insufficient for a fully powered intervention scale-up.

**Evidentiary limits**

- No positive result identifies the author implementation.
- A positive retrospective MLP or retrospective intervention scorer is explanatory forensic evidence only.
- Exact Figure 5/6 or Table 1 status may remain underdetermined even after the bounded campaign.

**Stop or escalation conditions**

- Stop before outcome access if the benchmark projects above the E01 CPU/storage ceiling.
- Stop an intervention branch if replay, source equivalence, action cardinality, or score validity fails; do not repair after outcomes.
- If only a retrospective mode succeeds, complete S18 with `RETROSPECTIVE_TEMPORAL_FITTING_DEPENDENCE` rather than adding another estimator or scorer.
- Any surprising positive prospective result returns for human review; it does not trigger an automatic scale-up.

**Completion criteria**

- S14 and S15 have complete paper-like and block-aware outputs for both candidates.
- S16 assigns separate retrospective and prospective prediction statuses.
- S17 assigns separate statuses to each frozen scoring semantics and to causal-control eligibility.
- S18 closes all 59 claims or marks genuinely unavailable claims underdetermined with explicit reasons.

**Expected E01 artifacts**

- S14–S18 reports and updated claim matrix;
- retrospective and prefix prediction tensors/manifests, model results, calibration and leakage audits;
- intervention candidate-score/action/pairing/replay tables and bounded pilot outcomes;
- reconstructed Figures 2–6 and Table 1 comparison;
- full source/specification/multiplicity/provenance manifest.

**Resulting E02+ handoff**

- E02 receives a maximally paper-directed reconstruction with each retrospective and prospective branch labelled.
- If prospective and intervention branches remain negative, E02 starts by testing alternative reaction coordinates and pipeline artifacts.
- If a prospective branch is unexpectedly positive, E02 receives it as a candidate requiring independent adversarial confirmation, not as established truth.

### Option 3 — Dual-track E01 closeout and adversarial bridge to E02

**When to choose it:** Choose this when the priority is to preserve the paper-facing reconstruction while using the remaining E01 work to determine whether PhiRL emergence adds anything scientifically useful beyond ordinary compositional stability.

**Scope**

- Run S14–S18 in two locked lanes.
- **Lane A, forensic:** finalize the S13Y completed-fit paper resemblance and the exact claim-by-claim nonreplication/underdetermination record.
- **Lane B, mechanistic bridge:** test prospective incremental prediction and bounded counterfactual control against H, L2 composition change, recurrence/attractor distance, random action, and displacement-matched controls.
- Do not search for another Phi estimator or threshold. Candidate 2 and candidate 3 remain separate and equally required.

**Methods and frozen evidence**

- **S14:** Produce the same frozen retrospective trend/spike description as Option 1, plus matched H/L2 time-series panels.
- **S15:** Estimate whether completed-fit and prefix emergence associations persist after continuous H, L2 change, and trajectory effects. Keep the exact deterministic-label result as the primary circularity finding.
- **S16:** Use independently refitted prefix emergence only for prospective models. Compare exact H, H plus ordinary composition/flux features, PhiRL alone, and H/composition plus PhiRL under nested run-grouped validation. Evaluate calibration, AUROC/AUPRC, and first-arrival lead time, not accuracy alone.
- **S17:** Use a small, preregistered paired counterfactual benchmark rather than claim exact paper intervention replication. Compare past-only PhiRL-directed actions with H/attractor-directed, random, no-op, and displacement-matched actions at shared states. Require score separability, matched intervention magnitude, common-risk streams, and full action logging.
- **S18:** Publish two matrices: paper-claim reproduction status and mechanism/adjudication status. The first does not borrow success from the second.

**Treatment of prediction and intervention claims**

- The literal paper MLP and Table 1 claims may remain underdetermined in Lane A.
- Lane B asks the stronger E02-ready questions: does prefix PhiRL add held-out value beyond H, and do PhiRL-directed actions outperform simpler matched actions?
- A negative result closes PhiRL as an independent E01 predictor/control candidate without denying the paper's report under an unavailable implementation.

**Validation and compute implications**

- Expected range: approximately 40–90 CPU-hours, with no new baseline scale-up. Use existing S13Y baselines for S14–S16 and generate only the bounded paired counterfactual trajectories or rollouts required by S17.
- Perform a preflight against the remaining E01 ceiling. CPU float64 remains authoritative; any GPU acceleration requires exact representative cross-checks.
- Freeze all model formulas, feature sets, splits, action controls, horizons, multiplicity handling, and stop rules before outcomes.

**Benefits**

- Best separates respect for the paper from the need for an independently useful mechanism.
- Directly answers the circularity problem exposed by S13Y.
- Produces the most informative handoff for E02's planned `INDEPENDENT_CONTROL_KNOB`, `USEFUL_PROXY`, `PIPELINE_ARTIFACT`, or `MODEL_SPECIFIC` adjudication.

**Risks**

- Blurs the original E01/E02 boundary by pulling a limited subset of E02-style controls into E01.
- Does not maximize the chance of an exact Figure 5 or Table 1 resemblance.
- Counterfactual action scoring may again be sparse or computationally expensive.

**Evidentiary limits**

- Lane A remains retrospective and source-informed.
- Lane B evaluates scientific utility under our confirmed reconstructions, not author identity.
- A small paired control benchmark cannot establish broad causal generality; it can only determine whether larger E02 work is warranted.

**Stop or escalation conditions**

- Stop if the preflight exceeds the remaining E01 compute ceiling or if either candidate fails replay/source/coverage checks.
- Stop the independent-Phi claim if PhiRL adds no held-out value beyond H/composition in both candidates.
- Stop the control-knob claim if PhiRL actions fail to beat random and matched H/attractor actions or remain mostly nonseparable.
- If both candidates show unexpected incremental prediction and matched-control intervention benefit, freeze the result and escalate to human review; confirmation belongs in E02, not another E01 extension.

**Completion criteria**

- S14–S18 are complete with separate forensic and mechanistic outputs.
- Every paper claim has a final reproduction status.
- PhiRL's incremental predictive and paired-control status is explicit for both candidates.
- E01 closes without treating a retrospective label-coupled effect as early warning or causal control.

**Expected E01 artifacts**

- all Option-1 closeout artifacts;
- continuous-H/L2/incremental-association tables;
- nested prospective prediction and calibration results;
- paired action-control score, outcome, cost, separability, and replay tables;
- dual claim/mechanism matrix and E02 decision-input manifest.

**Resulting E02+ handoff**

- E02 receives a pre-adjudicated candidate set: PhiRL emergence, H similarity, composition change, recurrence/attractor distance, and matched intervention controls.
- If PhiRL adds value, E02 tests whether it is an `INDEPENDENT_CONTROL_KNOB` or `USEFUL_PROXY` on new regimes.
- If it does not, E02 can classify it as a likely `PIPELINE_ARTIFACT` or model-specific proxy while retaining the validated GARD infrastructure and shifting emphasis to the strongest direct reaction coordinate.

## Compact choice guide

| Choice | Main objective | New scientific computation | Prediction/intervention disposition | E02 readiness |
| --- | --- | ---: | --- | --- |
| 1 | Close E01 conservatively | Minimal | Formally underdetermined/not supported in tested scope | Fastest, cleanest handoff |
| 2 | Make one last paper-directed reconstruction | Moderate to high | Literal retrospective and prospective branches plus bounded scoring-semantics audit | Richest paper-facing handoff, highest ambiguity risk |
| 3 | Preserve the paper-facing record and test independent utility | Moderate | Prospective incremental prediction and matched counterfactual controls | Strongest mechanistic handoff |

## Overall interpretation

Our work supports neither a blanket dismissal nor an exact replication claim. It supports a more precise statement:

> The public source family and defensible GARD reconstructions can generate punctuated causal-emergence trajectories and, under a particular molecular adjacent-similarity label, a reproducible positive completed-fit association. That association is retrospective, exactly coupled to the H coordinate defining the binary label, does not add unrestricted information beyond H, and reverses under past-only refitting. The paper's prediction and causal-intervention claims remain unreproduced or underdetermined because the exact implementation is unavailable and the tested prospective/control reconstructions were negative.

Under the human-directed assumption that the authors' claims are correct, the remaining task is to decide whether to close E01 with that honest boundary, make one bounded final paper-directed reconstruction, or use the remaining steps as a dual-track bridge that tests whether the signal is scientifically useful beyond H. No choice has been made or executed in this documentation step.

## Provenance sources

This document was derived from the workspace plans, uploaded paper extraction, S01 claim/source reconciliation, S02 ambiguity/discrepancy ledgers, registry v0.3.0, source-clue and paper-fingerprint ledgers, and canonical reports plus machine-readable decisions from S10–S13Y. Exact input and document hashes are recorded in the provenance sidecar. The separate gene-regulatory-network replication context is human supplied and is not represented as an independently verified E01 result.
