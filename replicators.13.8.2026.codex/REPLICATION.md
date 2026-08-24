# Clean-room replication contract

## Scope

This implementation targets only the proposed plastic-heredity discovery described in `PRE_PRINT_PAPER_DRAFT.md` and visualized by embedded Figures 3-5 in `PRE_PRINT_DISTILL.PUB.html`. It does not implement PhiID, the original paper-facing reconstruction, attractor transfer, or molecular interventions.

No code or generated artifact from the unavailable `eidosoma/groups/42` experiment branch is included. The implementation in `plastic_heredity/` was written independently from the supplied scientific description. The embedded plots are comparison targets, never simulator or model inputs.

The cited historical public GARD commit `86dff6320d5ae91b4e831471079ff46749b14df9` was consulted as a published specification for the basal kinetic equation and constants. No MATLAB implementation was copied or vendored; the Python engine, stochastic contracts, analysis, and tests here are independent.

## Directly specified

- `beta_ij = exp(-4 + 4 Z_ij)`, with 100 molecule types.
- Initial mass 40, fission mass 80, 100 fissions, and at most 1,000 growth updates.
- Relative reservoir abundance `1/100`; historical GARD basal join/leave constants `1e-2` and `1e-4`.
- Strict boundary inheritance is parent-to-selected-daughter cosine similarity `H > 0.9`.
- `JOINT_BREAK_RUN3` is a break followed by a newly certified run of three within 12 future fissions. An uninterrupted pre-existing run does not qualify.
- `JOINT_BREAK_RUN3` does not require the episode daughters to be mutually similar, distinct from the pre-break composition, recurrent, or persistent beyond three fissions. It is a break-and-renewal endpoint, not a registered regime-transition endpoint.
- The final predictor combines 195 state/graph coordinates reduced to 12 PCA components with nine stated direct history/phase variables, then uses L2 logistic regression with `C=0.1`.
- Confirmation uses 40 shared matrices, both candidates, generations 20/35/50/65/80, and 64 futures split prospectively into halves of 32.
- Inference is candidate-separated and treats the catalytic matrix as the resampling/permutation unit.

## Necessarily inferred and frozen

The supplied files do not expose the executable candidate-02/03 contracts, the 195 coordinate definitions, the L53 development cohort size/seeds, or every conditional process definition. Exact numerical reproduction is therefore impossible from the folder alone. This clean-room run freezes the following before confirmation:

| Item | Candidate 02 | Candidate 03 |
|---|---|---|
| Poisson exposure per update | 0.10 | 0.125 |
| Overshoot | apply update, then uniformly trim the whole assembly | admit proposed joiners uniformly only to remaining capacity |
| Fission | exactly 40 molecules sampled without replacement | independent binomial(0.5) partition |
| Continued daughter | first | second |

The 195-dimensional target-blind representation is exactly 15 molecule-level state/network profiles summarized by 13 symmetric statistics. This makes it invariant to simultaneous molecule-label permutation; a unit test enforces that property. The beta-only control applies the same map to a uniform pseudo-composition.

Conditional process summaries are:

- `resume_2`: any two inherited boundaries after the first break;
- `episode_3`: any three inherited boundaries after the first break;
- `persist_5`: any five inherited boundaries after the first break;
- `old_return`: the daughter at the first post-break resumption returns to `H > 0.9` of the pre-break parent;
- `positive_gain`: similarity to that old anchor increases at the first inherited boundary after the break;
- `repeat_return`: at least two old-neighbourhood observations at or after that resumption, conditional on an old return.

Development uses 40 new shared matrices, the same five landmarks, and 32 branches per state. Its seed domain is disjoint from confirmation. Every matrix, main trajectory, state future, bootstrap, and permutation seed is derived by SHA-256 from a fixed master seed and a domain label.

An extinct future is absorbing: an event certified before extinction remains positive, while a not-yet-certified event is negative because no later fissions can occur. Extinction itself is not called a fission or an inheritance break. `completed_horizon` is retained in the branch table. Because the stated cohorts contain complete 100-fission trajectories, landmark-generating main paths use deterministic, domain-separated retries after extinction; futures are never retried.

## Evidence gates

The discovery is considered qualitatively replicated only if, separately in both candidates:

1. independent branch halves show positive state-probability reliability;
2. the frozen full model has higher within-matrix rank than direct history;
3. the frozen full model improves branch log loss over direct history in both branch directions;
4. exact regeneration is byte-identical in the full profile.

The supplied paper's numerical ranges are evaluated only after simulation/model fitting and are written to `reported_comparison.csv`. They do not tune any parameter.

## Post-hoc episode-coherence audit

A later review identified an interpretation boundary rather than an implementation error: adjacent parent-to-daughter inheritance is not transitive, so three inherited fissions do not necessarily occupy one compositional neighbourhood. The standalone `episode_coherence` workflow leaves the registered target and every source bundle unchanged.

It verifies checksums and reconstructs the scaled5, MECHCONF, and MECHCONF2 confirmation states from their archived manifests and seeds. It then regenerates all 145,516 positive branches from 384,000 archived F12 futures. All states and target arrays match, all discrete process outcomes replay exactly, and continuous process values agree within `1.11e-16`.

For the first qualifying episode, the audit records all pairwise daughter similarities, each daughter's similarity to the first pre-break parent, uninterrupted run length, same-run persistence to five, and a later break followed by a second run of three observable within F12. This last quantity is not compositional recurrence. Continuous geometry is primary. Coherence cutoffs `>0.90`, `>0.95`, and `>0.975` and maximum old-anchor cutoffs `<=0.90`, `<=0.85`, and `<=0.80` are post-hoc sensitivity views. Persistence limited by the horizon is right-censored. Candidates, cohorts, and branch halves remain separate; 95% intervals use 4,096 whole-matrix bootstraps over all 200 matrices, including matrices with no qualifying episodes.

Across the six cohort/candidate comparisons, mean minimum pairwise daughter similarity is 0.681–0.704 and only 4.5–6.5% of episodes place every daughter pair above `H=0.9`. In contrast, 93.2–94.9% keep every daughter at or below `H=0.9` from the old anchor. Among episodes with resolved five-fission status, 75.9–78.9% persist in the same inherited run; a later break followed by a second renewal is observed in 10.2–11.7% within F12.

These results support break followed by renewed short-run inheritance, generally away from the old anchor. They do not establish one coherent new hereditary regime. Because the audit criteria were designed after the target result, even favourable geometry could only motivate a new prospective endpoint and untouched cohort.

## Prospective mechanistic-ablation extension

Review of the unavailable original-paper implementation identified that its full-versus-history comparison could mix current state/network information with duplicated variables, growth-clock history, and L2 penalty geometry. The original clean-room result remains an algorithm comparison; the separate `MECHCONF` workflow narrows attribution prospectively.

The retained 200-matrix scaled development cohort is the only fitting data. Main trajectories are deterministically reconstructed to capture the immediately preceding growth-step count and cumulative growth-step count. No development future is resimulated: reconstructed 195-coordinate state arrays, nine-variable history arrays, beta-only arrays, and all 64,000 retained target rows must match exactly before fitting.

The registered blocks are:

- `H8`: the unique direct-history variables, with the exact `fissions_since_break` duplicate removed;
- `H10`: `H8` plus normalized prior-cycle and cumulative growth clocks;
- `S`: mass-free composition and presence summaries with no beta input;
- `B`: the registered state-free beta representation;
- `I`: mass-free beta-conditioned state summaries, residualized against `H10 + S + B` using development states only;
- `D`: copies of normalized generation and mass used only as a negative control.

Constant and exact affine-duplicate columns are removed using development data. Added blocks receive separate development-only scaling and at most 12 PCA components. In the primary nested models, the common `H10` block and intercept are unpenalized and only added components receive the registered `C=0.1` L2 penalty. A same-penalty `H10`/`H10+D` pair directly measures duplicated-direction ridge behavior.

Primary prospective contrasts are `H10+S` versus `H10`, `H10+S+B` versus `H10+S`, and `H10+S+B+I` versus `H10+S+B`. Each must pass in both candidates and both preassigned branch-half directions: positive paired log-loss gain, a positive lower 95% matrix-bootstrap bound, and a Holm-adjusted paired whole-matrix randomization value below 0.05. The Holm family contains all 12 primary tests. Existing scaled confirmation data are not reused for this post-review claim.

## Prospective beta-completeness correction

A later review identified that the v1 `B` block began with all 195 uniform-pseudo-composition coordinates but reduced 87 distinct beta-derived directions to 12 unsupervised PCs, retaining about 76% of their standardized variance. The v1 `I` block was consequently residualized against an incomplete beta basis. The v1 MECHCONF result remains a valid test of its registered representations, but it cannot establish that beta is generally uninformative or that the incremental signal necessarily requires a beta–state interaction.

The versioned `mechanistic_v2` workflow corrects this without modifying any earlier bundle:

1. `prepare` reconstructs the unchanged 200-matrix scaled development cohort and all 64,000 targets; assigns every coordinate explicit state, beta, history, clock, mass, and phase provenance; fits the complete frozen suite; and seals code, inputs, models, penalties, gates, and a new seed before any result is scored.
2. `diagnose` applies that already sealed suite to old MECHCONF only as a labelled post-hoc diagnostic. It cannot alter the registration or support a prospective claim.
3. `confirm` uses 200 new matrices, both candidates, five landmarks, 64 F12 futures per state, and the disjoint `MECHCONF2` seed domain. All 128,000 futures are generated twice and required to have identical batch digests.

The provenance-selected static-beta block contains every eligible legacy invariant coordinate plus a fixed threshold-free panel: raw/log-beta distributions; row/column-strength distributions and correlation; reciprocity; normalized asymmetry; all 100 normalized singular values; stable rank; spectral entropy; and row/column strength entropy and concentration. It uses no thresholds, outcome-selected features, or PCA. After development-only constant and exact affine-duplicate removal, 240 beta directions remain. The mass-free state and beta-conditioned blocks retain 11/12 and 64 directions, respectively.

For backward-compatible simulator metadata, the shared `ExperimentConfig` serialized in the result manifest still contains its unused legacy `pca_components=12` field. The authoritative v2 model registration, model archive, development audit, and result report all record `uses_pca=false`; no v2 added block calls PCA.

Models are nested sequentially as `H10`, `H10+S`, `H10+S+B`, and `H10+S+B+I`. Each addition is a frozen offset-ridge correction, so a zero added coefficient reproduces the preceding prediction exactly. Added-block penalties are selected from `1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100` by deterministic five-fold whole-matrix development cross-validation; ties within `1e-12` choose the stronger penalty. The primary tests and all-candidate/all-half gate are the same 12-test family used for the v1 attribution.

Registration `7c6acd1e3bac96dae7931c48bff39deedfc6550344dfea3753949956f71701bd` was sealed before the post-hoc diagnostic and MECHCONF2 generation. No v2 primary contrast passed. The state gains were small and inconsistent with the full gate; static beta did not pass; and the interaction addition was harmful in candidate 02 and negative but inconclusive in candidate 03. This supersedes the stronger mechanistic interpretation: the original composite algorithm comparison remains supported, but composition beyond clocks and a necessary beta-conditioned interaction are not robustly established.

## Prospective inheritance-dependence extension

Review of the unavailable L44 implementation identified that its IID probability was fitted on every post-break suffix symbol but scored only on transition destinations. First symbols and singleton suffixes could therefore change the fitted IID baseline without contributing to either evaluated loss. The old `0.015–0.022` bits-per-transition statement is withdrawn rather than retroactively repaired.

The separate memory workflow uses three stages:

1. `diagnose` deterministically regenerates the existing scaled confirmation futures, validates all 128,000 retained branch rows and the original digest, and compares the mismatched and corrected IID baselines. In this distinct L54 clean-room cohort, the mismatch inflated the Markov gain by 0.000550/0.000391 bits per transition in candidates 02/03. This is a retrospective sensitivity result, not an L44 reproduction or confirmation.
2. `prepare` seals the complete analysis and source hashes before any new matrix is generated. The registration ID is `0a100eb3d626f3fdb92f5b4f84f1404b095fc1d21b1dfe6b3a83d2adf0e78f1f`.
3. `confirm` uses 200 new shared catalytic matrices, both candidates, five landmarks, 64 futures per state, and a 32-fission horizon. All 128,000 futures are exactly regenerated at the variable-length sequence level.

For each future, the analysis finds the first strict inheritance break and retains the observed symbols strictly after it. No-break, empty-suffix, and singleton-suffix futures contribute no transitions. All three models are fitted and scored on the identical remaining destination symbols:

- IID: one destination probability;
- Markov: destination probability conditional on the preceding symbol;
- duration-aware semi-Markov: destination probability conditional on the preceding symbol and its past-only run length in fixed bins `1, 2, 3, 4, 5+`.

Every Bernoulli cell uses a Beta(1,1) posterior mean. Models are cross-fitted by whole catalytic matrix in fixed even-to-odd and odd-to-even directions and separately by simulator candidate. The primary estimand is held-out, transition-weighted bits per transition; the equal-state macro-average is descriptive. Matrix bootstraps and paired whole-matrix sign randomizations each use 4,096 repetitions. Holm correction covers two contrasts by two candidates. A contrast must have positive pooled gain, a positive lower 95% matrix-bootstrap bound, Holm-adjusted `p<0.05`, and positive gain in both cross-fit directions, in both candidates.

Both registered contrasts passed. Markov versus IID gained 0.046953 bits/transition (95% CI 0.039503–0.056486) in candidate 02 and 0.033936 (0.028894–0.040376) in candidate 03. Duration-aware versus Markov gained 0.010770 (0.008657–0.013493) and 0.009984 (0.007921–0.012333), respectively; all Holm-adjusted values were 0.000976. These results establish out-of-matrix statistical first-order and duration-dependent prediction under the registered representation. They do not establish biological memory, molecular information storage, error correction, or causality; latent matrix/state heterogeneity remains a possible contributor.

This extension does not refit, rescore, or otherwise alter the L54 process-risk predictor. Its statistical sequence claims are independent of both mechanistic-ablation suites.

## Prospective strict-episode occurrence and prediction extension

The standalone `regime_confirmation` campaign prospectively replaced the post-hoc “new regime” interpretation with an operational F32 endpoint. After the first strict inheritance break, the primary requires eight consecutive inherited fissions, strict `H>0.9` for all 28 pairs among their daughters, and inclusive `H<=0.85` for every daughter relative to the pre-break parent. First-five all-pairs and eight-daughter centroid coherence are secondary and cannot rescue the primary. Development and confirmation each use 200 disjoint matrices, both candidates, five landmarks, 128 futures per state, fixed 64-future halves, and complete replay.

Strict occurrence passed in all four candidate/half cells at 1.81–2.11%. The registered h10-plus-state predictor did not: only candidate-03 half A passed its full interval/multiplicity gate. This supports occurrence of a rare distinct coherent eight-fission hereditary episode under the operational definition, not its robust state-added predictability, recurrence, basin identity, perturbation recovery, or causal control.

The next prediction campaign is isolated in `regime_prediction` and never modifies that seal:

1. `diagnose` verifies the old bundle, decomposes break/eight-run/strict rates, performs deterministic 80-of-200-matrix sensitivity analysis, and reconstructs all sealed feature blocks. It is post-hoc and cannot select a model.
2. `register-design` seals source hashes, endpoint/margin equivalence, feature provenance, six model families, disjoint seed domains, event-power gates, the pilot stop rule, and the sole confirmation test.
3. `pilot` uses 80 new matrices, both candidates, five landmarks, 128 F32 futures per state, and complete replay. Every eligible eight-run window is retained. A common model family must improve h10 in both candidates and both fixed halves, satisfy event gates, survive the one-standard-error simplicity rule, and be selected in at least 75% of 4,096 paired whole-matrix bootstraps. Each draw uses the same catalytic-matrix indices across both candidates and every family. Failure stops the campaign.
4. `confirm` is authorized only by a checksum-valid passing pilot. It applies the frozen candidate-specific parameters of the one selected family to 200 new matrices and 256,000 untouched futures. All four candidate/half tests require positive log-loss gain, a positive 95% whole-matrix bootstrap lower bound, and Holm-adjusted paired matrix-randomization `p<0.05`.

Generation and exact replay write separate atomic per-state checkpoints bound to the source hashes, complete experiment contract, branch count, and ordered state-ID digest. Reissuing an interrupted command resumes only matching checkpoints and refuses a changed contract. The read-only `status` command reports campaign phase and completed-state counts. Checkpoints are retained after sealing and are never silently deleted.

The delivered `regime_prediction_smoke` bundle exercises cohort construction, every feature block, rich endpoint/window output, serialization, checksum generation, and replay on eight deliberately non-scientific futures. Its two replay digests are identical with maximum continuous error zero. It validates plumbing only and cannot be used as pilot evidence.

The added local-dynamics block analytically evaluates the simulator's expected joining/leaving drift, event variance, tangent composition drift, drift-to-noise ratio, entropy/concentration derivatives, tangent-Jacobian stability, and recent post-fission velocity. All summaries are invariant to simultaneous molecule relabelling and carry explicit state/beta/history provenance. The fixed model menu contains direct offset ridge, a three-stage break/run8/geometry hurdle, hierarchical beta propensity plus state/dynamics, local dynamics, leakage-safe out-of-fold first-five/centroid stacking, and one bounded out-of-fold-calibrated histogram-gradient model. The common h10 baseline is unpenalized; added linear blocks are ridge-penalized without PCA after development-only constant and affine-duplicate removal.

First-five, centroid, continuous margins, hurdle stages, and post-break prediction are explanatory secondary analyses. They cannot rescue strict prediction. Matched-future molecular control remains outside this campaign and is deferred until a predictor passes untouched confirmation.
