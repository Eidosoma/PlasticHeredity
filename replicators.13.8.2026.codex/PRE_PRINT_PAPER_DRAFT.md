# From Retrospective PhiID Resemblance to a Past-Observable Plastic-Heredity Risk Coordinate in Two Reconstructed GARD Simulators

**Pre-print draft for human review**  
**Submission metadata status:** The scientific synthesis has been reviewed, but the author list, affiliations, contributions, acknowledgements, funding, competing interests, and corresponding-author details require completion and approval by the submitting investigators.

**Evidence cutoff:** 15 August 2026.

## Abstract

The Graded Autocatalysis Replication Domain (GARD) model provides a computational framework for studying compositional inheritance in mutually catalytic molecular assemblies. A recent preprint reported that a PhiID-derived causal-emergence quantity rises before the first self-replicator, predicts later replication from the first quarter of a trajectory, and functions as an intervention target. We performed a branch-explicit forensic reconstruction of those claims and then used the resulting evidence to ask a broader question: whether any past-observable property predicts future self-maintaining organization in reconstructed GARD dynamics.

We independently implemented and validated historical and paper-described GARD branches, audited self-replicator labels and compositional preprocessing, reproduced public PhiRL/IIGR source semantics, and reconstructed Figures 2–6 and Table 1 under two frozen simulator candidates. Paper-like punctuated information excursions and retrospective association directions were observed. However, aggregate trends differed, the closest associations were completed-fit and label-coupled, first-quarter prediction did not exceed appropriate controls, and literal Phi-directed interventions did not reproduce the reported max/control/min ordering. The immutable paper-facing verdict was therefore partial directional retrospective reconstruction, with prospective prediction and causal control unsupported within the tested scope.

We next estimated state-dependent empirical committors by restoring simulator states and launching independent future branches. A reliable committor into a completed-run attractor and an untouched-confirmed eight-step propagator coordinate were found, but independent lineages under the same catalytic matrix did not recover one transferable basin or basin family. We therefore replaced the destination with a prospective process: an inheritance break followed, within twelve fissions, by a new three-fission episode of strict parent–daughter compositional inheritance. This process distinguished plastic renewal from exact return to an old composition, which was extremely rare.

Finally, a frozen target-blind representation of current composition, catalytic-network-conditioned state, growth/fission phase, and recent heredity history was tested on 40 new catalytic matrices, 80 trajectories, 400 post-fission states, and 25,600 independent branch futures plus exact regeneration. In both simulator candidates, the coordinate ranked independently measured process probabilities overall (Spearman 0.895–0.918) and within matrices (0.550–0.697), improved proper scores beyond direct heredity history, passed whole-matrix permutations, and replayed exactly. However, a later provenance-complete, no-PCA ablation on 200 further untouched matrices did not robustly isolate incremental composition, static-beta, or beta-conditioned-state contributions; the successful composite coordinate therefore remains mechanistically unresolved. A separate support-matched, whole-matrix-cross-fitted confirmation on 200 new matrices found both first-order and run-duration dependence in post-break inheritance prediction. In a post-hoc audit of 145,516 qualifying F12 episodes, only 4.5–6.5% placed all three episode daughters in a mutually `H>0.9` neighbourhood, so the three-fission target does not establish a distinct new compositional regime. A subsequent prospective F32 campaign did confirm the rarer operational event of a break followed by eight inherited fissions whose daughters were mutually `H>0.9` and all `H<=0.85` from the old anchor: rates were 1.81–2.11% across candidates and branch halves. Its registered state-added predictor failed three of four complete gates. A six-family 80-matrix pilot then failed its 75% model-selection-stability gate, and a separately registered direct-plus-hurdle ensemble passed candidate 03 but failed candidate 02 on 200 new matrices. We conclude that the reconstructed simulator contains a past-observable propensity for plastic-heredity break-and-renewal, statistical sequence dependence, and rare distinct coherent eight-fission episodes, but not a common validated predictor for the strict event, an established network–state–history mechanism, a general regime transition, or cross-clean-room causal control. This is not PhiID support, biological-memory evidence, first-replicator prediction, Codex intervention evidence, or validation of prebiotic chemistry.

**Keywords:** GARD; compositional heredity; origin-of-life simulation; empirical committor; break-and-renewal; causal emergence; PhiID; reproducibility; stochastic shooting

## Introduction

Origin-of-life models ask how chemical systems can acquire persistent, heritable organization before template-based genetic replication is available. GARD models one such possibility: a noncovalent assembly grows through molecule exchange with an environment, while a weighted catalytic matrix modulates joining and leaving propensities; when the assembly reaches a size threshold it divides, and a daughter continues the lineage [1–4]. Earlier GARD work described compotypes or composomes—quasi-stationary regions of composition space exhibiting compositional inheritance—and explored their fidelity, quasispecies-like dynamics, evolvability, and relation to catalytic-network structure [2–6]. These are model-level claims about compositional assemblies, not experimental demonstrations of primordial life.

Pigozzi and Levin recently proposed a different diagnostic layer [7]. Their preprint reported that a PhiID-related measure of causal architecture, denoted Phi-r, exhibits punctuated dynamics, is positively associated with self-replication, predicts the later replication trajectory from its first quarter, and can be manipulated to change replicator longevity and abundance. If reproducible, such a result would connect an information-dynamic measure to the formation of self-maintaining organization before conventional evolutionary selection.

The preprint does not provide supplementary material and states that code will be released upon publication. Its prose, figures, public source lineage, and earlier GARD implementations leave material ambiguities: stochastic update semantics, Poisson exposure, overshoot, fission, daughter continuation, molecular versus generational clocks, the meaning of “most recurring composition,” the exact PhiID scalar, temporal fitting, variable-length prediction tensors, and intervention refitting. These ambiguities make a conventional push-button reproduction impossible. They also create a scientific risk: a completed-trajectory label or estimator can resemble an outcome retrospectively while containing information unavailable before the claimed event.

We therefore treated replication as a layered forensic problem. First, we asked which computations could be reconstructed and validated without inventing missing settings. Second, we separated visible paper resemblance from prospective prediction and causal control. Third, when the paper-facing pipeline remained nonidentifiable, we explored whether a better-defined dynamical event possessed a reliable state-dependent probability. The exploration eventually changed the scientific object from entry into one recurring composition to a process of disruption and renewal in parent–daughter heredity.

This manuscript reports both the constraints and the resulting positive finding. The constraints matter: public evidence did not support the target paper's prospective or causal conclusions under the tested reconstruction. The positive result is different: a completely frozen past-observable coordinate predicts the independently measured probability of an inheritance break followed by a new short hereditary episode. We refer to this bounded simulator phenomenon as **plastic-heredity break-and-renewal**. A later experiment separately confirmed a stricter distinct, mutually coherent eight-fission episode, but recurrence, attractor identity, and “regime switching” remain hypotheses rather than established consequences.

## Research Plan Status

### Program architecture

The full research program contains seven serial Experiments and 112 originally planned research steps. Its design principle is that later Experiments must consume frozen earlier evidence rather than retroactively modify it.

| Experiment | Planned scientific objective | Current status |
|---|---|---|
| E01 | Forensic reconstruction of GARD, Phi-r, labels, prediction, and intervention claims | **Complete and deterministically closed** |
| E02 | Adversarial validation of leakage, labels, preprocessing, estimators, attractor geometry, controls, and replacement variables | First stage authorized; not executed |
| E03 | Search a broader GARD morphospace for robust predictive or controllable regimes | Not started; depends on E02 |
| E04 | Determine whether causal individuality lies at molecule, module, assembly, or coalition scale | Not started |
| E05 | Test attractor commitment, regenerative return, and compositional memory | Not started |
| E06 | Test operational minimal competency, equifinality, memory, and flexible response | Not started |
| E07 | Test collective pattern maintenance, distributed memory, group heredity, and expanded individuality | Not started |

### E01 completion

E01 produced two independently structured GARD engines, stochastic and numerical validation, an executable ambiguity registry, a source and metric lineage audit, 59-claim paper adjudication, prediction and intervention reconstructions, and an additive S19 continuation running through L54, including separately versioned repair steps L06R, L11R, and L49R. S18 froze the paper-facing verdict. S20 performed closeout only and generated no new scientific outcome.

The extensive S19 continuation was adaptive and exploratory. It tested paper-literal labels and tensors, then attractor and committor hypotheses, and finally process-based heredity. Failed-closed attempts, repairs, nulls, and contradictory outcomes were retained. Only L54 is an untouched confirmation of the final process-risk coordinate; it does not confirm the earlier exploratory sequence as a whole.

### Work remaining

A future E02-style program should treat the L54 coordinate as a **candidate replacement causal-architecture variable**, not as PhiID, paper replication, or causal control. The independent Codex branch has already applied adversarial locks for provenance, common support, candidate consistency, strict-event geometry, replay and multiplicity. Its 80-matrix six-family pilot failed stable model selection, and a separately registered 200-matrix direct-plus-hurdle ensemble failed the required all-candidate gate. Strict-eight predictor search is now closed. A separate clean-room directive authorizes intervention tests on the already validated F12 `JOINT_BREAK_RUN3` coordinate, but it has no scientific result at this evidence cutoff; PhiID remains a nonprivileged comparator.

## Methods, Data & Tools

### Evidence sources and provenance

The primary scientific context was the uploaded arXiv v1 manuscript and its native-resolution figures [7]. The original PDF had SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`. Public sources were pinned by commit, including historical GARD commit `86dff6320d5ae91b4e831471079ff46749b14df9`, PhiRL commit `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`, related IIGR ancestry, and the `phyid` reference implementation. Related-team `BreakingGRNMemories` code was audited as lineage inspiration, not treated as the unavailable GARD-paper implementation.

Every research step used domain-separated seeds, explicit input identities, scope checks, replay or regeneration, and compact artifact hashes. Catalytic matrices were the higher-level independent unit in scientific inference. Candidate simulators, labels, clocks, completed-fit versus past-only values, and development versus validation/confirmation cohorts remained separate.

### Claim and ambiguity registration

Before simulation, the manuscript was decomposed into 59 scientific claims and 120 implementation parameters. Unresolved or conflicting fields remained typed sentinels. No registry containing unresolved fields could be executed. This prevented paper proximity from silently selecting an exposure, clock, daughter rule, label, estimator, or denominator.

### GARD reconstruction

The reconstructed model used integer molecular-composition vectors and a nonnegative catalytic matrix

\[
\beta_{ij}=\exp(A+\sigma Z_{ij}), \qquad A=-4,\ \sigma=4,
\]

with joining and leaving propensities built from basal rates, reservoir abundances, total mass, and catalytic boost. The full engine exposed categorical-event, direct-Gillespie, and vector-Poisson update branches; fixed-size and binomial fission; first, second, and uniformly selected daughter continuation; maximum-step, overshoot, and extinction semantics. Historical and independent engines were compared only when configured to identical explicit branches.

In the clean S13Y cohort, 100 new shared catalytic matrices and 100 matched mass-40 distinct-singleton initial states were generated. Each matrix was run under two frozen candidate contracts for 100 fissions, yielding 200 trajectories. The candidates preserved different exposure/daughter/trim semantics inherited from the public-source audit; neither was designated the author implementation.

### Stochastic and numerical validation

Simulator validation included exact propensity fixtures, analytical event probabilities, catalytic-matrix log moments, fission laws, Poisson count distributions, Gillespie waiting-time probability-integral transforms, conservation invariants, stream separation, and injected failures. The information pipeline was validated on independent Gaussian noise, duplicated Markov bits, XOR synergy, coupled autoregression, and planted block systems. All 16 PhiID atoms were tracked. CPU float64 was authoritative; optional Gaussian GPU calculations required CPU agreement.

### Compositional preprocessing and similarity

Relative compositions were closed to unit sum. Paper-facing PhiRL analyses used additive-0.5 zero replacement, full CLR transformation, and removal of the final component after verifying round-trip error. Historical cosine similarity was

\[
H(x,y)=\frac{x\cdot y}{\lVert x\rVert_2\lVert y\rVert_2}.
\]

The original adjacent molecular label was `Y_t = I(H_t>0.9)`. Boundary inheritance in later process analyses used strict parent-to-selected-daughter `H>0.9` at fission. Strict and inclusive thresholds were never interchanged.

### PhiRL reconstruction

The pinned PhiRL pipeline filtered near-constant variables, z-scored retained dimensions, formed fast lag-one Gaussian mutual information, added a small graph floor, used an unnormalized Fiedler-sign partition, averaged variables within the two partitions, regularized Gaussian covariance, decomposed local information through a safe PhiID lattice, and exposed distinct `integrated` and `emergence` values. Source-defined emergence was synergy plus downward-causation terms. Completed-fit values estimated active dimensions, graph, partition, means, and covariance from the complete trajectory; they were marked future-dependent. Prefix-only values refit using only available history.

### Paper-facing reconstruction

Figure 2 was reconstructed using candidate-specific aggregate and run-level local information trajectories, slope tests, three-standard-deviation excursions, robust excursions, and Ljung–Box diagnostics. Its primary aggregate used an available-case median at each molecular index, so unequal-length tail positions had fewer contributing trajectories; full-cohort-support, majority-support, and normalized-lifetime alignments were retained as robustness views. The manuscript does not specify its exact alignment or Ljung–Box lag, so numerical agreement for those elements remains underdetermined. Figures 3–4 used runwise level and first-difference correlations, state contrasts, matrix bootstraps, circular shifts, and pooled/run-summary alternatives. Figure 5 reused the frozen first-quarter-to-final-three-quarters tensor, matrix-grouped 64/16/20 fit/validation/test partitions, exact MLP architecture, masks, and ten repetitions. Subsequent forensic loops tested already frozen recurring-attractor targets, zero padding, length cues, and at most source-supported tensor alternatives without changing the simulator or H threshold. Figure 6/Table 1 used a literal online append-edit-refit-current-prefix scorer and common random streams.

### Empirical committors and target transfer

For a restored state \(s\), an empirical finite-horizon committor was estimated as

\[
\hat q(s)=\frac{1}{B}\sum_{b=1}^{B} I(\text{branch }b\text{ enters the target within the horizon}).
\]

L28 used 128 futures per state and two independently frozen halves of 64. L30 used a separate 64-branch, eight-observation ensemble to construct a short-propagator teacher for the H32 probability. L31 applied the frozen coordinate to new matrices. L36–L37 generated independent lineages under the same matrix and used reciprocal leave-one-lineage-out basins and any-attractor atlases to test whether the target existed independently of the completed evaluated lineage.

### Plastic-heredity outcomes

Later analyses abandoned completed-run centroids. At each fission, inheritance was a strict parent–daughter `H>0.9`. Separate outcomes included break, generic resumption, a three-fission episode, a five-fission episode, old-neighbourhood return, continuous old-anchor gain, and repeated return. Retrospective physical onset was kept separate from online certification. Fixed-count permutations tested temporal order beyond marginal inheritance counts. A review then found that the original L44 IID comparator had been fitted on first and singleton symbols that neither model scored, so its numerical Markov gain was withdrawn. A new preregistered analysis fitted IID, first-order Markov, and duration-aware semi-Markov models on identical transition destinations; cross-fitting held out whole catalytic matrices in both directions, and inference resampled matrices.

The final target, `JOINT_BREAK_RUN3`, was fixed prospectively:

> within the next 12 fissions, observe an inheritance break and subsequently certify three consecutive inherited fissions.

An uninterrupted inherited run was not counted as a new episode. The target used no completed-run centroid and no observations beyond the fixed horizon. It did not require the three episode daughters to be mutually similar, far from the pre-break composition, recurrent, or persistent beyond three fissions. Because cosine similarity is not transitive, three successful parent-to-daughter inheritance boundaries do not by themselves define one coherent compositional neighbourhood.

A later reviewer-motivated audit regenerated every positive target branch from the scaled5, MECHCONF, and MECHCONF2 confirmation cohorts: three disjoint 200-matrix campaigns containing 384,000 archived F12 futures and 145,516 qualifying episodes. For the first qualifying episode in each future, the audit measured all pairwise similarities among the three daughters, similarity of every daughter to the first pre-break parent, uninterrupted run length, same-run persistence to five, and a later break followed by another run of three within F12. The last quantity is a second renewal, not return to the same composition. Continuous geometry was primary. Threshold views at pairwise coherence `>0.90`, `>0.95`, and `>0.975` and old-anchor separation `<=0.90`, `<=0.85`, and `<=0.80` were explicitly post-hoc sensitivities. Estimates and 95% intervals used 4,096 whole-matrix bootstraps over all 200 matrices, including matrices with no qualifying episodes, and were reported separately by cohort, candidate, and branch half; persistence limited by the F12 boundary was right-censored.

### Past-observable process-risk model

L53 registered four students: a training prior; nine direct history/phase variables; beta-only graph structure; and full state/graph/history. The nine direct variables were normalized generation, current mass, prefix inheritance fraction, recent-five inheritance fraction, trailing inheritance run, latest parent–daughter H, fissions since the latest break, current inheritance state, and current inherited/non-inherited run duration.

The full target-blind representation contained 195 coordinates encoding current composition and catalytic-network-conditioned state while preserving molecule-label permutation invariance. Development-only scaling and 12-component PCA were frozen. The 12 components were combined with the nine direct variables in ridge logistic regression with `C=0.1`. L53 was an adaptive discovery step motivated by earlier results and evaluated registered targets at F4, F8, and F12. Within that step, the graph layer, feature subset, PCA dimension, and regularization value were fixed rather than searched. The final F12 joint-event transform, model, coefficients, target, threshold, landmarks, and candidate rules were then carried unchanged into the untouched L54 confirmation.

### Untouched confirmation

L54 used a new 256-bit seed domain, 40 new shared catalytic matrices and initial states, and both candidate contracts, producing 80 complete 100-fission trajectories. Five post-fission states per trajectory were restored at generations 20, 35, 50, 65, and 80: 400 states total. Each state received 64 independent F12 futures, divided before outcomes into two halves of 32. The primary campaign contained 25,600 branches and was exactly regenerated in a second campaign.

All L53 scalers, PCA objects, coefficients, priors, probability mappings, and gates were unchanged. Primary comparisons were full state/graph/history versus direct history/phase, beta-only structure, and the training prior. Reliability, calibration, branch log loss, q-Brier score, overall Spearman rank, matrix-centered Spearman rank, 4,096 matrix bootstraps, and 512 whole-matrix permutations were preregistered.

### Review-driven mechanistic attribution

Code review showed that the original composite comparison mixed duplicate directions, prior-cycle and cumulative growth clocks, composition, static beta structure, and beta-conditioned state. A first prospective ablation separated unique history plus clocks (`H10`), mass-free composition (`S`), a state-free beta block (`B`), and a mass-free beta-conditioned block (`I`). Added blocks used up to 12 development-only PCs, with `I` residualized against `H10+S+B`. That suite was sealed before evaluation on 200 new `MECHCONF` matrices.

A subsequent review identified that the first beta block compressed 87 distinct directions to 12 unsupervised PCs. A versioned correction assigned every feature explicit state, beta, history, clock, mass, and phase provenance. Its fixed threshold-free beta panel included all eligible legacy invariant coordinates, raw/log-beta and row/column-strength summaries, row/column-strength correlation, reciprocity, normalized asymmetry, all 100 normalized singular values, stable rank, spectral entropy, and strength concentration. Constants and exact affine duplicates were removed on development data; no added block used PCA. Sequential offset-ridge additions preserved the preceding prediction exactly, and penalties were selected by deterministic five-fold whole-matrix development cross-validation. The complete protocol, 12-test Holm family, and a disjoint seed were sealed before an explicitly post-hoc old-cohort diagnostic and before 200 new `MECHCONF2` matrices, 2,000 states, and 128,000 F12 futures were generated.

### Strict eight-fission endpoint and prediction follow-up

A later design froze an F32 primary endpoint at ordinary post-fission landmarks. After the first `H<=0.9` break, it required eight consecutive strict `H>0.9` inheritances, strict `H>0.9` similarity among all 28 daughter pairs, and inclusive `H<=0.85` similarity between every daughter and the pre-break parent. Five-daughter pairwise and eight-daughter centroid definitions were prespecified secondary endpoints with no rescue role. Development and confirmation each used 200 disjoint catalytic matrices, both candidates, five landmarks, 128 futures per state, fixed branch halves, whole-matrix inference, and complete replay.

That campaign confirmed occurrence but not its registered state-added prediction contrast. A separately versioned next-predictor workflow left the sealed result untouched. It recorded every eligible post-break eight-run and an exact continuous joint geometry margin; added permutation-invariant analytic summaries of expected joining/leaving drift, event noise, entropy/concentration change, tangent-Jacobian stability, and recent compositional velocity; and decomposed the endpoint into break, later eight-run, and strict geometry. Six model families were fixed in advance: direct offset ridge, a three-stage hurdle, hierarchical matrix propensity plus state/dynamics, local dynamics, leakage-safe first-five/centroid auxiliary stacking, and one bounded calibrated histogram-gradient model. The 80-matrix by 128-future pilot had to improve strict log loss in both candidates and both branch halves and achieve at least 75% matrix-bootstrap selection stability. Four families had positive gains in all four cells, but bootstrap selection divided between direct ridge (43.63%) and hurdle (55.44%); no family passed, so the registered confirmation path stopped.

Because an equal direct-plus-hurdle ensemble was conceived after the pilot outcome, it was frozen as a distinct prospective hypothesis and could not rescue the failed pilot. The new registration prohibited refitting and recalibration and used one 200-matrix by 128-future cohort. Its primary gate required positive gain, a positive whole-matrix-bootstrap lower bound and Holm-adjusted matrix-randomization `p<0.05` in all four candidate/half cells. Candidate 03 passed both halves, candidate 02 passed neither, and the all-candidate verdict was false; all 256,000 futures replayed exactly.

### Statistical principles

Molecular observations and stochastic branches were not treated as independent catalytic systems. Matrix bootstraps resampled catalytic matrices and carried their states/branches together. Candidates were never pooled to rescue disagreement. Undefined values were retained with reasons. Repeated model splits were paper-facing diagnostics, not independent biological experiments. A favorable completed-fit branch could not satisfy a prospective gate.

### Tools

The computational environment used Python 3.13, NumPy, SciPy, pandas, PyArrow, scikit-learn, statsmodels, NetworkX, Matplotlib, and pinned source repositories. CPU float64 was authoritative for simulation, Phi, and scientific summaries. Up to eight CPU workers were used with one numerical-library thread per worker where parallelism materially reduced runtime. GPU paths were optional validation branches only and were never the sole scientific reference.

### Mounted Dataset Use

No mounted datasets were required or used; all scientific data were generated by validated simulations, with the uploaded manuscript and pinned source repositories serving as research inputs.

## Results (Illustrated)

### 1. An auditable simulator family was established

The independent simulator reproduced the public-historical branch at the distribution level. Across the initial engine comparison, 512 propensity cases agreed exactly, and event, fission, and one-generation distribution distances passed prospectively fixed gates. In the larger stochastic audit, 26/26 primary distribution tests, 54/54 invariants, and 7/7 failure injections passed.

![Validated event probabilities for the reconstructed simulator branches.](figures/foundation_s07_event_validation.png)

**Figure 1. Reconstructed GARD stochastic validation.** Standardized residuals compare observed event frequencies with their named analytical targets across explicit branches. This result establishes internal simulation validity, not identity with the unavailable author implementation.

Information validation was branch-specific. Eligible reference and Gaussian CPU/GPU paths agreed within `1e-10`, while a discrete accelerated branch failed all relabel-invariance tests and was excluded. The strict fixed-window estimator required at least 512 effective observations; shorter windows were not made valid by relaxing this criterion.

### 2. Paper-like Phi dynamics were retrospective and incomplete

The clean S13Y cohort found small positive completed-fit PhiRL association with the exact adjacent-H label. Median runwise Spearman correlation was 0.058 and 0.055; 78/99 and 76/99 defined runs were positive. Yet the label was deterministically `I(H>0.9)`, giving zero conditional entropy after exact H, and prefix-only median correlations reversed to −0.074 and −0.069.

Figure 2-like spikes were common, but aggregate slopes were significantly positive instead of nonsignificant under the preregistered available-case alignment. Because trajectories had unequal lengths, late aggregate positions had fewer contributors; none of the retained support-restricted or normalized-time alternatives identified the manuscript's unpublished alignment. The paper's Ljung–Box lag also remains unspecified.

![Candidate-specific aggregate and run-level Figure 2 reconstruction.](figures/paper_s14_figure2_reconstruction.png)

**Figure 2. Punctuated resemblance with an aggregate contradiction.** Representative trajectories show abrupt excursions resembling the target figure. Under the primary available-case alignment, candidate aggregates nevertheless exhibit significant positive slopes, contrary to the reported no-trend result. Unequal tail support and the unresolved author alignment and Ljung–Box lag limit exact numerical comparison.

Figures 3–4 correlation and state-contrast directions were also partially recovered, but effect sizes, counts, and temporal semantics differed. S18 therefore classified the paper-facing result as `PARTIAL_DIRECTIONAL_RETROSPECTIVE_RECONSTRUCTION`.

### 3. Figure 5 and causal control were not reconstructed

The adjacent-H label occupied approximately 98% of real molecular observations. As a result, majority-class accuracy was also approximately 98%, while all valid-cell balanced accuracies remained around 0.5 and all matrices were already positive by the first-quarter cutoff.

![Figure 5 prediction results under frozen tensor semantics.](figures/paper_s16_figure5_reconstruction.png)

**Figure 3. Target saturation in Figure 5 reconstruction.** High raw accuracy is shared by every model and the dummy because the target is nearly always positive. It does not constitute discrimination or prediction before first appearance.

Alternative recurring-attractor targets could reproduce a roughly 60% majority dummy but not the paper's feature ordering. Unmasked padding and length information produced inflated all-cell accuracy, whereas valid-cell models remained nondiscriminative. No source-grounded tensor hypothesis reconciled all visible Figure 5 constraints.

The literal online intervention procedure was executed and replayed, but max/control/min persistence and occupancy did not follow the reported ordering.

![Literal Figure 6 intervention reconstruction.](figures/paper_s17_figure6_reconstruction.png)

**Figure 4. Executable scorer without supported causal ordering.** The pipeline enumerates molecular edits and refits the current prefix online. Treatment distributions do not satisfy the manuscript's max ≥ control ≥ min claim, so mechanical reconstruction must not be equated with causal control.

Across 59 claims, S18 recorded 3 supported, 17 directionally supported, 21 not supported within tested scope, 2 underdetermined, and 16 not evaluated.

### 4. A reliable committor existed for a nontransferable target

L28 showed that the matrix-specific completed-run basin had a real finite-horizon probability. Corrected between-state variance was 0.0937/0.0857, split-half rank reliability 0.926/0.933, and 42/36 states lay in the transition region.

![Independent branch-half committor estimates.](figures/attractor_l28_split_half_reliability.png)

**Figure 5. State-dependent empirical committor.** Independent branch halves agree closely and, together with positive noise-corrected between-state variance, support state-dependent stochastic outcome probability even though earlier static representations failed.

The eight-step propagator predicted the H32 committor and transferred unchanged to untouched matrices. Nevertheless, reciprocal independent-lineage targets failed. Cross-lineage strict centroid agreement was only about 0.60–0.75, and a multilineage any-attractor atlas did not support one transferable destination.

![Multilineage target-transfer gates.](figures/attractor_l37_multilineage_decision.png)

**Figure 6. The destination is not network-stable.** Technical and local recurrence gates pass in green, but reciprocal rank, original-teacher transfer, attractor-family, and independent-target committor gates fail in red. A predictable completed-run target is therefore not automatically a general replicator attractor.

### 5. Heredity renews without restoring the old composition

Process-based analysis separated disruption, resumption, sustained inheritance, and restoration. In L44 validation and confirmation cohorts, break probability was approximately 0.64–0.73, resumption given a break 0.88–0.91, a new run of three 0.76–0.82, and a persistent run of five 0.53–0.60. In contrast, old-neighbourhood recovery-event prevalence was about 0.0026–0.0069 in the primary reliability summaries; the corresponding conditional-on-break rate was about 0.0024–0.0051 in the separate plasticity decomposition.

![Seven process probabilities for heredity and return.](figures/heredity_l44_process_prevalence.png)

**Figure 7. Plasticity decomposition.** Common break, resumption, and inherited-episode events are quantitatively distinct from rare old-composition return. The data support renewal of hereditary capacity rather than fixed-composition recovery.

Continuous evidence agreed: mean H gain toward the old anchor at resumption was about −0.26 in every cohort and candidate. Count-matched order tests found positive excess for three-fission episodes (0.026–0.032) and five-fission episodes (0.043–0.060).

The post-hoc episode audit sharply limited the dynamical interpretation. Across the six cohort/candidate comparisons, the mean minimum pairwise daughter similarity within the first qualifying episode was only 0.681–0.704, and only 4.5–6.5% of episodes placed every daughter pair above 0.9. Separation from the pre-break parent was common: 93.2–94.9% kept every episode daughter at or below 0.9 similarity to that anchor. Among episodes whose five-fission status was observable within F12, 75.9–78.9% continued in the same inherited run to five; a later break followed by another observed run of three occurred in 10.2–11.7% of all qualifying futures. This last result is a second renewal, not compositional recurrence. Thus renewal is usually compositionally distinct from the old anchor and often persists when observable, but the three daughters usually do not form one mutually coherent `H>0.9` neighbourhood. These post-hoc findings support the narrower break-and-renewal description and do not prospectively establish a new regime.

The earlier L44 estimate of 0.015–0.022 bits per transition is withdrawn because its IID fit and score used different symbol supports. In a separate prospective 32-fission confirmation on 200 untouched matrices, support-matched first-order Markov models improved over IID by 0.04695/0.03394 bits per transition in candidates 02/03; duration-aware models improved over Markov by another 0.01077/0.00998. Both contrasts had positive matrix-bootstrap lower bounds, Holm-adjusted `p=0.000976`, and positive gains in both whole-matrix cross-fit directions. These effects establish statistical first-order and duration-dependent prediction under the registered models, but not biological memory, molecular storage, error correction, or causality.

Past-only PhiID did not add held-out information beyond direct heredity variables. Descriptive post-break functional coherence was explained by composition and chronology in the registered incremental tests.

### 6. A frozen past-observable coordinate transferred to untouched matrices

The untouched L54 event probability ranged across the transition region: 138/200 and 149/200 states had `0.1<q<0.9`. Independent branch halves were reliable (Spearman 0.938 and 0.924; lower bounds 0.903 and 0.872). After matrix-centering, reliability remained 0.625 and 0.606 with positive lower bounds.

The frozen full-state graph-plus-history model achieved overall q Spearman 0.895–0.918 across branch directions. Direct history alone achieved 0.742–0.822, while beta-only structure was approximately uncorrelated. Within matrices, the full model achieved 0.550–0.697 compared with 0.198–0.345 for direct history. This separation shows that the coordinate contains both stable catalytic-matrix propensity and state-local risk information.

![Frozen rank transfer on untouched matrices.](figures/process_l54_rank_transfer.png)

**Figure 8. Untouched rank transfer.** The full present-state/catalytic-graph/history coordinate outperforms direct history and beta-only structure both overall and after removing matrix means.

Branch log-loss improvement over direct history was 0.041–0.052 across the four candidate/direction comparisons. In the preregistered full-versus-direct gate table, the minimum 95% matrix-bootstrap lower bounds were 0.025922 for candidate 02 and 0.035512 for candidate 03. q-Brier improvements were 0.012–0.018, with all lower bounds positive. All whole-matrix permutation p-values were 0.001949.

![Frozen predictions and independently measured F12 process probabilities.](figures/process_l54_prediction_calibration.png)

**Figure 9. Prediction versus empirical process probability.** The frozen coordinate transfers monotonically in both candidates. Calibration is informative but imperfect, and scatter remains at the state level; the claim is probability ranking and proper-score improvement, not deterministic fate prediction.

All preregistered confirmation gates passed. The finding was classified as an untouched past-observable simulator process-risk coordinate for plastic-heredity break-and-renewal and explicitly as `NOT_PAPER_REPLICATION`.

### 7. The stricter mechanistic decomposition did not confirm

The first review-driven MECHCONF ablation passed its registered state and beta-conditioned-state contrasts but not its static-beta contrast. Because the beta baseline retained only twelve unsupervised components, these results applied to that representation and did not establish that beta was generally uninformative or that a network–state interaction was necessary.

The corrected suite retained 240 distinct static-beta directions, 11/12 state directions, and all 64 beta-conditioned-state directions without PCA. On untouched MECHCONF2, none of the three registered contrasts passed in both candidates and both branch halves. State-over-history log-loss gains were 0.001419/0.001282 in candidate 02 and 0.002216/0.001577 in candidate 03, but confidence intervals or Holm-adjusted randomization values failed. Comprehensive-beta gains beyond history plus state were −0.000226/0.000037 and 0.001546/0.001475; none survived the registered family. Beta-conditioned-state gains were −0.009269/−0.009987 and −0.001738/−0.001221, respectively, and therefore also failed.

Independent branch halves remained reliable (overall Spearman 0.9260/0.9247 and matrix-centered 0.6858/0.6849 in candidates 02/03), all 128,000 futures replayed exactly, and primary gains independently recomputed within `1e-14`. The null attribution is therefore not explained by an unreliable empirical outcome or failed replay. It leaves the original composite algorithm comparison intact but withdraws the stronger conclusion that composition beyond clocks or a necessary beta-conditioned-state interaction has been robustly isolated.

### 8. Rare distinct coherent eight-fission episodes occur, but tested predictors do not transfer across both candidates

The prospective strict F32 occurrence gate passed in both candidates and both fixed branch halves. Candidate-02 rates were 0.01869 and 0.01809, with matrix-bootstrap intervals `[0.01391, 0.02369]` and `[0.01387, 0.02283]`. Candidate-03 rates were 0.02089 and 0.02109, with intervals `[0.01563, 0.02648]` and `[0.01626, 0.02677]`. Across candidates, 5,041 strict events occurred, distributed over 140 and 149 event-positive confirmation matrices. Every discrete and continuous future replayed exactly.

The registered h10-plus-state predictor did not pass the complete four-cell gate. Log-loss gains were 0.000525/0.000284 in candidate 02 and 0.000388/0.000356 in candidate 03. Only candidate-03 half A met all interval and multiplicity requirements; candidate-03 half B narrowly missed the Holm threshold (`p=0.051989`), and both candidate-02 cells failed. Prespecified first-five occurrence was 7.14–8.35%, and centroid occurrence was 8.27–9.41%; their more favorable descriptive prediction patterns cannot rescue the strict all-eight endpoint.

A read-only decomposition of the sealed futures found break rates of 0.6907/0.7386 and later-eight-run rates of 0.5274/0.5624 in candidates 02/03. Conditional on an observed later eight-run, only 0.0349/0.0373 met strict coherence and distinctness. Thus the main bottleneck is not producing a break or a long adjacent-inheritance run, but landing in the narrow geometric subset that is simultaneously coherent and separated from the old anchor. This decomposition is post-hoc and motivates, but cannot validate, the newly registered staged/local-dynamics pilot.

The six-family pilot subsequently generated 102,400 F32 futures from 80 new matrices and replayed them exactly. Event power was adequate and four families improved over `h10` in all four cells, but direct ridge won 43.63% and the hurdle family 55.44% of paired whole-matrix selection bootstraps; neither met the frozen 75% threshold. No family was selected, and the original confirmation path stopped as registered.

A new protocol then froze the pilot-derived `0.5 × direct + 0.5 × hurdle` probability ensemble without refitting or recalibration and tested it on 200 untouched matrices, 2,000 states and 256,000 futures. Gains were positive in all four cells, but candidate-02 intervals crossed zero and Holm-adjusted tests failed (`p=0.1181`), whereas both candidate-03 cells passed (`p=0.0225`). The required all-candidate result was therefore false. The strict event itself recurred at 0.01710/0.02116 in candidates 02/03, and every future replayed exactly.

The supported statement is operational: these simulators sometimes undergo a break followed by a distinct, mutually coherent eight-fission hereditary episode. Calling this general regime switching still requires recurrence, common-basin, perturbation-recovery, or equivalent prospective evidence. The completed prediction tests suggest candidate-dependent signal but do not establish a predictor that transfers prospectively across both simulator contracts.

## Novelty

A web-based novelty check was conducted on 13 August 2026 using title, GARD, compositional-heredity, committor, regime-switching, and causal-emergence queries. Prior work already establishes all core components separately: GARD compositional inheritance [2], compotype/quasispecies behavior [3], beta-network prediction of GARD species [5], transferable heredity metrics [6], PhiID and causal-emergence formalisms [8,9], and committor learning by repeated path sampling [10]. The present work therefore does not claim novelty for compositional heredity, catalytic-network prediction, regime switching, PhiID, or committor estimation in isolation.

No directly matching publication was found that combines all of the following:

1. a prospective GARD process event defined as inheritance break plus a new three-fission hereditary episode;
2. an empirical state-specific F12 probability estimated with independent branch halves;
3. a target-blind current-state/catalytic-graph/history representation;
4. unchanged transfer to seed-firewalled matrices in two candidate simulators;
5. overall and within-matrix reliability, proper-score, permutation, and replay gates.

This negative search is not proof of priority and may miss differently worded or unpublished work. The defensible novelty claim is the operational combination and untouched confirmation inside the reconstructed simulator. A second potentially useful contribution is methodological: the work demonstrates that a reliable committor can coexist with a nontransferable destination, making target reproducibility an independent prerequisite for interpreting transition prediction.

## Discussion

### Forensic reconstruction as scientific evidence

The paper-facing result is neither a complete reproduction nor a simple failure. Several visible directions recur: punctuated local information dynamics, positive completed-fit associations, state contrasts, and some intervention differences. Yet the discrepancies are structurally aligned with missing semantics. Completed-fit PhiRL uses future observations to determine partitions and Gaussian parameters. The principal adjacent-H label is exactly a thresholded stability coordinate. Figure 5 depends on an unresolved target/denominator/tensor convention. The intervention scorer is executable, but its output does not show the reported bidirectional ordering.

These findings place a firm boundary around the interpretation. Public-source resemblance supports a hypothesis about retrospective covariation between information dynamics and compositional stability. It does not support a warning signal available before first replication or causal efficacy of Phi-directed edits.

### Why the attractor framing failed

A recurring composition inferred from the same completed lineage is an intuitively attractive target, but it combines two sources of hindsight: the destination is selected after observing the run, and earlier states are then judged relative to that destination. The committor experiments showed that this target has a genuine local transition probability. Independent-lineage experiments then showed that the destination itself varies across stochastic histories.

This distinction explains why increasingly rich static representations failed to reproduce the short-propagator teacher. The predictor was partly learning approach to a lineage-specific object. The failure was not evidence that the simulator lacks organization; it was evidence that organization should not be identified with one global centroid.

### Plastic heredity as a process

The process analysis suggests a different ontology. Parent–daughter similarity is common but not uninterrupted. When it breaks, a short inherited sequence usually reappears, generally at compositions separated from the old anchor. The post-hoc geometry audit shows that the daughters in such an episode usually do not occupy one mutually `H>0.9` neighbourhood, so this result is renewal of adjacent parent-to-daughter inheritance rather than demonstrated formation of a new compositional regime. In a separate prospectively registered test, recent inheritance state and its run duration improved out-of-matrix next-boundary prediction over nested IID and first-order baselines. Thus the system exhibits a recurring **capacity for short-run heredity** across changing molecular realizations. The predictive dependence may still partly reflect latent matrix or compositional-state heterogeneity and is not, by itself, biological memory.

“Plastic heredity” is deliberately narrower than homeostasis or regime switching. Exact restoration was nearly absent, registered functional variables did not show a residual restored function beyond composition and chronology, and mutually coherent three-daughter episodes were uncommon in the post-hoc audit. The data instead support stochastic breaks followed by renewed short sequences of adjacent compositional inheritance. This aligns with the broader GARD idea that compositional assemblies can carry heritable organization, while modifying the assumption that one privileged composition is the operative replicator.

### What the confirmed coordinate contains—and what remains unresolved

The original frozen composite coordinate is predictively useful beyond its registered direct-history comparator. That is an algorithm comparison, not a decomposition. The first grouped ablation suggested incremental composition and beta-conditioned-state information, but the provenance-complete no-PCA correction did not reproduce either contrast on MECHCONF2. Static beta also failed under both registered suites, although a model-specific null is not proof that beta is generally uninformative.

Accordingly, the data do not establish that the operative object is a catalytic-network × current-state × history interaction. The successful composite may exploit a representation-specific mixture of history, phase, composition, beta-conditioned summaries, regularization, or other correlated directions. Identifying a stable physical compression remains open.

### Relation to the original paper's broader premise

The confirmed result supports a cautious version of the target paper's broad organizational premise: a catalytic assembly's present observables can carry information about the probability of a future inheritance break followed by short-run renewal. The result does not support the paper's chosen quantity or event. The signal is not PhiID, the event is not first arrival of one replicator, and no Codex intervention has yet shown causal control; Fable's positive intervention findings remain external until independently replicated.

This difference is scientifically meaningful. It suggests testing renewal after disruption as a process, rather than assuming a first crossing into one composition-space cluster. The strict F32 campaign shows that a rare coherent, old-anchor-distinct eight-fission episode does occur. Whether those episodes recur, share a transferable basin, recover after perturbation, or constitute a coherent dynamical regime remains a prospective question.

## Limitations and Caveats

### Supported findings

- The named simulator branches and numerical pipelines pass extensive internal validation and replay.
- The L54 process probability and frozen coordinate pass untouched confirmation in both candidates.
- Incremental value of the original composite algorithm beyond its registered direct-history algorithm is supported by proper scores and matrix-centered ranks under the frozen design.

### Null and contradictory results

- A coherent public end-to-end paper pipeline was not identified.
- Past-only Phi/PhiID did not support the original early-warning claim or add to the final process controls.
- Figure 5 valid-cell prediction and Figure 6/Table 1 causal ordering were not reproduced.
- Independent lineages did not support one stable attractor destination.
- Exact old-composition return and residual functional restoration were not supported.
- The provenance-complete no-PCA MECHCONF2 ablation did not confirm incremental state, static-beta, or beta-conditioned-state contributions across both candidates and branch halves.
- The post-hoc coherence audit found that most qualifying three-fission episodes do not place all three daughters in one mutually `H>0.9` compositional neighbourhood; a distinct new regime is not established.
- The registered state-added predictor did not pass the complete strict eight-fission prediction gate, despite confirmed occurrence of that endpoint.

### Scope limitations

1. **Simulator only.** All positive evidence concerns two reconstructed GARD candidates. It is not experimental chemistry, biological heredity, or validation of an origin-of-life scenario.
2. **Author ambiguity.** The authors' implementation, exact self-replicator label, Phi scalar, tensor semantics, and intervention scorer remain unavailable. The reconstruction cannot prove that the paper's private pipeline would fail.
3. **Operational targets.** Strict `H>0.9`, a run of three, F12, and five post-fission landmarks define the original process event. The later strict endpoint adds F32, eight inherited fissions, all-pairs `H>0.9`, and old-anchor `H<=0.85`. Both are registered operational choices; other sensible definitions may differ.
4. **Adaptive discovery.** L19–L53 explored many hypotheses. This weakens discovery-stage confirmation credibility even though L54 was prospectively frozen and seed-firewalled.
5. **Probability, not fate.** One realized future is a noisy draw. The coordinate predicts an empirical probability, not a deterministic event for each state.
6. **Calibration.** Proper scores and ranking transferred, but calibration is not exact and state-level scatter remains.
7. **Representation.** The successful 195-coordinate graph/state PCA model is engineered, and its proposed decomposition failed to transfer across registered representations. The result does not identify one interpretable physical mechanism.
8. **Matrix and phase coverage.** The original confirmation used forty matrices and the corrections used 200 each; five post-fission landmarks still do not span the complete GARD morphospace or every growth-cycle phase.
9. **No Codex causal intervention yet.** Changing a predictor is not the same as changing the process probability. The new intervention directive has no outcome at this evidence cutoff, while Fable's molecular-edit results remain branch-specific.
10. **Procedural complexity.** Multiple additive repairs and one recorded S17 compute waiver were retained. No failed result was overwritten, but specification multiplicity is substantial.
11. **Memory boundary.** The corrected sequence analysis establishes held-out first-order and run-duration prediction after a break. It does not distinguish intrinsic memory from latent catalytic-matrix or compositional-state heterogeneity, and it does not demonstrate molecular storage or error correction.
12. **Event boundary.** `JOINT_BREAK_RUN3` certifies three adjacent parent-to-daughter inheritance successes after a break. It does not require episode-wide compositional coherence, distinctness, recurrence, or persistence, and the later geometry audit is post-hoc.
13. **Strict-event boundary.** The F32 all-eight endpoint prospectively establishes a coherent, old-anchor-distinct episode under fixed cosine thresholds. It does not establish recurrence, perturbation recovery, a universal attractor, regime switching outside this operational definition, or causal controllability. Its first state-added predictor failed the complete gate, its six-family pilot failed stable selection, and its separately registered direct-plus-hurdle ensemble failed the all-candidate gate.

### Missing validation before broader claims

The composite coordinate still requires adversarial target/preprocessing sensitivity, alternative matrix distributions and parameter regimes, and a representation-stable decomposition before symbolic or mechanistic claims. The commissioned intervention program can test causal actionability without first declaring that decomposition solved, but any positive result must remain tied to its frozen algorithm or independently specified physical rule. External experimental validation would require a physical system with repeatable state preparation and observable fission/heredity events; none is supplied here.

## Conclusion & Future Directions

This investigation reconstructed a substantial portion of a recent GARD/PhiID paper while sharply narrowing its defensible interpretation. Paper-like completed-fit associations and spikes were recoverable, but prospective first-replicator prediction and Phi-directed causal control were not supported within the tested public reconstruction. The closest attractor-based transition signal was real yet aimed at a lineage-specific destination.

The productive scientific result emerged after changing the target from a composition to a process. In the reconstructed simulator, parent–daughter heredity is plastic: it breaks and is often followed by a new short inherited sequence whose daughters are generally separated from the old anchor but are not usually mutually coherent at `H>0.9`. A frozen composite state/graph/history algorithm predicts the independently measured probability of that future break-and-renewal event on untouched matrices in both simulator candidates. The corrected ablation did not identify a representation-robust state, static-beta, or beta-conditioned-state component responsible for that performance. Separate prospective experiments established that rarer distinct, mutually coherent eight-fission episodes occur at about 2% probability, while the initial state-added predictor, a six-family pilot and a frozen two-model ensemble all failed to establish prediction across both simulator candidates.

The failed strict-event pilot and ensemble are closed outcomes and must not be rescued by another strict-eight predictor search. The next Codex stage instead asks whether the already validated F12 composite is causally actionable and whether Fable's proposed catalytic-support mechanism transfers to Codex's independent contracts. A new directive requires fresh matrices and seeds, frozen pre-outcome choices, common random streams, whole-matrix inference, exact replay and serial stop gates. Until those experiments report, Fable's causal results remain external hypotheses rather than cross-clean-room findings.

## New Hypotheses

The following are hypotheses motivated by the completed evidence, not established findings beyond the stated simulator scope.

### H1. The validated F12 composite is causally actionable across simulator contracts

**Rationale.** The original F12 combined algorithm transferred beyond its direct-history comparator in both Codex candidates, but prediction alone does not imply that edits selected by the coordinate change process probability. Fable reports such control in a separate implementation.

**Hypothesis.** One-molecule mass-preserving substitutions selected by the frozen Codex F12 predictor bidirectionally change `JOINT_BREAK_RUN3` probability in both Codex candidates, while a uniformly random legal substitution is equivalent to no-op.

**Test.** Follow the commissioned CR1 protocol on 200 fresh matrices, five landmarks and 64 F12 futures per model-up, model-down, random and no-op arm, using common random streams, fixed branch halves, whole-matrix bootstrap/randomization inference and complete replay. Require every candidate/half cell to pass the directional, interval, multiplicity and random-equivalence gates.

### H2. Duration dependence survives controls for latent heterogeneity

**Rationale.** A prospectively registered, support-matched semi-Markov model improved out-of-matrix one-step transition scoring beyond a first-order Markov model in both candidates. The pooled models did not condition on current composition or matrix-specific random effects, so this is predictive duration dependence rather than proof of an intrinsic memory mechanism.

**Hypothesis.** Episode age retains predictive value after controlling for catalytic-matrix propensity and the current compositional state.

**Test.** Freeze matrix-random-effect and state-conditioned duration models and compare them with matched first-order controls on another untouched matrix cohort; require preserved duration gain in both candidates before using mechanistic-memory language.

### H3. The relevant invariant is hereditary capacity, not molecular identity

**Rationale.** Generic resumption was common, exact return was extremely rare, old-anchor gain was negative, and independent lineages did not share one basin.

**Hypothesis.** GARD supports equivalence classes of locally hereditary states whose compositions can change while maintaining comparable parent–daughter inheritance dynamics.

**Test.** Define independent-lineage process-based equivalence classes using transition kernels rather than centroids and test reciprocal transfer without the evaluated lineage's completed future.

### H4. Phi/PhiID is at most a retrospective stability proxy under the tested reconstruction

**Rationale.** Completed-fit values resembled paper associations, past-only directions reversed or added no process value, and the exact adjacent-H target was definitionally circular.

**Hypothesis.** Any robust PhiID association in this setup is mediated by compositional stability, temporal fitting, or target prevalence rather than incremental prospective organization.

**Test.** In E02, compare past-only Phi atoms with the confirmed continuous process committor while preserving exact H, history, phase, and graph/state controls.

### H5. A simple catalytic-support rule captures a transferable part of the control law

**Rationale.** Fable reports that a one-scalar catalytic-support rule retained much of its learned controller's effect and that norm-matched beta surgery changed break risk. Those findings remain branch-specific and provide externally specified hypotheses rather than training targets.

**Hypothesis.** Substitutions that increase or decrease the catalytic support of currently present molecular types bidirectionally change F12 break-and-renewal probability, and tightening versus loosening the occupied catalytic subnetwork shifts risk at fixed composition; matched random controls remain null.

**Test.** Apply the directive's independently implemented CR3 molecular rule and CR4 norm-matched beta-surgery protocols with orientation and perturbation magnitude sealed before outcomes. Require effects, intervals, multiplicity gates, random-control equivalence and exact replay in both candidates and branch halves.

## Code, Artifacts, and Reproducibility

Repository-backed implementations are preserved on the experiment branch `eidosoma/groups/42` of `Eidosoma/arrival-of-self-replicators`. Research-step reports, machine-readable result tables, figures, seed firewalls, runtime records, and artifact hashes are retained in the experiment artifact hierarchy. Public-source identities include [ModelingOriginsofLife/GARD](https://github.com/ModelingOriginsofLife/GARD), [pigozzif/PhiRL](https://github.com/pigozzif/PhiRL), [pigozzif/IntegratedInformationGeneRegulation](https://github.com/pigozzif/IntegratedInformationGeneRegulation), and [pigozzif/BreakingGRNMemories](https://github.com/pigozzif/BreakingGRNMemories). Source lineage is cited for reproducibility and does not imply that any repository is the missing paper implementation.

The clean-room replication folder additionally retains checksum-sealed strict-regime development and confirmation bundles, the read-only next-predictor diagnostic, source-hashed registrations, the sealed failed pilot, its separately versioned post-hoc decision report, and the independently registered ensemble confirmation. The prediction implementation is isolated from the earlier sealed workflow and exposes separate `diagnose`, `register-design`, `pilot`, `confirm`, read-only `status`, and non-scientific `smoke` commands; the original `confirm` correctly refused the failed pilot. A separate `regime_ensemble_confirmation` workflow froze and tested the equal direct-plus-hurdle ensemble on a disjoint cohort, replayed all 256,000 futures exactly, and recorded a failed all-candidate primary verdict. `FULL_FABLE_REPLICATION_INSTRUCTIONS.md` now specifies the next independent intervention program, but no registration or scientific result from that program is claimed in this draft.

## References

1. Segré D, Ben-Eli D, Deamer DW, Lancet D. Compositional genomes: prebiotic information transfer in mutually catalytic noncovalent assemblies. *Proceedings of the National Academy of Sciences*. 2000;97:4112–4117. [GARD source context](https://github.com/ModelingOriginsofLife/GARD).
2. Segré D, Shenhav B, Kafri R, Lancet D. The molecular roots of compositional inheritance. *Journal of Theoretical Biology*. 2001;213(3):481–491. [doi:10.1006/jtbi.2001.2440](https://pubmed.ncbi.nlm.nih.gov/11735293/).
3. Gross R, Fouxon I, Lancet D, Markovitch O. Quasispecies in population of compositional assemblies. *BMC Evolutionary Biology*. 2014;14:265. [doi:10.1186/s12862-014-0265-1](https://pmc.ncbi.nlm.nih.gov/articles/PMC4357159/).
4. Lancet D, Zidovetzki R, Markovitch O. Systems protobiology: origin of life in lipid catalytic networks. *Journal of the Royal Society Interface*. 2018;15:20180159. [doi:10.1098/rsif.2018.0159](https://pubmed.ncbi.nlm.nih.gov/30045888/).
5. Markovitch O, Krasnogor N. Predicting species emergence in simulated complex pre-biotic networks. *PLOS ONE*. 2018;13:e0192871. [doi:10.1371/journal.pone.0192871](https://pmc.ncbi.nlm.nih.gov/articles/PMC5813963/).
6. Guttenberg N, Laneuville M, Ilardo M, Aubert-Kato N. Transferable measurements of heredity in models of the origins of life. *PLOS ONE*. 2015;10:e0140663. [doi:10.1371/journal.pone.0140663](https://pmc.ncbi.nlm.nih.gov/articles/PMC4610668/).
7. Pigozzi F, Levin M. Causal Architecture Dynamics Prior to Arrival of Self-replicators in a Model of Catalytic Networks Relevant to Origin-of-Life. arXiv:2607.28250v1. 2026. [arXiv](https://arxiv.org/abs/2607.28250).
8. Mediano PAM, Rosas FE, Carhart-Harris RL, Seth AK, Barrett AB. Beyond integrated information: a taxonomy of information dynamics phenomena. arXiv:1909.02297. [arXiv](https://arxiv.org/abs/1909.02297).
9. Rosas FE, Mediano PAM, Jensen HJ, Seth AK, Barrett AB, Carhart-Harris RL, Bor D. Reconciling emergences: an information-theoretic approach to identify causal emergence in multivariate data. arXiv:2004.08220. [arXiv](https://arxiv.org/abs/2004.08220).
10. Jung H, Covino R, Arjun A, Leitold C, Dellago C, Bolhuis PG, Hummer G, et al. Machine-guided path sampling to discover mechanisms of molecular self-organization. *Nature Computational Science*. 2023;3:334–345. [doi:10.1038/s43588-023-00428-z](https://www.nature.com/articles/s43588-023-00428-z).
11. PhiRL public source repository. Commit `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`. [GitHub](https://github.com/pigozzif/PhiRL).
12. Historical GARD public source repository. Commit `86dff6320d5ae91b4e831471079ff46749b14df9`. [GitHub](https://github.com/ModelingOriginsofLife/GARD).
