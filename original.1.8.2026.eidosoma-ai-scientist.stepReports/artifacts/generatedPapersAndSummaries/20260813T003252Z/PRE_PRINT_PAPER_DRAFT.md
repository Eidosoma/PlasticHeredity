# From Retrospective PhiID Resemblance to a Past-Observable Plastic-Heredity Risk Coordinate in Two Reconstructed GARD Simulators

**Pre-print draft for human review**  
**Submission metadata status:** The scientific synthesis has been reviewed, but the author list, affiliations, contributions, acknowledgements, funding, competing interests, and corresponding-author details require completion and approval by the submitting investigators.

## Abstract

The Graded Autocatalysis Replication Domain (GARD) model provides a computational framework for studying compositional inheritance in mutually catalytic molecular assemblies. A recent preprint reported that a PhiID-derived causal-emergence quantity rises before the first self-replicator, predicts later replication from the first quarter of a trajectory, and functions as an intervention target. We performed a branch-explicit forensic reconstruction of those claims and then used the resulting evidence to ask a broader question: whether any past-observable property predicts future self-maintaining organization in reconstructed GARD dynamics.

We independently implemented and validated historical and paper-described GARD branches, audited self-replicator labels and compositional preprocessing, reproduced public PhiRL/IIGR source semantics, and reconstructed Figures 2–6 and Table 1 under two frozen simulator candidates. Paper-like punctuated information excursions and retrospective association directions were observed. However, aggregate trends differed, the closest associations were completed-fit and label-coupled, first-quarter prediction did not exceed appropriate controls, and literal Phi-directed interventions did not reproduce the reported max/control/min ordering. The immutable paper-facing verdict was therefore partial directional retrospective reconstruction, with prospective prediction and causal control unsupported within the tested scope.

We next estimated state-dependent empirical committors by restoring simulator states and launching independent future branches. A reliable committor into a completed-run attractor and an untouched-confirmed eight-step propagator coordinate were found, but independent lineages under the same catalytic matrix did not recover one transferable basin or basin family. We therefore replaced the destination with a prospective process: an inheritance break followed, within twelve fissions, by a new three-fission episode of strict parent–daughter compositional inheritance. This process distinguished plastic renewal from exact return to an old composition, which was extremely rare.

Finally, a frozen target-blind representation of current composition, catalytic-network-conditioned state, growth/fission phase, and recent heredity history was tested on 40 new catalytic matrices, 80 trajectories, 400 post-fission states, and 25,600 independent branch futures plus exact regeneration. In both simulator candidates, the coordinate ranked independently measured process probabilities overall (Spearman 0.895–0.918) and within matrices (0.550–0.697), improved proper scores beyond direct heredity history, passed whole-matrix permutations, and replayed exactly. We conclude that the reconstructed simulator contains a past-observable propensity for plastic-heredity regime switching. This is not PhiID support, first-replicator prediction, intervention evidence, or validation of prebiotic chemistry. It is a simulator-specific, untouched-confirmed process-risk result that reframes emergence as renewal of hereditary capacity rather than arrival at one privileged composition.

**Keywords:** GARD; compositional heredity; origin-of-life simulation; empirical committor; regime switching; causal emergence; PhiID; reproducibility; stochastic shooting

## Introduction

Origin-of-life models ask how chemical systems can acquire persistent, heritable organization before template-based genetic replication is available. GARD models one such possibility: a noncovalent assembly grows through molecule exchange with an environment, while a weighted catalytic matrix modulates joining and leaving propensities; when the assembly reaches a size threshold it divides, and a daughter continues the lineage [1–4]. Earlier GARD work described compotypes or composomes—quasi-stationary regions of composition space exhibiting compositional inheritance—and explored their fidelity, quasispecies-like dynamics, evolvability, and relation to catalytic-network structure [2–6]. These are model-level claims about compositional assemblies, not experimental demonstrations of primordial life.

Pigozzi and Levin recently proposed a different diagnostic layer [7]. Their preprint reported that a PhiID-related measure of causal architecture, denoted Phi-r, exhibits punctuated dynamics, is positively associated with self-replication, predicts the later replication trajectory from its first quarter, and can be manipulated to change replicator longevity and abundance. If reproducible, such a result would connect an information-dynamic measure to the formation of self-maintaining organization before conventional evolutionary selection.

The preprint does not provide supplementary material and states that code will be released upon publication. Its prose, figures, public source lineage, and earlier GARD implementations leave material ambiguities: stochastic update semantics, Poisson exposure, overshoot, fission, daughter continuation, molecular versus generational clocks, the meaning of “most recurring composition,” the exact PhiID scalar, temporal fitting, variable-length prediction tensors, and intervention refitting. These ambiguities make a conventional push-button reproduction impossible. They also create a scientific risk: a completed-trajectory label or estimator can resemble an outcome retrospectively while containing information unavailable before the claimed event.

We therefore treated replication as a layered forensic problem. First, we asked which computations could be reconstructed and validated without inventing missing settings. Second, we separated visible paper resemblance from prospective prediction and causal control. Third, when the paper-facing pipeline remained nonidentifiable, we explored whether a better-defined dynamical event possessed a reliable state-dependent probability. The exploration eventually changed the scientific object from entry into one recurring composition to a process of disruption and renewal in parent–daughter heredity.

This manuscript reports both the constraints and the resulting positive finding. The constraints matter: public evidence did not support the target paper's prospective or causal conclusions under the tested reconstruction. The positive result is different: a completely frozen past-observable coordinate predicts the independently measured probability that a hereditary regime will break and a new short hereditary episode will form. We refer to this bounded simulator phenomenon as **plastic-heredity regime switching**.

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

E02 must now treat the L54 coordinate as a **candidate replacement causal-architecture variable**, not as PhiID, paper replication, or causal control. Before seeing E02 outcomes, the program requires locks for future-suffix leakage, calibration, candidate consistency, numerical robustness, preprocessing and target sensitivity, and incremental value beyond exact H, direct history, ordinary stability, phase, and matrix propensity. PhiID remains a nonprivileged comparator. Intervention work is not authorized until this adversarial gate is passed.

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

Later analyses abandoned completed-run centroids. At each fission, inheritance was a strict parent–daughter `H>0.9`. Separate outcomes included break, generic resumption, a three-fission episode, a five-fission episode, old-neighbourhood return, continuous old-anchor gain, and repeated return. Retrospective physical onset was kept separate from online certification. Fixed-count permutations tested temporal order beyond marginal inheritance counts; cross-fitted Markov and semi-Markov models tested dependence beyond IID inheritance frequency.

The final target, `JOINT_BREAK_RUN3`, was fixed prospectively:

> within the next 12 fissions, observe an inheritance break and subsequently certify three consecutive inherited fissions.

An uninterrupted inherited run was not counted as a new episode. The target used no completed-run centroid and no observations beyond the fixed horizon.

### Past-observable process-risk model

L53 registered four students: a training prior; nine direct history/phase variables; beta-only graph structure; and full state/graph/history. The nine direct variables were normalized generation, current mass, prefix inheritance fraction, recent-five inheritance fraction, trailing inheritance run, latest parent–daughter H, fissions since the latest break, current inheritance state, and current regime duration.

The full target-blind representation contained 195 coordinates encoding current composition and catalytic-network-conditioned state while preserving molecule-label permutation invariance. Development-only scaling and 12-component PCA were frozen. The 12 components were combined with the nine direct variables in ridge logistic regression with `C=0.1`. L53 was an adaptive discovery step motivated by earlier results and evaluated registered targets at F4, F8, and F12. Within that step, the graph layer, feature subset, PCA dimension, and regularization value were fixed rather than searched. The final F12 joint-event transform, model, coefficients, target, threshold, landmarks, and candidate rules were then carried unchanged into the untouched L54 confirmation.

### Untouched confirmation

L54 used a new 256-bit seed domain, 40 new shared catalytic matrices and initial states, and both candidate contracts, producing 80 complete 100-fission trajectories. Five post-fission states per trajectory were restored at generations 20, 35, 50, 65, and 80: 400 states total. Each state received 64 independent F12 futures, divided before outcomes into two halves of 32. The primary campaign contained 25,600 branches and was exactly regenerated in a second campaign.

All L53 scalers, PCA objects, coefficients, priors, probability mappings, and gates were unchanged. Primary comparisons were full state/graph/history versus direct history/phase, beta-only structure, and the training prior. Reliability, calibration, branch log loss, q-Brier score, overall Spearman rank, matrix-centered Spearman rank, 4,096 matrix bootstraps, and 512 whole-matrix permutations were preregistered.

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

Continuous evidence agreed: mean H gain toward the old anchor at resumption was about −0.26 in every cohort and candidate. Count-matched order tests found positive excess for three-fission episodes (0.026–0.032) and five-fission episodes (0.043–0.060). First-order Markov inheritance improved on an IID frequency model by 0.015–0.022 bits per transition. These effects indicate sticky local regimes, but they do not establish biological memory or error correction.

Past-only PhiID did not add held-out information beyond direct heredity variables. Descriptive post-break functional coherence was explained by composition and chronology in the registered incremental tests.

### 6. A frozen past-observable coordinate transferred to untouched matrices

The untouched L54 event probability ranged across the transition region: 138/200 and 149/200 states had `0.1<q<0.9`. Independent branch halves were reliable (Spearman 0.938 and 0.924; lower bounds 0.903 and 0.872). After matrix-centering, reliability remained 0.625 and 0.606 with positive lower bounds.

The frozen full-state graph-plus-history model achieved overall q Spearman 0.895–0.918 across branch directions. Direct history alone achieved 0.742–0.822, while beta-only structure was approximately uncorrelated. Within matrices, the full model achieved 0.550–0.697 compared with 0.198–0.345 for direct history. This separation shows that the coordinate contains both stable catalytic-matrix propensity and state-local risk information.

![Frozen rank transfer on untouched matrices.](figures/process_l54_rank_transfer.png)

**Figure 8. Untouched rank transfer.** The full present-state/catalytic-graph/history coordinate outperforms direct history and beta-only structure both overall and after removing matrix means.

Branch log-loss improvement over direct history was 0.041–0.052 across the four candidate/direction comparisons. In the preregistered full-versus-direct gate table, the minimum 95% matrix-bootstrap lower bounds were 0.025922 for candidate 02 and 0.035512 for candidate 03. q-Brier improvements were 0.012–0.018, with all lower bounds positive. All whole-matrix permutation p-values were 0.001949.

![Frozen predictions and independently measured F12 process probabilities.](figures/process_l54_prediction_calibration.png)

**Figure 9. Prediction versus empirical process probability.** The frozen coordinate transfers monotonically in both candidates. Calibration is informative but imperfect, and scatter remains at the state level; the claim is probability ranking and proper-score improvement, not deterministic fate prediction.

All preregistered confirmation gates passed. The finding was classified as an untouched past-observable simulator process-risk coordinate for plastic-heredity switching and explicitly as `NOT_PAPER_REPLICATION`.

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

The process analysis suggests a different ontology. Parent–daughter similarity is common but not uninterrupted. When it breaks, a locally hereditary regime usually reappears, generally at a composition farther from the old anchor. Temporal order and dwell duration modestly exceed an IID inheritance-frequency baseline. Thus the system exhibits a persistent **capacity for heredity** across changing molecular realizations.

“Plastic heredity” is deliberately narrower than homeostasis. Exact restoration was nearly absent, and registered functional variables did not show a residual restored function beyond composition and chronology. The data instead support stochastic switching among local hereditary regimes. This aligns with the broader GARD idea that compositional assemblies can carry heritable organization, while modifying the assumption that one privileged composition is the operative replicator.

### What the confirmed coordinate contains

The full coordinate is useful because neither beta alone nor history alone explains it. Beta-only graph features carry little state-local information. Direct heredity history ranks much of the stable risk, but the current state conditioned by catalytic network structure adds proper-score and within-matrix value. The relevant object is an interaction:

\[
\text{catalytic network}\times\text{current composition/state}\times\text{recent hereditary history and phase}.
\]

The exact mechanistic compression remains unresolved. The frozen PCA/ridge coordinate is compact enough to transfer but is not yet a symbolic physical law. Its strongest coefficients include direct heredity variables, while grouped evidence shows that the graph/state block contributes incrementally.

### Relation to the original paper's broader premise

The confirmed result supports a cautious version of the target paper's broad organizational premise: before a future self-maintaining event, a catalytic assembly's present organization can carry information about that event's probability. The result does not support the paper's chosen quantity or event. The signal is not PhiID, the event is not first arrival of one replicator, and no intervention has shown causal control.

This difference is scientifically meaningful. It suggests that an “arrival” in a catalytic medium may be the renewal of a dynamical capacity after disruption, rather than the first crossing into one composition-space cluster. Such a process framing may be more robust to lineage variation and better suited to prospective testing.

## Limitations and Caveats

### Supported findings

- The named simulator branches and numerical pipelines pass extensive internal validation and replay.
- The L54 process probability and frozen coordinate pass untouched confirmation in both candidates.
- Incremental value beyond direct history is supported by proper scores and matrix-centered ranks under the frozen design.

### Null and contradictory results

- A coherent public end-to-end paper pipeline was not identified.
- Past-only Phi/PhiID did not support the original early-warning claim or add to the final process controls.
- Figure 5 valid-cell prediction and Figure 6/Table 1 causal ordering were not reproduced.
- Independent lineages did not support one stable attractor destination.
- Exact old-composition return and residual functional restoration were not supported.

### Scope limitations

1. **Simulator only.** All positive evidence concerns two reconstructed GARD candidates. It is not experimental chemistry, biological heredity, or validation of an origin-of-life scenario.
2. **Author ambiguity.** The authors' implementation, exact self-replicator label, Phi scalar, tensor semantics, and intervention scorer remain unavailable. The reconstruction cannot prove that the paper's private pipeline would fail.
3. **Operational target.** Strict `H>0.9`, a run of three, F12, and five post-fission landmarks are registered choices. Other sensible process definitions may differ.
4. **Adaptive discovery.** L19–L53 explored many hypotheses. This weakens discovery-stage confirmation credibility even though L54 was prospectively frozen and seed-firewalled.
5. **Probability, not fate.** One realized future is a noisy draw. The coordinate predicts an empirical probability, not a deterministic event for each state.
6. **Calibration.** Proper scores and ranking transferred, but calibration is not exact and state-level scatter remains.
7. **Representation.** The 195-coordinate graph/state block and PCA are engineered. The result does not yet identify one interpretable physical mechanism.
8. **Matrix and phase coverage.** Forty confirmation matrices and five post-fission landmarks do not span the complete GARD morphospace or every growth-cycle phase.
9. **No causal intervention.** Changing a predictor is not the same as changing the process probability. No molecular edit has yet established causal control.
10. **Procedural complexity.** Multiple additive repairs and one recorded S17 compute waiver were retained. No failed result was overwritten, but specification multiplicity is substantial.

### Missing validation before broader claims

The coordinate still requires adversarial target/preprocessing sensitivity, alternative matrix distributions and parameter regimes, symbolic or mechanistic compression, and matched interventions. External experimental validation would require a physical system with repeatable state preparation and observable fission/heredity events; none is supplied here.

## Conclusion & Future Directions

This investigation reconstructed a substantial portion of a recent GARD/PhiID paper while sharply narrowing its defensible interpretation. Paper-like completed-fit associations and spikes were recoverable, but prospective first-replicator prediction and Phi-directed causal control were not supported within the tested public reconstruction. The closest attractor-based transition signal was real yet aimed at a lineage-specific destination.

The productive scientific result emerged after changing the target from a composition to a process. In the reconstructed simulator, parent–daughter heredity is plastic: it breaks and often renews around a different molecular state. A frozen present-state/catalytic-graph/history coordinate predicts the independently measured probability of that future break-and-renewal event on untouched matrices in both simulator candidates.

The next stage should try to falsify this coordinate, not celebrate it. If it survives E02 sensitivity and simpler-control tests, the following experiment should use matched stochastic futures to test whether small molecular edits that raise or lower the coordinate also change the process probability. Only such an intervention could begin to support causal-control language. Broader regime discovery, causal-boundary, memory, competency, and collective-organization studies should remain downstream of that gate.

## New Hypotheses

The following are hypotheses motivated by the completed evidence, not established findings beyond the stated simulator scope.

### H1. Plastic-heredity transition propensity is encoded by a network–state–history interaction

**Rationale.** Beta-only structure did not transfer, and direct history lost substantial within-matrix information. The combined state/graph/history coordinate passed untouched incremental gates.

**Hypothesis.** A small set of propensity-weighted graph/state interactions, conditioned on regime duration and recent parent–daughter H, mechanistically controls the local hazard of break and renewal.

**Test.** Freeze grouped ablations and symbolic students on the existing development evidence, confirm them on new matrices, and require proper-score and centered-rank preservation.

### H2. Heredity dynamics are semi-Markov rather than IID or purely Markov

**Rationale.** Inheritance probability depended on current regime and dwell duration; duration models improved one-step transition scoring, although simple process models did not reconstruct the full committor.

**Hypothesis.** The hazard of leaving or re-entering a hereditary regime depends on episode age in addition to the current binary state and matrix propensity.

**Test.** Use longer branch horizons and preregistered duration bins on untouched matrices; compare pooled, matrix-random-effect, and state-conditioned semi-Markov models.

### H3. The relevant invariant is hereditary capacity, not molecular identity

**Rationale.** Generic resumption was common, exact return was extremely rare, old-anchor gain was negative, and independent lineages did not share one basin.

**Hypothesis.** GARD supports equivalence classes of locally hereditary states whose compositions can change while maintaining comparable parent–daughter inheritance dynamics.

**Test.** Define independent-lineage process-based equivalence classes using transition kernels rather than centroids and test reciprocal transfer without the evaluated lineage's completed future.

### H4. Phi/PhiID is at most a retrospective stability proxy under the tested reconstruction

**Rationale.** Completed-fit values resembled paper associations, past-only directions reversed or added no process value, and the exact adjacent-H target was definitionally circular.

**Hypothesis.** Any robust PhiID association in this setup is mediated by compositional stability, temporal fitting, or target prevalence rather than incremental prospective organization.

**Test.** In E02, compare past-only Phi atoms with the confirmed continuous process committor while preserving exact H, history, phase, and graph/state controls.

### H5. Molecular edits that shift the frozen process-risk coordinate may shift branch probability

**Rationale.** The coordinate predicts future probability, but prediction alone does not imply manipulability.

**Hypothesis.** Small molecule additions or deletions that produce large target-blind changes in the frozen score causally change F12 break-and-renewal probability under matched common-random-number branches.

**Test.** Only after E02 confirmation, register score-raising, score-lowering, matched-random, and no-op edits; estimate paired branch-probability contrasts without refitting or selecting by outcome.

## Code, Artifacts, and Reproducibility

Repository-backed implementations are preserved on the experiment branch `eidosoma/groups/42` of `Eidosoma/arrival-of-self-replicators`. Research-step reports, machine-readable result tables, figures, seed firewalls, runtime records, and artifact hashes are retained in the experiment artifact hierarchy. Public-source identities include [ModelingOriginsofLife/GARD](https://github.com/ModelingOriginsofLife/GARD), [pigozzif/PhiRL](https://github.com/pigozzif/PhiRL), [pigozzif/IntegratedInformationGeneRegulation](https://github.com/pigozzif/IntegratedInformationGeneRegulation), and [pigozzif/BreakingGRNMemories](https://github.com/pigozzif/BreakingGRNMemories). Source lineage is cited for reproducibility and does not imply that any repository is the missing paper implementation.

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
