# Different Arrivals of Replicators: A Potential Discovery

## The potential discovery

The strongest new result in the E01 continuation is not a reconstruction of PhiID announcing the first appearance of one fixed self-replicator. It is an untouched-confirmed, past-observable probability coordinate for a different process: **plastic hereditary-regime switching**.

Within the two frozen reconstructed GARD simulator candidates, parent-to-daughter compositional inheritance is common. Hereditary episodes can break and a new locally hereditary episode can form without returning to the old molecular composition. L54 confirmed that current physical/catalytic state plus directly observed hereditary history carries information about the probability of that break-and-renewal process over the next twelve fissions.

This should be called a potential discovery because it is robust inside the reconstructed simulator yet still simulation-specific. It is neither evidence that the paper's author code was identified nor evidence about real prebiotic chemistry.

## How the question changed

The paper frames replication as arrival at recurring composition-space clusters and argues that causal architecture changes before replicators appear. E01 initially followed that framing literally. Adjacent-H labels were too prevalent, recurring-centroid definitions were too sparse or sticky, and completed-run attractors often failed to transfer across independent lineages. The evidence instead favored a process view:

1. compositional heredity is frequent;
2. heredity breaks;
3. exact return to the previous composition is rare;
4. a new locally hereditary regime often forms;
5. the scientifically useful object is the probability of a future break followed by renewed heredity, not entry into a privileged completed-run centroid.

Retrospective physical onset and online certification were kept separate. The final target used only a prospectively fixed future process over a bounded fission horizon and did not discover a target basin from the evaluated future.

## Reproducible path from L44 to L54

### L44 — establish the process family

L44 reused 35,840 frozen F12 branch futures and separated ordinary inheritance, break, resumption, exact leave-return and new hereditary episodes. It found sticky hereditary episodes beyond an IID baseline and selected the online event `NEW_HEREDITARY_EPISODE_RUN3`: after an inheritance break, three consecutive strict-`H>0.9` parent/daughter fissions certify a new episode. This was exploratory and not confirmation.

### L45 — test Phi/PhiID incrementally

L45 used the fixed run-3 process target and asked whether past-only or completed-fit Phi-related summaries add value beyond direct hereditary history. The past-only branch did not add held-out value. Completed-fit values remained future-dependent. Classification: `PAST_ONLY_PHI_NOT_INCREMENTAL_FOR_HEREDITARY_EPISODE` and `PHI_PROCESS_NON_SUPPORT`.

### L46–L47 — distinguish composition from function

L46 compared old-regime restoration with local functional coherence. It found no restoration of the old functional regime, although new local regimes could be coherent. L47 showed that the registered functional vector did not add beyond composition and chronology. This pruned a claim of functional homeostasis.

### L48 — quantify shooting requirements

L48 compared branch budgets and a registered adaptive allocation. The conservative reliability contract required the full 64-branch half; the tested adaptive scheme did not improve it. Stochastic shooting remained the measurement method rather than a static biomarker.

### L49/L49R — test longitudinal risk

L49 stopped before branches because one frozen state lacked twelve future fissions; it remains failed closed. L49R made the additive, outcome-blind eligibility repair, restored 400 states and generated 25,600 new F12 futures. It did not establish a reliable within-lineage risk trajectory, constraining a universal rising-warning interpretation.

### L50 — align state, event and horizon

L50 selected 80 shared matrices—40 development and 40 validation—from the frozen L23 cohort, kept candidate 2 and candidate 3 separate, restored five post-fission states per matrix at generations 20, 35, 50, 65 and 80, and generated 51,200 independent branch futures. Each state had 64 branches and nested F4, F8 and F12 outcomes. The target used strict parent/daughter `H>0.9`; its primary F12 joint event was the first inheritance break followed by a new run of three inherited fissions. Break, conditional resumption and joint probability were kept separate. The empirical process probability was reliable, but shooting did not yet add beyond direct history.

### L51–L52 — identify regime duration and the branch teacher

L51 fit fixed IID, Markov, duration-dependent/semi-Markov and matrix-prefix baselines. It established duration-dependent switching and strong between-matrix variation, but the registered process models did not reconstruct the empirical committor. L52 cross-fit pooled, other-landmark-within-matrix and state-local duration hazards between independent branch halves. The empirical committor was compressible to state-local duration hazards, but those hazards were branch-derived and not past-observable. L52 therefore supplied the teacher signal, not the final operational coordinate.

### L53 — distill the teacher into past-observable students

L53 used no new simulation. It froze four models before derived outcomes:

- `TRAINING_PRIOR`;
- `DIRECT_HISTORY_PHASE`, nine online variables: normalized generation, mass, prefix inheritance fraction, recent-five inheritance fraction, trailing inherited run, latest parent/daughter H, fissions since latest break, current inheritance state and current regime duration;
- `BETA_STRUCTURE`, twelve development-only PCs of twenty beta-only graph summaries;
- `FULL_STATE_GRAPH_HISTORY`, twelve development-only PCs of a 195-coordinate target-blind graph/current-state representation plus the nine direct variables.

The graph representation preserved current molecule counts and catalytic-network relationships while remaining molecule-permutation invariant. Models were fixed ridge-logistic students with `C=0.1`; PCA dimension was 12; there was no hyperparameter search. They fit only development matrices using branch half A or B and scored validation matrices using the independent opposite half. F4, F8 and F12 plus break, conditional run-3 and joint targets stayed separate. The full state-plus-history model added proper-score and within-matrix ranking value in both candidates, while beta-only structure did not explain transferable capacity. L53 was an adaptive lead, not confirmation.

### L54 — untouched confirmation

L54 applied the entire L53 transformation, PCA objects, coefficients, priors, probability mapping and thresholds unchanged. It generated a new 256-bit seed domain, 40 new shared catalytic matrices and matched initial states, 80 complete 100-fission trajectories (40 per candidate), and 400 fixed post-fission states. Each state received 64 independent F12 futures, split prospectively into two halves of 32. One primary campaign contained 25,600 branch futures; a second exact campaign regenerated them. The four event/daughter/fission/trim seed streams were independently domain-separated.

The primary F12 joint event was unchanged: a future inheritance break followed by a new three-consecutive-fission hereditary episode within twelve fissions. The fitted coordinate never used the branch outcome, a completed-run centroid, PhiID or a suffix-derived feature.

All confirmation gates passed in both candidates:

- all 400 states available;
- split-half committor Spearman `0.9376/0.9237`, with bootstrap lower bounds `0.9028/0.8725`;
- within-matrix centered split-half lower bounds `0.4559/0.4747`;
- frozen full-state overall committor ranks approximately `0.895–0.918`;
- within-matrix centered ranks approximately `0.550–0.697`;
- minimum proper-score improvement lower bounds beyond direct history were positive (`0.02592` candidate 2 and `0.03551` candidate 3 across registered directions; all registered q-Brier lower bounds also positive);
- all registered whole-matrix permutation p-values `0.001949`;
- old L53 model/prediction replay, input firewall, branch identities, second-campaign regeneration, report regeneration and artifact hashes all passed.

The catalytic matrix was the higher-level independent unit; 4,096 matrix bootstraps and 512 whole-matrix permutations were used. Repeated states and branches were never treated as independent catalytic systems.

## Failed Phi/PhiID replications

The potential discovery must not be presented as belated support for PhiID.

- The manuscript's displayed equation, its “one atom” wording and public PhiRL/IIGR outputs do not uniquely identify one metric. L12 classified the identity as `PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT` and concluded `AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION`.
- Source-defined completed-fit emergence can show paper-like retrospective association, but it fits partitions and Gaussian parameters using the completed trajectory. It is future-dependent and cannot support early-warning language.
- Under the frozen adjacent-H label, the target was exactly determined by `H>0.9`, making unrestricted increment beyond contemporaneous H impossible.
- S16 did not support prospective first-quarter prediction within the tested task; Figure 5 tensor audits exposed prevalence, padding and length ambiguities rather than a validated prospective Phi advantage.
- S17 did not support prospective causal control; max/control/min intervention ordering and stronger paired-control gates failed in the tested reconstruction.
- L45 directly tested Phi-related quantities against the new process target. Past-only Phi was not incremental beyond direct heredity controls; completed-fit quantities remained retrospective.
- No Phi or PhiID quantity was computed in L54.

Phi/PhiID is therefore retained only as a nonprivileged E02 comparator, alongside all its null and contradictory evidence.

## Relation to the Levin/Pigozzi paper

The result does not reproduce the paper's specific claim that Phi-r rises before the first self-replicator. It also does not identify the authors' simulation, label, prediction tensor or intervention scorer.

It does support a narrower version of the paper's broader organizational premise: a catalytic assembly can have a measurable, higher-order state-dependent propensity for future self-maintaining organization. Here the organization is not arrival at one fixed recurring composition. It is a network-conditioned ability to move through a break and establish another locally hereditary episode. The predictor needs catalytic graph/current-state structure plus hereditary history and phase; neither a global prior, direct history alone nor beta-only propensity explains the whole confirmed signal.

That is a different “arrival”: the arrival of renewed hereditary capacity after disruption, with molecular identity allowed to change. It is compatible with plastic heredity and regime switching, not fixed-composition homeostasis.

## How to reproduce the result

1. Check out repository branch `eidosoma/groups/42` at the frozen L54 implementation commit recorded in `L54/implementation_lock.json`.
2. Verify the L50–L54 manifests and all upstream hashes; never use an invalidated cache or replace a unit.
3. Reconstruct the strict `H>0.9` parent/daughter inheritance sequence at post-fission boundaries and the F12 joint break-plus-run-3 target exactly as frozen.
4. Rebuild L53's 195 graph/current-state coordinates and nine history/phase coordinates, then apply the recorded development-only scaling, 12-component PCA, ridge coefficients and probability mapping without refitting.
5. For untouched confirmation, use new shared matrix/initial-state identities, both frozen candidates, 100 fissions, landmarks 20/35/50/65/80, and 64 independently seeded F12 branches per state.
6. Split branches 32/32 before outcomes. Score each half against models fit on the opposite frozen L53 half and retain candidate/direction separation.
7. Require exact trajectory/state/feature/model/prediction/branch replay; a zero-overlap seed firewall; split-half reliability; calibration/proper scores; overall and within-matrix committor ranks; 4,096 matrix bootstraps; 512 whole-matrix permutations; suffix/target blindness; and a complete second branch campaign.
8. Reproduce the machine-authoritative gates in `L54/scientific_gate_results.parquet` and the exact report/hash manifest.

## Interpretation limits

The strict-H inheritance event is operational and threshold-dependent. A run of three is short; F12 is one opportunity horizon; the five landmarks are post-fission and do not cover every phase. The graph representation is a compact engineered summary. Confirmation occurred only in the two reconstructed simulator candidates and does not identify author code, real chemistry or biological heredity.

The coordinate predicts a probability. Prediction is not intervention and not causal control. It has not shown that deliberately changing its value changes the process probability. That is a later scientific question requiring matched future branches and separate authorization.

## Next test posture

The separately authorized E02 first stage is `E02-S01-ADVERSARIAL-VALIDATION-OF-PLASTIC-HEREDITY-PROCESS-RISK-v1.0.0`. It should consume S18 and L54 immutably and try to falsify the coordinate through leakage, calibration, candidate-consistency, numerical, preprocessing, target-sensitivity and incremental-control tests. The result may survive, narrow, become model-specific or fail. E02 must preserve each outcome and must not privilege this coordinate—or PhiID—because of the story that led to it.
