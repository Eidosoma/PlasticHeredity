# Codex research directive:
# Independent clean-room replication of the Fable causal-intervention program

## Mission

The strict coherent-eight prediction program is now closed.

Preserve the existing result exactly:

- The strict coherent, old-anchor-distinct eight-fission event is a
  cross-clean-room occurrence result.
- The prospectively frozen direct-plus-hurdle ensemble did not pass the
  registered four-cell prediction gate.
- Candidate 03 showed candidate-specific prospective prediction, but no
  common predictor transferred across both Codex simulator candidates.
- Do not run another strict-eight predictor search, change ensemble weights,
  introduce candidate-specific rescue models, or reinterpret the failed
  primary gate.

Begin a new, independently registered intervention program targeting the
already validated common process:

    JOINT_BREAK_RUN3

This is the Codex clean-room replication of causal findings reported by a
separate independent implementation.

The scientific question is:

> Do Codex’s independently reconstructed GARD candidates exhibit causal,
> graded, physically interpretable, and feedback-maintainable control of
> heredity break-and-renewal under small molecular or catalytic-network
> interventions?

Use Codex’s own simulator contracts, frozen predictors, feature pipeline,
states, seeds, code, and analysis. Do not copy or import Fable code, frozen
models, matrices, states, seeds, selected edits, result files, or controller
objects.

The Fable results are external hypotheses and post-seal comparison
benchmarks only.

This program must remain completely separate from strict-eight occurrence
and prediction. Do not target the strict-eight event in this program.

---

# 1. Fixed scientific definitions

## 1.1 Simulator contracts

Use the existing Codex candidates without alteration:

Candidate 02:
- Poisson exposure 0.10.
- Whole-assembly trimming after overshoot.
- Fixed-size daughter sampling.
- First daughter continues.

Candidate 03:
- Poisson exposure 0.125.
- Admit joiners only to remaining capacity.
- Binomial partition.
- Second daughter continues.

If repository terminology differs, use the already sealed Codex contracts
that produced the confirmed JOINT_BREAK_RUN3 and strict-eight campaigns.
Do not change them to resemble the other clean room.

Candidates must never be pooled to rescue disagreement.

## 1.2 Inheritance

For a pre-fission parent composition p and the selected daughter d:

    H(p,d) = cosine_similarity(p,d)

Strict inheritance:

    H(p,d) > 0.9

Inheritance break:

    H(p,d) <= 0.9

Use unrounded float64 values.

## 1.3 Primary intervention target

JOINT_BREAK_RUN3 is positive when, within the next 12 fissions:

1. At least one strict inheritance break occurs.
2. Strictly after that break, three consecutive inherited fissions are
   certified.

An uninterrupted inherited run already underway at the restored state does
not qualify.

A positive event certified before later extinction remains positive.
Extinction before certification is negative.

## 1.4 Legal molecular intervention

A legal edit is one mass-preserving one-molecule substitution:

- remove one molecule of type i, requiring n_i >= 1;
- add one molecule of type j;
- require i != j;
- total assembly mass remains unchanged.

The edit occurs instantaneously at the restored post-fission state or at the
specified feedback boundary.

The state’s already observed history variables are held fixed during the
instantaneous edit. Future history evolves normally after simulation
resumes.

## 1.5 Frozen predictor

Use the strongest already frozen, prospectively confirmed Codex predictor
for JOINT_BREAK_RUN3, preferably the immutable 5x-development composite
model if that is the latest confirmed candidate-separated model.

Before any intervention cohort is generated:

- reconstruct its scalers, feature transforms, PCA or other transforms,
  coefficients, priors, and prediction mapping;
- reproduce its archived confirmation predictions to the existing numerical
  tolerance;
- record model and source hashes;
- do not refit, recalibrate, simplify, retrain, or change thresholds.

Do not use:

- the failed strict-eight predictor;
- the direct-plus-hurdle strict-eight ensemble;
- any Fable predictor;
- any outcome from the new intervention cohorts.

---

# 2. Independence and procedural requirements

Create a new versioned intervention module and a new durable ledger.

Suggested artifacts:

- CODEX_INTERVENTION_REPLICATION_PREREGISTRATION.md
- intervention_protocol.json
- intervention_seed_registry.json
- INTERVENTION_RESULTS_LEDGER.md
- test_intervention_replication.py
- results_intervention_replication/
- replay_audits/
- figures/
- frozen_seals/

Before generating a single scientific intervention matrix, seal:

- all scientific questions;
- all endpoints;
- all cohort sizes;
- all arms;
- all edit-selection algorithms;
- all seed domains;
- all branch-half assignments;
- all bootstrap and randomization procedures;
- all equivalence margins;
- all stop rules;
- source hashes;
- frozen-model hashes;
- endpoint and intervention test results.

The registration and source tree must be hashed before scientific matrix
generation.

A smoke run may test I/O, legality, replay, and artifact creation. It must
not disclose intervention effect sizes, arm ordering, event rates, or
candidate differences.

No post-outcome changes are permitted to:

- simulator contracts;
- target thresholds or horizons;
- state landmarks;
- edit-selection algorithms;
- random-control sampling;
- model choice;
- branch count;
- exclusion rules;
- matrix replacement;
- confidence intervals;
- equivalence margins;
- phase advancement gates.

Do not replace failed or extinct matrices after the matrix seed list is
sealed.

Main-path landmark generation may use only the already frozen Codex
extinction/retry contract. Intervention futures are never retried.

The catalytic matrix is always the inference unit. States, branch halves,
arms, controller lineages, and repeated landmarks from one matrix travel
together in every bootstrap or randomization draw.

Use:

- 4,096 whole-matrix bootstrap draws for major one-shot phases;
- 4,096 paired whole-matrix sign randomizations;
- Holm correction across the four candidate-by-branch-half primary cells;
- standard branch-level Bernoulli log loss where predictions are scored;
- ordinary branch-level Brier score where Brier is reported;
- no Jeffreys-q squared-error statistic labeled as a proper Brier score.

All major scientific campaigns require complete deterministic replay.

---

# 3. Randomness and common-random-stream requirements

For paired intervention arms, the arm identity must not appear in the future
simulation seed key.

For a given:

- scientific phase;
- candidate;
- catalytic matrix;
- restored landmark or controller lineage;
- branch index;

all arms receive common random streams.

Once edited states diverge, they may consume random values differently.
Describe this accurately as:

    common random streams

not:

    identical realized futures

Edit-selection randomness for the random arm must use a separate,
domain-separated stream.

Random-arm selection must not consume the future simulation stream.

Use purpose-keyed or independently spawned streams for at least:

- matrix generation;
- main trajectory;
- landmark restoration;
- swap screening;
- random-arm swap choice;
- future event simulation;
- fission;
- controller action;
- bootstrap;
- randomization;
- replay.

No-op trajectories must be verified bitwise against the plain simulator
without an intervention callback.

---

# 4. Mandatory pre-scientific validation suite

At minimum, test:

1. Legal one-molecule substitutions preserve mass.
2. Illegal same-type substitutions are rejected.
3. Removing an absent molecule is rejected.
4. Every legal edit produces a nonnegative integer composition.
5. Edited-state features are permutation-invariant under simultaneous
   molecule-label and beta-label permutation.
6. History features are unchanged by the instantaneous edit.
7. The frozen predictor gives identical predictions before and after
   serialization.
8. Exhaustive swap enumeration contains every and only legal swap.
9. Extreme selection is deterministic under ties.
10. Random-arm selection is uniform over legal swaps.
11. Random selection streams are distinct from future streams.
12. Future branch streams are paired across arms.
13. No-op intervention is bitwise identical to plain simulation.
14. JOINT_BREAK_RUN3 endpoint fixtures cover threshold and horizon edges.
15. A positive event before extinction stays positive.
16. Extinction before certification is negative.
17. Candidate-specific selected-daughter semantics are preserved.
18. Matrix-block bootstrap keeps all within-matrix observations together.
19. Whole-matrix sign randomization preserves arm pairing.
20. Replay exactly reproduces state, edit, endpoint, and process outcomes.
21. Beta-surgery operations preserve positivity and the registered norm.
22. Random beta surgery is norm-matched to targeted surgery.
23. Closed-loop callback does not alter no-op simulator behavior.
24. Release mode applies exactly zero interventions after release.

Do not proceed until the complete suite passes.

---

# 5. Phase CR1 — one-shot predictor-guided causal intervention

## Question

Do single molecular substitutions selected by Codex’s own frozen
JOINT_BREAK_RUN3 predictor causally raise or lower the realized event
probability?

## Cohort

Use:

- 200 completely fresh catalytic matrices shared across candidates;
- both Codex candidates;
- natural, untreated main trajectories;
- restored post-fission landmarks 20, 35, 50, 65, and 80;
- 2 candidates x 200 matrices x 5 states = 2,000 restored states;
- 64 F12 futures per arm per state;
- fixed branch halves:
    A = branches 0–31
    B = branches 32–63;
- complete second replay of all futures.

Do not preselect states by predicted risk, matrix propensity, inheritance
history, or any observed future outcome.

## Swap scoring

For every state, score every legal one-molecule substitution exactly using
the frozen Codex predictor.

Do not use approximate top-k screening unless exhaustive scoring is
computationally impossible. If approximation is necessary:

- preregister it before scientific execution;
- validate it against exhaustive scoring on non-scientific fixtures;
- do not alter it after seeing intervention outcomes.

Define four arms:

1. MODEL_UP:
   legal swap with the largest predicted increase in JOINT_BREAK_RUN3
   probability.

2. MODEL_DOWN:
   legal swap with the largest predicted decrease.

3. RANDOM:
   one uniformly sampled legal swap from an independent selection stream.

4. NOOP:
   unchanged state.

Persist every scored legal swap and the selected edit.

## Outcomes

Primary:

- JOINT_BREAK_RUN3 branch probability.

Registered secondary outcomes:

- break within F12;
- run3 after the first break;
- inherited-boundary count;
- first-break time;
- renewal-certification time;
- survival;
- growth updates per fission;
- entropy and occupied types at horizon end.

## Primary gates

Evaluate separately in:

- candidate 02 half A;
- candidate 02 half B;
- candidate 03 half A;
- candidate 03 half B.

CR1 passes only if all four cells satisfy:

1. Mean paired MODEL_UP minus MODEL_DOWN probability > 0.
2. The 95% whole-matrix bootstrap lower bound is > 0.
3. Holm-adjusted whole-matrix randomization p < 0.05.
4. MODEL_UP > NOOP with a positive bootstrap lower bound.
5. NOOP > MODEL_DOWN with a positive bootstrap lower bound.
6. RANDOM is equivalent to NOOP under both:
   - a preregistered TOST margin of +/-0.025 probability;
   - absolute RANDOM−NOOP no greater than 25% of MODEL_UP−MODEL_DOWN.
7. Exact replay passes.

Report:

- arm means;
- paired effects;
- confidence intervals;
- adjusted p-values;
- per-matrix effects;
- number of matrices with the expected sign;
- maximum single-matrix influence;
- predicted versus realized shift;
- effect by landmark;
- branch-half agreement.

No pooling is allowed to rescue one failed candidate or half.

## Stop rule

If CR1 fails, preserve the result and do not run predictor-guided
dose-response, transfer, or model-guided feedback steering.

The externally specified physical-rule and beta-surgery phases may still run
because they test independent mechanistic hypotheses and do not require the
Codex predictor to control outcomes.

---

# 6. Phase CR2 — graded molecular dose response

Run only if CR1 passes.

## Question

Does the frozen predicted molecular-edit effect contain graded causal
information rather than only identifying one extreme pair?

## Design

Use the same restored states as CR1 but a fresh branch seed domain.

For every state:

1. Compute predicted shift for every legal swap relative to NOOP.
2. Select swaps at fixed empirical quantiles:
   0%, 20%, 40%, 60%, 80%, and 100%.
3. Resolve ties deterministically.
4. Launch 64 F12 futures per selected swap in fixed branch halves.

Do not select quantiles using realized outcomes.

## Primary analyses

Separately by candidate and branch half:

1. Compute within-state Spearman correlation between predicted edit shift
   and realized branch probability across the six edits.
2. Average state-level correlations with matrix-block inference.
3. Fit a state-centered linear calibration:
       realized_delta_q ~ predicted_delta_p
   with matrix-block bootstrap inference.
4. Report monotonic arm means descriptively.

## Gates

All four candidate/half cells must have:

- mean within-state Spearman > 0;
- bootstrap lower bound > 0;
- state-centered calibration slope > 0;
- bootstrap lower bound > 0.

Report attenuation and calibration, but do not recalibrate the predictor.

---

# 7. Phase CR3 — independent replication of the physical catalytic rule

This phase is independent of CR1 and may run even if CR1 fails.

## 7.1 Exact externally specified scalar

Let beta_ij denote the catalytic influence of molecule type i on target type
j under the Codex simulator’s convention.

For candidate target type j in current composition x, define the Fable-rule
support scalar:

    c_j(x,beta) = sum_i x_i * beta_ij

Before execution, verify the beta index orientation against the Codex
propensity equation and document it. Do not choose between row and column
orientations using scientific outcomes.

If Codex stores the transpose convention, implement the mathematically
equivalent catalyst-to-target expression.

## 7.2 Rule arms

For each restored state:

RULE_DOWN, predicted stabilization:
- among present source types, remove one molecule of the type with the
  smallest c_j;
- add one molecule of the type with the largest c_j;
- require a legal substitution and deterministic tie handling.

RULE_UP, predicted destabilization:
- remove one molecule of the present type with the largest c_j;
- add one molecule of the type with the smallest c_j.

Also include:

- RANDOM legal substitution;
- NOOP.

Use a new seed domain and the same 200-matrix, five-landmark, 64-branch
design as CR1.

## Gate

In all four candidate/half cells:

- q_RULE_UP − q_RULE_DOWN > 0;
- matrix-bootstrap lower bound > 0;
- Holm-adjusted matrix-randomization p < 0.05;
- RANDOM equivalent to NOOP under the CR1 specificity margin;
- replay exact.

If CR1 also passed, report the rule’s efficiency:

    (q_RULE_UP − q_RULE_DOWN)
    --------------------------------
    (q_MODEL_UP − q_MODEL_DOWN)

This ratio is descriptive unless separately registered.

---

# 8. Phase CR4 — catalytic-network surgery at fixed composition

This phase tests whether catalytic-web support is causally sufficient
without changing one molecule.

## State cohort

Use fresh states or the CR3 states under a new branch seed domain. State
composition and history remain identical across arms.

## Present set

For current state x, let:

    P = {i : x_i > 0}

The targeted surgery acts on the present-present beta submatrix P x P.

## Fixed surgery magnitude

Set the primary perturbation norm before outcomes:

    delta = 0.05 * ||beta[P,P]||_F

Include secondary registered sensitivities at 0.025 and 0.10 if compute
permits. The primary decision uses delta = 0.05 only.

## Arms

1. TIGHTEN:
   multiplicatively increase all beta entries in P x P by a common positive
   factor chosen numerically so that:

       ||beta_tight - beta||_F = delta

2. LOOSEN:
   multiplicatively decrease all beta entries in P x P by a positive factor
   chosen so that:

       ||beta_loose - beta||_F = delta

3. RANDOM_SURGERY:
   alter the same number of beta entries selected independently of x,
   preserving positivity and matching the exact Frobenius norm delta.
   Use a zero-mean or balanced log-perturbation so this is not merely a
   global increase or decrease in beta.

4. NOOP:
   original beta.

The surgically modified beta remains in effect for the complete F12 future.
Composition is identical at branch launch in all arms.

Persist every changed edge and norm audit.

## Primary prediction

Tightening the currently occupied catalytic web stabilizes heredity.
Loosening it raises break-and-renewal probability.

## Gate

In all four candidate/half cells:

    q_LOOSEN − q_TIGHTEN > 0

with:

- positive 95% matrix-bootstrap lower bound;
- Holm-adjusted randomization p < 0.05;
- RANDOM_SURGERY equivalent to NOOP within +/-0.025;
- exact replay.

Also report break hazard separately. Do not infer the effect is specifically
on renewal unless the shared-break-state experiment below supports that.

---

# 9. Phase CR5 — clean decomposition into resistance and resilience

Run after CR1 or CR3 establishes a causal molecular control effect.

This phase must avoid conditioning on treatment-created breaks.

## 9.1 Development and model freezing

Use only Codex development matrices that have never contributed to any
intervention confirmation.

If the existing 5x development cohort can be exactly regenerated with the
required targets, use it. Otherwise generate:

- 200 new development matrices;
- both candidates;
- five landmarks;
- independent branch futures.

Train and freeze two candidate-separated students using one preregistered
architecture:

1. BREAK STUDENT:

       q_B(s) = P(first break within 6 fissions | s)

2. RENEWAL STUDENT:

       q_R(s) = P(run3 within 8 fissions | a post-break daughter state s)

Use Codex’s existing past-observable feature pipeline. A fixed
candidate-specific ridge penalty may be selected only by whole-matrix
cross-validation on development data from the preregistered grid:

    {0.001, 0.01, 0.1, 1, 10, 100}

Freeze all transforms, penalties, coefficients, and hashes before generating
confirmation matrices.

## 9.2 Stage A — resistance

Confirmation:

- 200 new matrices;
- both candidates;
- five natural landmarks;
- 64 F6 futures per arm.

Arms selected using q_B:

- BREAK_UP;
- BREAK_DOWN;
- RANDOM;
- NOOP.

Primary outcome:

- break within six fissions.

Require positive BREAK_UP−BREAK_DOWN effects with positive bootstrap lower
bounds and Holm-adjusted p < 0.05 in all four candidate/half cells.
Require RANDOM equivalent to NOOP.

## 9.3 Stage B — resilience from an identical broken state

For each confirmation matrix and lineage:

1. Simulate naturally until the first inheritance break.
2. Save the exact selected daughter immediately after that break.
3. Restore that identical daughter state across every intervention arm.
4. Only then apply:
   - RENEWAL_UP;
   - RENEWAL_DOWN;
   - RANDOM;
   - NOOP.
5. Launch 64 independent F8 futures per arm.

Primary:

- run3 within eight fissions.

Secondary:

- run5;
- time to renewal;
- inherited-boundary count;
- old-anchor similarity;
- survival.

This is a causal recovery comparison because every arm starts from the same
already-broken state.

Require positive RENEWAL_UP−RENEWAL_DOWN effects in all four cells and a
clean random-control null.

Compare resistance and resilience effect magnitudes descriptively. Do not
call conditional renewal a major control axis unless the shared-state
causal margin supports it.

---

# 10. Phase CR6 — zero-shot parameter-regime transfer and predicted null

Run only if CR1 passes.

Use the frozen home-regime predictor and edit-selection algorithm without
retraining.

Test four beta-distribution regimes:

- (A, sigma) = (-4, 5)
- (A, sigma) = (-3, 4)
- (A, sigma) = (-5, 4)
- (A, sigma) = (-4, 3)

Use:

- 40 fresh matrices per regime;
- both candidates;
- landmarks 35 and 65;
- MODEL_UP, MODEL_DOWN, RANDOM, NOOP;
- optionally RULE_UP and RULE_DOWN as a separately registered family;
- 48 F12 futures per arm;
- complete replay.

Registered external hypothesis:

- the first three regimes show positive targeted control;
- the weak-heterogeneity (-4,3) regime shows little or no targeted effect.

Transfer gate for each of the first three regimes:

- MODEL_UP−MODEL_DOWN > 0 in both candidates;
- positive matrix-bootstrap lower bound;
- randomization p < 0.05;
- RANDOM equivalent to NOOP.

Null-regime gate:

Use a preregistered TOST equivalence margin:

    +/-0.04 probability

The (-4,3) result is classified as a confirmed null only if the complete
confidence interval lies within that margin in both candidates.

A confidence interval merely crossing zero is not sufficient to claim
equivalence.

---

# 11. Phase CR7 — closed-loop hereditary steering

Run only if CR1 passes.

## Cohort

Use:

- 48 completely fresh matrices;
- both candidates;
- six replicate lineages per controller;
- 60 fissions per lineage;
- controller-independent lineage streams;
- controller-action streams domain-separated from simulation;
- complete replay.

## Controllers

1. MODEL_UP:
   after every fission, score all legal one-molecule substitutions and
   apply the largest predicted increase in JOINT_BREAK_RUN3 risk.

2. MODEL_DOWN:
   after every fission, apply the largest predicted decrease.

3. RULE_UP:
   apply the physical catalytic-influence destabilizing rule.

4. RULE_DOWN:
   apply the physical catalytic-influence stabilizing rule.

5. RANDOM:
   one uniformly random legal substitution after every fission.

6. NOOP:
   no edit.

## Outcomes over 60 fissions

Primary:

- inherited-boundary fraction;
- total breaks;
- certified JOINT_BREAK_RUN3 episodes;
- longest inherited run.

Secondary:

- entropy;
- occupied molecular types;
- top-1 abundance share;
- catalytic throughput;
- growth updates per fission;
- survival/extinction;
- cross-lineage compositional similarity;
- number of distinct swaps;
- repeated and reversing swaps;
- out-of-development-envelope fraction.

## Primary steering gates

In both candidates:

1. MODEL_DOWN inheritance > NOOP, lower CI > 0.
2. RULE_DOWN inheritance > NOOP, lower CI > 0.
3. MODEL_UP inheritance < NOOP, upper CI < 0.
4. MODEL_UP episode count > MODEL_DOWN, lower CI > 0.
5. RANDOM equivalent to NOOP.
6. No-op callback is bitwise identical to plain simulation.
7. Replay exact.

Also report the fraction of the MODEL_DOWN gain recovered by RULE_DOWN:

    (inheritance_RULE_DOWN - inheritance_NOOP)
    -------------------------------------------
    (inheritance_MODEL_DOWN - inheritance_NOOP)

Register the external replication hypothesis that RULE_DOWN recovers at
least 80% of MODEL_DOWN’s maintenance gain. Require a matrix-bootstrap lower
bound above 0.70 for a strong replication classification.

## Extension

If the 60-fission stabilization gates pass, extend MODEL_DOWN, RULE_DOWN,
and NOOP for another 60 fissions under continued control.

Do not call this passive persistence. It remains persistence under active
feedback.

---

# 12. Phase CR8 — steer, release, and challenge

Run only if CR7 establishes sustained stabilization.

## Preparation

Generate controller-written states using 60 fissions of:

- MODEL_DOWN;
- RULE_DOWN.

Generate matched natural states using NOOP under the same matrix and lineage
stream structure.

Before release, verify that the written states are genuinely altered:

- lower frozen risk;
- higher inheritance;
- altered entropy/concentration;
- or higher catalytic support.

## Release

Turn every controller off completely.

Run 60 additional untreated fissions.

Track:

- similarity to the written anchor;
- frozen risk;
- inheritance;
- entropy;
- occupied types;
- top-1 share;
- throughput;
- similarity to matched natural states.

## Challenge

At release end, define the release-end composition as the challenge anchor.

For each written and matched natural state, apply:

- no perturbation;
- random k-swap for k in {2,4,8,16};
- one adversarial legal perturbation.

Launch:

- 32 independent 24-fission futures per arm;
- complete replay.

Use the preregistered classifier:

- departure: anchor similarity < 0.7;
- return: anchor similarity > 0.9 for at least three consecutive fissions;
- mode recovery: at least 5 inherited boundaries in the final 6 and top-1
  share >= 0.45;
- categories: held, returned, mode-recovered, lost.

## External hypothesis

The separate clean room found that control maintains but does not install a
self-restoring composition.

For a Codex “written-but-passive” replication require:

1. Written-state anchor similarity falls below 0.7 within the free-release
   horizon in both candidates.
2. Final release inheritance becomes equivalent to matched natural
   inheritance within +/-0.03.
3. Written minus natural held+returned probability is equivalent within
   +/-0.05 at every registered dose.
4. No positive dose-dependent return advantage is observed.
5. Registered basin radius is zero.
6. Exact replay passes.

If Codex instead finds a positive nonzero return basin, report it as a
cross-clean-room disagreement. Do not change thresholds or reinterpret it
to force agreement.

Use the terminology:

    controller-maintained compotype-like state

not:

    installed compotype

unless autonomous release-and-return gates pass.

---

# 13. Phase CR9 — control half-life and minimum feedback rate

Run only after CR7.

## Pulse ladder

Apply MODEL_DOWN for:

    1, 2, 4, 8, 16, 32, or 60 fissions

then release without intervention.

Track post-release:

- anchor similarity;
- frozen risk;
- inheritance;
- entropy;
- top-1 share;
- occupied types;
- throughput.

Define persistence as the first post-release fission at which anchor
similarity falls below 0.7.

Primary hysteresis test:

    Spearman(steering pulse length, post-release persistence)

Require a positive matrix-bootstrap lower bound in both candidates to
confirm accumulating hysteresis.

This is transient persistence, not a restoring basin.

## Periodic control

Apply one MODEL_DOWN edit:

- every fission;
- every 2;
- every 4;
- every 8;
- every 16.

Include edit-budget-matched random controls.

Report inheritance versus intervention budget.

## Event-triggered control

Score frozen risk after every fission and intervene only if predicted risk
exceeds:

    0.15, 0.25, or 0.35

Do not tune thresholds on confirmation outcomes.

Report:

- inherited-boundary fraction;
- edits used per 60 fissions;
- edits per maintained generation;
- threshold excursions;
- comparison with continuous control.

The key question is whether a good risk sensor permits sparse intervention
while the physical corrective action remains simple.

---

# 14. Phase CR10 — exploratory internalization ladder

This phase is exploratory and cannot rescue a failed confirmatory phase.

Run only after CR3 and CR7.

## Policies

L0:
- apply the memoryless RULE_DOWN physical influence substitution after every
  fission.

L1:
- apply RULE_DOWN only immediately after a non-inherited boundary.

L2:
- apply RULE_DOWN only when trailing inherited run length is < 3.

L3:
- distill MODEL_DOWN choices into two depth-3 per-type decision trees:
  one remove-side tree and one add-side tree.
- permitted local per-type features:
  - abundance share;
  - Fable-defined catalytic influence percentile;
  - in-boost percentile;
  - presence.
- train only on Codex development states;
- freeze trees before confirmation;
- do not use confirmation outcomes.

Compare with:

- MODEL_DOWN;
- RANDOM;
- NOOP.

Test:

- 60-fission maintenance;
- recovery after a k=8 substitution at fission 30;
- transfer to the three positive regimes.

Report action frequency separately. A policy that acts every generation
must not be described as informationally superior to a sparse policy merely
because its inheritance is higher.

## Kinetic prototype

As an explicitly labeled model extension, test leave-rate modification:

    leave_rate(type) *=
        1 / (1 + lambda * influence_percentile(type))

for:

    lambda in {0, 0.1, 0.3}

Do not change the frozen baseline simulator paths.

Report this as one retention-only embodiment attempt. A null result does
not establish that no chemical internalization is possible.

---

# 15. Phase advancement and stop rules

The program is serial and bounded.

1. Run CR0 validation and sealing.
2. Run CR1 one-shot model-guided control.
3. Run CR3 physical rule and CR4 beta surgery even if CR1 fails.
4. Run CR2, CR5, and CR6 only when their required upstream gates pass.
5. Run CR7 only if model-guided or physical-rule control passes.
6. Run CR8 and CR9 only if closed-loop stabilization passes.
7. Run CR10 last and label it exploratory.

Do not add new intervention families after seeing confirmation outcomes.

Do not:

- target strict-eight in this program;
- search for a new JOINT_BREAK_RUN3 predictor;
- change the frozen predictor;
- tune edit magnitude on confirmation;
- create candidate-specific rescue protocols;
- pool candidates;
- replace matrices;
- silently drop extinct or adverse lineages;
- treat repeated branches as independent matrices;
- describe a null merely because its CI crosses zero;
- overwrite a failed preregistered prediction.

A phase failure must be reported at full prominence and retained in the
cumulative ledger.

---

# 16. External numerical benchmarks — descriptive only after sealing

Only after each Codex result is sealed may it be compared with the following
external clean-room values.

These are not fitting targets or pass thresholds unless an explicit gate
above says otherwise.

One-shot predictor-guided control:
- up−down approximately +0.156 / +0.158.
- random approximately no-op.

Dose response:
- within-state Spearman approximately 0.475 / 0.471.
- predicted-to-realized slope approximately 0.699 / 0.733.

Simple physical rule:
- rule-up minus rule-down approximately +0.103 in both candidates.

Beta surgery:
- tightening versus loosening separation approximately 0.125 / 0.099 in
  the stabilizing direction.
- norm-matched random surgery approximately null.

Resistance:
- break-up minus break-down approximately +0.145 / +0.147.

Resilience from an identical post-break state:
- renewal-up minus renewal-down approximately +0.026 / +0.027.
- natural renewal was already near ceiling.

Closed-loop control:
- model-down inheritance approximately 0.984 / 0.983.
- no-op approximately 0.891 / 0.887.
- model-up approximately 0.712 / 0.720.
- random approximately no-op.

Sparse feedback:
- event-triggered inheritance approximately 0.974 / 0.977 with about
  16–17 edits per 60 fissions.

Local physical rule:
- continuous L0 recovered approximately 92–96% of the full model’s
  maintenance gain.

Release:
- controller-written states relaxed toward natural behavior after control
  stopped and showed no registered nonzero return basin.

These values are comparison context only. Codex must report compatible,
contract-sensitive, partial, null, or contradictory results honestly.

---

# 17. Required reporting

For every phase, produce:

1. A preregistration or protocol file.
2. Registration ID and source hashes.
3. Seed-domain manifest.
4. Complete validation output.
5. Machine-readable arm/state/matrix/branch tables.
6. Selected-edit tables.
7. Matrix-level effect tables.
8. Bootstrap and randomization arrays or deterministic seeds.
9. Exact replay audit.
10. Full scientific report.
11. Lay summary.
12. Claim-boundary section.
13. Updated cumulative results ledger.
14. Explicit list of:
    - supported claims;
    - failed predictions;
    - deviations;
    - unresolved questions;
    - prohibited interpretations.

The final cumulative conclusion must distinguish:

- model-guided molecular control;
- simple physical-rule control;
- network-level causal sufficiency;
- hereditary resistance versus post-break resilience;
- continuous versus sparse feedback;
- maintained versus autonomously installed organization;
- confirmatory versus exploratory results;
- Codex-specific versus cross-clean-room conclusions.

---

# 18. Claim boundaries

A passing program may support:

> In Codex’s independent GARD reconstruction, small mass-preserving molecular
> substitutions and/or targeted catalytic-network changes causally alter
> JOINT_BREAK_RUN3 probability and hereditary stability under matched common
> random streams.

If model-guided and physical-rule results both pass, it may additionally
support:

> A substantial part of the control law is captured by the catalytic support
> of molecular types currently present in the assembly.

If closed-loop control passes:

> Repeated state-dependent intervention externally maintains distinct
> hereditary modes over many fissions.

If release fails to preserve the state:

> The controller maintains a compotype-like state while active but does not
> install an autonomous compositional attractor.

Do not claim:

- support for Phi or PhiID intervention;
- control of the strict-eight event;
- autonomous agency;
- biological memory;
- error correction;
- a living organism;
- a universal origin-of-life mechanism;
- real prebiotic chemistry;
- independent cross-clean-room replication until the relevant Codex gate
  actually passes.

The central cross-clean-room question is not whether Codex reproduces every
decimal from Fable. It is whether independently reconstructed GARD dynamics
show the same qualitative causal architecture:

    targeted catalytic support changes hereditary stability;
    random changes do not;
    feedback can maintain organization;
    and the maintained state may still fail to become autonomous.
