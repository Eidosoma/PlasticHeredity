# Replication report — heredity break-and-renewal (discovery only)

**Initial replication date:** 2026-08-13.

**Extended adversarial program through:** 2026-08-14.

**Status:** Branch closed after the sealed Phase H occurrence replication.

**Scope:** independent replication of the pre-print's single positive
discovery — the untouched-confirmed, past-observable process-risk
coordinate for heredity break-and-renewal (`JOINT_BREAK_RUN3`) in
two reconstructed GARD simulator candidates. The paper-facing PhiID,
Figure-5 prediction, and intervention analyses were **not** attempted:
the pre-print itself reports them as unsupported.

## Design (independent implementation, paper-scale confirmation)

- GARD kinetics pinned to the historical source
  (`ModelingOriginsofLife/GARD` @ `86dff632`): `Kf=1e-2`, `Kb=1e-4`,
  uniform `rho=1/100`, boost `1 + (βn)/N`; paper parameters `NG=100`,
  `nmin=40`, `nmax=80`, `β=exp(−4+4Z)`, 100 fissions.
- Two candidate contracts: `02` = historical categorical events,
  exact-size fission, equal hypergeometric split, first daughter;
  `03` = vector-Poisson exposure, overshoot, binomial(0.5) fission,
  uniform daughter.
- Target fixed prospectively: within 12 fissions, an inheritance break
  (parent→daughter cosine `H ≤ 0.9`) followed by a run of 3 consecutive
  inherited fissions starting strictly after the break.
- Development (seed domain 1): 40 matrices × 2 candidates × 100
  fissions; 6,956 realized-outcome training rows; four students per
  candidate (prior / direct-9 / beta-only / full = PCA-12 of a
  195-coordinate permutation-invariant graph-state block + 9 direct
  variables, ridge logistic `C=0.1`); frozen
  (`frozen_models.pkl`, SHA-256 `9b58549ed5ee0193…`).
- Untouched confirmation (new 256-bit seed domain): 40 new matrices,
  80 trajectories, 5 restored post-fission landmark states each
  (fissions 20/35/50/65/80 → 400 states), 64 independent F12 branches
  per state in prospectively fixed halves of 32 → **25,600 branches**,
  regenerated exactly in a second campaign.

## Headline result

**The discovery replicates.** All preregistered confirmation gates pass
in both candidates: the F12 process probability is reliably
state-dependent; the frozen full state/graph/history coordinate ranks it
overall and within matrices; it beats direct heredity history on proper
scores with positive whole-matrix bootstrap lower bounds; whole-matrix
permutations fail in the intended direction at the identical minimal
p-value; and the campaign replays exactly.

## Quantitative comparison

| Registered quantity | Paper (cand 02 / 03) | This replication (02 / 03) | Agreement |
|---|---|---|---|
| States in transition region 0.1<q<0.9 | 138/200, 149/200 | 155/200, 166/200 | ✔ |
| Branch-half reliability (Spearman) | 0.938 / 0.924 (95% low 0.903/0.872) | 0.926 / 0.911 (low 0.873/0.851) | ✔ |
| Matrix-centered reliability | 0.625 / 0.606 (low 0.456/0.475) | 0.696 / 0.652 (low 0.550/0.509) | ✔ |
| Full model, overall q Spearman | 0.895–0.918 | 0.872 / 0.867–0.888 | ✔ (slightly lower) |
| Direct history, overall | 0.742–0.822 | 0.717–0.719 / 0.732–0.763 | ✔ |
| Full model, matrix-centered | 0.550–0.697 | 0.625–0.631 / 0.664–0.705 | ✔ |
| Direct history, matrix-centered | 0.198–0.345 | 0.211–0.218 / 0.335–0.377 | ✔ |
| Beta-only, matrix-centered | ≈ 0 | 0.019–0.050 / −0.006–0.005 | ✔ |
| Beta-only, overall | ≈ 0 | **0.44–0.51** | ✖ see note 1 |
| Branch log-loss gain, full vs direct | 0.041–0.052 (min low 0.0259/0.0355) | 0.0423 / 0.0476 (low 0.0284/0.0318) | ✔ |
| q-Brier improvement | 0.012–0.018 (all lows > 0) | 0.0151 / 0.0168 (lows 0.0101/0.0109) | ✔ |
| Whole-matrix permutation p | all 0.001949 | all 0.001949 (0 of 512 exceed) | ✔ exact |
| Exact replay of the campaign | pass | pass (identical SHA-256) | ✔ |

Plasticity decomposition (branch-pooled, conditional on break where the
paper is conditional):

| Process | Paper | Replication (02 / 03) | Agreement |
|---|---|---|---|
| Break within 12 fissions | 0.64–0.73 | 0.46 / 0.51 | direction ✔, level lower (note 2) |
| Resumption given break | 0.88–0.91 | 0.85 / 0.86 | ✔ |
| New 3-fission episode given break | 0.76–0.82 | 0.76 / 0.77 | ✔ |
| Persistent 5-episode given break | 0.53–0.60 | 0.54 / 0.53 | ✔ |
| Old-neighbourhood return (prevalence) | 0.0026–0.0069 | 0.0048 / 0.0045 | ✔ |
| Mean H gain toward old anchor | ≈ −0.26 | −0.19 / −0.19 | direction ✔ (note 3) |
| Positive anchor gain | ~0.08–0.10 | 0.12 / 0.14 | ≈ |
| Repeated return | 0.16–0.24 | 0.03 / 0.05 | ✖ see note 3 |

**Notes on divergences.**

1. *Beta-only overall.* The paper reports beta-only structure as
   approximately uncorrelated even overall; our beta-only student
   transfers real matrix-level propensity (ρ≈0.45–0.51 overall) while
   carrying no within-matrix information (centered ≈ 0), exactly as the
   matrix-propensity decomposition predicts. This is a *stronger*
   beta-only baseline than the paper's, and the discovery survives it:
   the full coordinate still dominates overall and within matrices.
   The paper's near-zero overall beta baseline likely reflects its
   specific beta-feature inventory failing to transfer.
2. *Break level.* Our unconditional break probability is lower
   (0.46–0.51 vs 0.64–0.73). The pre-print's two candidate contracts
   are not published; break level is sensitive to exposure/fission
   semantics. The qualitative regime — breaks common, heavy
   matrix-to-matrix heterogeneity — is preserved.
3. *Gain / repeated return.* Operationalizations for the anchor-gain
   reference point and the "repeated return" cycle are not fully
   prescriptive in the pre-print; ours are registered in `features.py`
   (repeat = a complete second break→3-episode cycle inside the same
   12-fission window, a strict reading). Signs and orderings agree.

## Conclusions supported by this replication

1. Parent→daughter compositional heredity in reconstructed GARD is
   **plastic**: breaks are common, renewal of a new hereditary episode
   after a break is the norm (~0.77 for a 3-episode), and return to the
   old composition is rare (~0.005), with negative mean drift toward
   the old anchor — renewal, not restoration.
2. The empirical F12 probability of `JOINT_BREAK_RUN3` is a real,
   reliably state-dependent quantity (branch-half ρ ≈ 0.92, centered
   ≈ 0.65–0.70).
3. A **frozen, past-observable** coordinate combining current
   composition, catalytic-network-conditioned state, phase, and recent
   heredity history transfers unchanged to untouched matrices in both
   candidates, ranks the process probability overall (ρ ≈ 0.87–0.89)
   and within matrices (ρ ≈ 0.63–0.70), and adds proper-score value
   beyond direct heredity history (log-loss gain lower bounds > 0.028;
   q-Brier lower bounds > 0.010; permutation p = 0.001949).
4. As in the pre-print: this is a simulator-specific, probabilistic,
   non-causal result. It is not PhiID support, not first-replicator
   prediction, and not intervention evidence.

## 5x scale-up (second, seed-firewalled campaign)

A second complete run at 5x scale under fresh entropy domains
(`--tag 5x-2026-08-13`): development 200 matrices x 2 candidates
(35,158 training rows; frozen models SHA-256 `51145445506061c2…`),
confirmation 200 new matrices, 400 trajectories, **2,000 restored
states**, 64 branches each in halves of 32 = **128,000 branches per
campaign**, regenerated exactly (campaign SHA-256 `82f570fb57b3e4a5…`
identical across both campaigns). Results in `results_5x/`.

| Registered quantity | Paper (02 / 03) | 1x replication | 5x replication | Note |
|---|---|---|---|---|
| States in transition region | 69% / 74.5% | 77.5% / 83% | 76.1% / 77.8% | ✔ |
| Branch-half reliability | 0.938 / 0.924 | 0.926 / 0.911 | 0.929 / 0.930 (lows 0.913) | ✔ |
| Centered reliability | 0.625 / 0.606 | 0.696 / 0.652 | 0.698 / 0.683 (lows 0.650/0.631) | ✔ |
| Full overall Spearman | 0.895–0.918 | 0.867–0.888 | 0.883–0.893 | ✔ moved toward paper |
| Direct overall | 0.742–0.822 | 0.717–0.763 | 0.766–0.791 | ✔ inside range |
| Full centered | 0.550–0.697 | 0.625–0.705 | 0.648–0.691 | ✔ inside range |
| Direct centered | 0.198–0.345 | 0.211–0.377 | 0.321–0.379 | ✔ (cand 02 slightly above) |
| Beta-only overall | ≈ 0 | 0.44–0.51 | 0.66–0.69 | ✖ diverges further (note 1) |
| Beta-only centered | ≈ 0 | ≈ 0 | −0.010–0.007 | ✔ |
| Log-loss gain full-vs-direct | 0.041–0.052 | 0.042 / 0.048 | 0.041 / 0.036 (lows 0.035/0.029) | ✔ |
| q-Brier gain | 0.012–0.018 | 0.015 / 0.017 | 0.014 / 0.012 (lows 0.012/0.010) | ✔ |
| Permutation p | all 0.001949 | all 0.001949 | all 0.001949 | ✔ exact |
| Replay | pass | pass | pass | ✔ |
| Mean anchor gain | ≈ −0.26 | −0.19 | −0.20 / −0.20 | direction ✔ |
| Episode-3 given break | 0.76–0.82 | 0.76 / 0.77 | 0.74 / 0.75 | ≈ (marginally below) |
| Persist-5 given break | 0.53–0.60 | 0.54 / 0.53 | 0.50 / 0.50 | ≈ (marginally below) |

Reading: every preregistered gate passes again at 5x with materially
tighter bootstrap lower bounds (e.g., reliability lower bound 0.913 vs
0.851–0.873 at 1x; log-loss gain lower bounds 0.029–0.035). The full
model's overall rank moved toward the paper's range with the larger
development cohort, supporting the interpretation that the residual 1x
shortfall was training-base size, not a pipeline difference. The one
divergence that *grew* is the beta-only overall baseline (0.66–0.69):
with 200 development matrices the beta-only student transfers matrix
propensity even better, while still carrying zero within-matrix
information. This strengthens the note-1 conclusion: the paper's ≈0
overall beta-only baseline appears to be a weakness of its beta feature
inventory rather than a property of the simulator. The discovery's core
contrast — full state dominates direct history within matrices
(0.65–0.69 vs 0.32–0.38) — is unchanged.

## 25x scale-up (third, seed-firewalled campaign)

A third complete run at 25x the original scale, fresh entropy domains
(`--tag 25x-2026-08-13`): development 1,000 matrices x 2 candidates
(175,642 training rows; 3 of 2,000 trajectories died too early to
contribute; frozen models SHA-256 `4f817180b310f627…`), confirmation
1,000 new matrices, 2,000 trajectories, **9,999 restored states** (one
trajectory died before fission 80), 64 branches each = **~640,000
branches per campaign**, regenerated exactly (campaign SHA-256
`38d61f4dd888634b…` identical). Results in `results_25x/`. Runtime:
dev 160 s; confirmation 2 x ~23 min; 12 workers.

| Registered quantity | Paper (02 / 03) | 1x | 5x | 25x | Note |
|---|---|---|---|---|---|
| States in transition region | 69% / 74.5% | 77.5% / 83% | 76.1% / 77.8% | 76.7% / 78.9% | ✔ stable |
| Branch-half reliability | 0.938 / 0.924 | 0.926 / 0.911 | 0.929 / 0.930 | 0.931 / 0.931 (lows 0.925/0.924) | ✔ |
| Centered reliability | 0.625 / 0.606 | 0.696 / 0.652 | 0.698 / 0.683 | 0.675 / 0.691 (lows 0.652/0.669) | ✔ |
| Full overall Spearman | 0.895–0.918 | 0.867–0.888 | 0.883–0.893 | **0.896 / 0.892–0.893** | ✔ cand 02 inside range; cand 03 within 0.003 |
| Direct overall | 0.742–0.822 | 0.717–0.763 | 0.766–0.791 | 0.763–0.765 / 0.760 | ✔ inside |
| Full centered | 0.550–0.697 | 0.625–0.705 | 0.648–0.691 | 0.664–0.668 / 0.671 | ✔ inside |
| Direct centered | 0.198–0.345 | 0.211–0.377 | 0.321–0.379 | 0.309–0.310 / 0.282–0.292 | ✔ fully inside |
| Beta-only overall | ≈ 0 | 0.44–0.51 | 0.66–0.69 | 0.72–0.73 / 0.72–0.73 | ✖ grows with dev scale (note 1) |
| Beta-only centered | ≈ 0 | ≈ 0 | ≈ 0 | 0.004 / 0.004 | ✔ exactly zero state-local info |
| Log-loss gain full-vs-direct | 0.041–0.052 | 0.042/0.048 | 0.041/0.036 | 0.0435 / 0.0442 (lows 0.0404/0.0410) | ✔ squarely inside |
| q-Brier gain | 0.012–0.018 | 0.015/0.017 | 0.014/0.012 | 0.0158 / 0.0163 (lows 0.0146/0.0151) | ✔ inside |
| Permutation p | all 0.001949 | same | same | all 0.001949 | ✔ exact |
| Replay | pass | pass | pass | pass | ✔ |
| Mean anchor gain | ≈ −0.26 | −0.19 | −0.20 | −0.20 / −0.20 | direction ✔ |
| Old-composition return | 0.0026–0.0069 | 0.005 | 0.004–0.005 | 0.0037 / 0.0052 | ✔ |
| Episode-3 given break | 0.76–0.82 | 0.76/0.77 | 0.74/0.75 | 0.74 / 0.75 | ≈ marginally below |
| Persist-5 given break | 0.53–0.60 | 0.54/0.53 | 0.50/0.50 | 0.50 / 0.50 | ≈ marginally below |

**Convergence reading.** Across 1x → 5x → 25x, the frozen full model's
overall rank converged monotonically into the paper's reported range
(0.867–0.888 → 0.883–0.893 → 0.892–0.896), direct-history centered rank
settled fully inside the paper's range, and the proper-score gains
stabilized in the middle of the published intervals with lower bounds
now an order of magnitude above zero (log-loss gain lows 0.040+). This
is the pattern expected if the pipeline is the same computation as the
paper's and residual differences at small scale were sampling and
training-base noise. The beta-only overall baseline continued to grow
(0.72–0.73) while its matrix-centered rank remained exactly ~0.004 —
static beta structure predicts matrix-level propensity increasingly
well with more development matrices but contains no state-local
information at any scale, sharpening the note-1 divergence against the
paper's ≈0 overall beta claim.

## Plain-language summary: what the ablation means for the paper

The paper studies a simulated chemical soup that grows into an assembly,
splits, and grows again. Sometimes the "child" closely resembles its
parent (heredity works); sometimes it does not (heredity breaks). The
discovery: from a snapshot of the present state alone, one can predict
the odds that heredity will soon break and then re-establish itself in a
new form — like a forecaster giving rain odds without tracking every
cloud. The paper further claims the predictive power comes from a
specific place: how the assembly's current contents interact with the
catalytic "rulebook" (the beta network) it lives under.

The reviewer's worry, in plain terms, was that the fancy predictor might
be winning for boring reasons: some inputs were fed to the model twice
under different names (double counting), and the fancy model received
"clock" variables the simple baseline never saw (hidden extra clues).
If so, the headline sentence would collapse into "a model with more
history variables beats one with fewer" — true but trivial.

The ablation suite measured each boring explanation directly, on fresh
untouched data:

- **Double counting: real but harmless.** The duplicated variables
  exist (the reviewer read the code correctly, and the methods should
  say so), but deliberately feeding the model twelve duplicated columns
  changed held-out predictions by less than one ten-thousandth.
  Duplication changes internal bookkeeping, not what the model knows.
- **Hidden clues: ruled out.** This independent rebuild never contained
  the clock variables at all, and the advantage appeared at full
  strength anyway.
- **Where the power actually lives.** Splitting the inputs into "the
  raw ingredient list" versus "how those ingredients interact with the
  catalytic rulebook": the ingredient list alone added only a sliver
  (+0.012); the rulebook-conditioned block alone reproduced the entire
  advantage (+0.043).

So the paper's core mechanistic sentence — network-conditioned current
state predicts beyond direct history — has moved from "plausible
interpretation" to "measured, with every boring alternative measured at
zero, on untouched data." The remaining concessions (document the
duplicate variable, rerun the suite on the original code, narrow the
beta-only claim to within-matrix) are wording and hygiene fixes, not
threats to the finding.

**Second round, in the same plain terms.** The second review targeted a
side claim — that heredity has "memory" (streaks are real, not a coin
with fixed bias) — by showing the measuring instrument was miscalibrated:
it graded the two competing explanations on slightly different portions
of the data. We built the scale both ways and weighed the same data on
each. The tilt is real (on artificial memory-free data the tilted scale
invents an effect; the fair one reads zero) but on the real data it
accounts for only ~1.5% of the measurement — and measured fairly, the
memory effect is two to three times *larger* than the paper claimed. We
also found a flaw the reviewer missed: weighing sequences from many
soups together lets "this soup runs hot" masquerade as memory; even
after subtracting that, the effect remains large. Separately, we rebuilt
the prediction machine with every criticized redundant part removed
(registry v2) and tested it on completely fresh soups: it predicted just
as well in one simulator and slightly better in the other — the
scaffolding was holding nothing up. Every criticism so far has, when
measured, left the discovery unchanged or stronger, and it now exists in
a cleaner form than the paper originally described.

## Ablation suite (reviewer response)

A reviewer of the original code identified potential confounds in the
full-vs-direct comparison: duplicated mass/time directions inside and
across the PCA and direct blocks, ridge coefficient-splitting on
duplicated columns, an exact identity `fissionsSinceLatestBreak ≡
trailingInheritanceRun`, and two growth-clock variables present only in
the full block. A frozen ablation suite (`run_ablation.py`) was trained
on the regenerated 25x development cohort and evaluated twice: as a
diagnostic on the existing 25x confirmation, and on a **fresh untouched
cohort** (new entropy tag, 200 matrices, 2,000 states, 128,000
branches), since the suite was designed after seeing earlier results.

Students (all + scaled direct-9 where noted, ridge C=0.1): direct;
direct-unique (duplicate variable removed); dup-control (direct + 12
exact duplicate mass/generation columns — ridge-geometry negative
control); beta-matrix; state-only (PCA-12 of the 53 pure-composition
coordinates + direct-9); beta-cond (PCA-12 of the 142 beta-conditioned
coordinates + direct-9); full (registered model).

Fresh untouched cohort (cand 02 / 03; gain = branch log-loss gain vs
direct, 2,048 matrix-bootstrap lower bounds):

| Student | Overall q rank | Centered q rank | Log-loss gain (low) |
|---|---|---|---|
| direct | 0.759–0.773 / 0.756–0.765 | 0.283–0.344 / 0.319–0.325 | 0 (baseline) |
| direct-unique | identical to direct | identical | ±0.0000 |
| dup-control | identical to direct | identical | ±0.0000 |
| beta-matrix | 0.690–0.693 / 0.688–0.691 | ≈ 0 | −0.013 / −0.009 |
| state-only | 0.795–0.802 / 0.789–0.794 | 0.399–0.442 / 0.417–0.419 | +0.012 (+0.007) / +0.012 (+0.007) |
| **beta-cond** | **0.892–0.897 / 0.890–0.898** | **0.639–0.684 / 0.675–0.716** | **+0.043 (+0.036) / +0.043 (+0.037)** |
| full | 0.885–0.888 / 0.880–0.885 | 0.632–0.662 / 0.652–0.684 | +0.040 (+0.032) / +0.040 (+0.033) |

Findings:

1. **The reviewer's redundancy identities are real and inherent to the
   registered definitions**: `fissionsSinceLatestBreak ≡
   trailingInheritanceRun` holds exactly on all ~175k development rows
   in both candidates, and `regimeDuration ≡ trailingRun` whenever the
   current boundary is inherited. The effective direct baseline has ~7
   distinct variables, not 9.
2. **The duplication/ridge-geometry concern is empirically inert**:
   removing the duplicate (direct-unique) and adding 12 exact duplicate
   columns (dup-control) both leave held-out log loss unchanged to four
   decimals. Coefficient-splitting changes penalties, not held-out
   information.
3. **The clock confound cannot explain the effect**: this replication's
   195 block contains no generation/step/phase clocks at all (only mass
   is shared with the direct block), and the full-vs-direct gain
   appears at full published magnitude.
4. **The mechanistic claim is now positively isolated**: pure
   composition (state-only) adds only +0.012 beyond direct history,
   while the beta-conditioned state block alone adds +0.043 and
   reproduces (slightly exceeds) the full model on every metric —
   overall, matrix-centered, and proper scores — on a fresh untouched
   cohort. The signal is specifically **catalytic-network-conditioned
   current state**, exactly as the manuscript's Discussion asserts.
5. Incidentally, the 53 raw-composition coordinates are dead weight in
   the PCA: beta-cond ≥ full everywhere.

Caveat: these results certify the *reconstructed* pipeline. The
original implementation does contain the two growth-clock variables;
this suite shows they are unnecessary for the effect, not that they are
inert in the original's specific coefficients. Running this same suite
on the original pipeline (and a fresh cohort) remains the right
correction there.

## Markov-vs-IID, corrected (reviewer #2 response)

The reviewer showed the original fits the IID inheritance model on all
symbols (first symbols and singletons included) but scores it only on
transition destinations, biasing the reported 0.015–0.022
bits/transition Markov gain. This analysis had never been implemented
in the clean room; `markov_iid.py` + `run_markov.py` now implement it
both ways on the regenerated 5x confirmation branches (post-break F12
suffixes; 30,213 / 32,472 suffixes; 188k / 205k transitions), with
identical Jeffreys smoothing and identical two-way matrix cross-fitting
for both models, whole-matrix bootstraps, and two null calibrations.

| Quantity (cand 02 / 03) | Value |
|---|---|
| First-symbol vs destination inheritance rate | 0.685 vs 0.798 / 0.719 vs 0.805 |
| Biased gain (as published), pooled | +0.0623 [+0.0545,+0.0714] / +0.0405 [+0.0348,+0.0472] bits |
| **Corrected gain (reviewer's fix), pooled** | **+0.0614 [+0.0514,+0.0728] / +0.0399 [+0.0331,+0.0479] bits** |
| Macro (equal-sequence) corrected | +0.0580 / +0.0370 bits |
| Support-mismatch bias (biased − corrected) | ≈ +0.0009 / +0.0006 bits |
| Stationary IID null (both estimators) | +0.0045 / +0.0036 bits |
| Nonstationary no-Markov null, biased | +0.0058 / +0.0044 bits |
| Nonstationary no-Markov null, corrected | +0.0049 / +0.0038 bits |

Synthetic validation: on homogeneous sequences the estimators behave
exactly as theory predicts — stationary IID null 0.0000 for both; a
nonstationary no-dependence null yields +0.0084 bits for the biased
pipeline and 0.0000 for the corrected one, demonstrating the reviewer's
mechanism in isolation.

Findings:

1. **The bug is real and its direction is as predicted** — first
   post-break symbols are inheritance-poorer than destinations
   (0.69–0.72 vs 0.80) — but in this data its magnitude is only
   ~0.001 bits (~1.5% of the gain), because suffixes average 7.2
   symbols and the contamination is diluted.
2. **The substantive claim survives strongly**: the corrected
   Markov-over-IID gain is +0.040/+0.061 bits per transition, 2–3x the
   originally published 0.015–0.022, with bootstrap CIs far from zero.
3. **A new confound the reviewer did not raise**: with pooled
   cross-fitting across heterogeneous matrices, BOTH estimators show a
   ~+0.004-bit gain on a purely IID null, because the previous symbol
   proxies matrix identity (between-matrix variance masquerades as
   temporal dependence). The honest net within-matrix dependence signal
   is therefore ≈ +0.036 / +0.056 bits — still large. Reruns of the
   original should either fit per-matrix models or report this null
   floor alongside the gain.
4. Caveat: these are clean-room candidates and branch suffixes; the
   original's exact 0.015–0.022 remains implementation-specific and
   still requires the reviewer's rerun on the original pipeline. What
   the clean room establishes is that "dependence beyond IID" is a
   robust property of the process, not an artifact of the support bug.

## Registry v2 (reviewer #1 fix)

Registry v2 removes every duplication the reviewer's category of
concern applies to: **PCA-12 of the 142 beta-conditioned coordinates
(no mass, no raw-composition block) + the 8 unique direct variables**
(`fissionsSinceLatestBreak` dropped as an exact duplicate of
`trailingInheritanceRun`; `regimeDuration` retained and its conditional
identity documented). Trained on the regenerated 25x development
cohort, frozen (`results_v2/frozen_models_v2.pkl`, SHA-256
`0420cb49a2077bd2…`), and confirmed on a **fresh untouched cohort**
(tag `v2-conf-2026-08-13`, 200 matrices, 2,000 states, 128,000
branches).

| Model (cand 02 / 03) | Overall q rank | Centered | Log-loss gain vs direct (low) |
|---|---|---|---|
| direct (9 vars) | 0.793–0.796 / 0.764–0.770 | 0.291–0.313 / 0.327–0.329 | baseline |
| direct-8 (unique) | identical to direct | identical | ±0.0000 |
| v1 full (195+9) | 0.904–0.911 / 0.895–0.899 | 0.668–0.672 / 0.676–0.683 | +0.0433 (+0.0369) / +0.0451 (+0.0390) |
| **v2 (142+8)** | **0.906–0.909 / 0.903–0.905** | **0.669–0.687 / 0.692–0.694** | **+0.0434 (+0.0372) / +0.0476 (+0.0412)** |

v2 − v1 log-loss gain: +0.0001 [−0.0013,+0.0015] (02) and +0.0025
[+0.0006,+0.0044] (03). Reliability on the fresh cohort 0.940 / 0.926;
v2 permutation p = 0.001949 in both candidates.

**Verdict: the preregistered v2 gate passes.** The deduplicated,
smaller coordinate matches v1 in candidate 02 and slightly exceeds it
in candidate 03, at zero cost from dropping the duplicate direct
variable. Registry v2 is now the recommended frozen coordinate: every
redundancy the reviewer flagged is structurally absent, and the
mechanistic content (network-conditioned state + unique history) is all
that remains.

## Hardening: validation suite, sensitivity, calibration, regimes

### Formal validation suite

`test_validation.py` (self-contained, 14 checks, all passing) now
enshrines: 512 exact propensity fixtures against the closed-form GARD
equation; chi-square of the categorical event sampler; Poisson-exposure
moments; hypergeometric and binomial fission-law moments with exact
mass conservation; trajectory invariants (exact-size fission for
candidate 02, bounded overshoot for 03, H range); strict-threshold
semantics; seed-domain separation, spawn-key distinctness, and bitwise
replay; a subset replay against the frozen 1x campaign artifact;
`JOINT_BREAK_RUN3` and process-outcome unit cases (including the
documented registered semantics that anchor departures count from the
fission after the breaking fission's daughter); Markov/IID estimator
null calibration; the registered direct-variable identities; and
feature-provenance consistency. A small behavior-preserving refactor
exposed `event_rates` and `_sample_categorical` for testing; the 1x
confirmation campaign regenerated under the refactor with a
byte-identical SHA-256 (`c6cfebce…`).

Feature provenance is now typed metadata
(`features.GRAPH_STATE_PROVENANCE`, `DIRECT9_PROVENANCE`):
`COMP_IDX`/`BETA_IDX` are derived from it, with a test asserting the
derived sets equal the frozen literals.

### Target-definition sensitivity (registered grid)

The v2 confirmation cohort's branches were regenerated with raw
per-fission H capture (`results_sensitivity/v2_cohort.pkl`, also
persisting the cohort) and the joint break-then-run event re-scored
across H_thr ∈ {0.85, 0.90, 0.95} × run ∈ {2,3,4} × horizon ∈
{8,10,12} (horizon 16 is not evaluable from 12-fission branches; the
registered primary 0.90/3/12 reproduced the stored v2 numbers exactly,
0.9075/0.9037).

- **The frozen v2 coordinate beats direct-8 on matrix-centered rank at
  all 27/27 grid points in both candidates.** v2 centered spans
  0.39–0.73 (02) and 0.44–0.74 (03) across the entire grid; direct-8
  spans 0.17–0.37.
- Within-matrix ranking peaks at the registered threshold (0.64–0.74
  at H>0.90) and remains clearly positive at H>0.85 (0.45–0.58) and
  H>0.95 (0.39–0.52).
- Overall rank is high at H>0.85–0.90 (0.74–0.92) but drops at H>0.95
  (0.23–0.49): the strictest threshold reorders matrix-level
  propensity, while the state-local signal survives. The discovery's
  within-matrix content is not a knife-edge artifact of H>0.9.
- Split-half q reliability is 0.68–0.95 over the whole grid: the
  process probability is measurable at every examined definition.

### Calibration

At the registered target, the frozen v2 10-bin reliability curve tracks
the identity line closely across the full range, with a mild
underestimate in the top bin only; an isotonic overlay shows minimal
correction (`results_sensitivity/figures/fig_calibration_v2.png`).
The earlier "informative but imperfect" statement can be upgraded:
v2 is near-calibrated on untouched matrices.

### Permutation upgrade

Whole-matrix permutations rerun at 4,096: p = 0.000244 (0 of 4,096
exceed) for v2 on its fresh cohort and for v1-full on the 25x cohort,
both candidates — the previous 0.001949 values were floor-limited, not
marginal.

### Parameter-regime probe (20 dev + 20 conf matrices per regime)

| Regime (A, σ) | break | q reliability | frozen-v2 centered | matched-v2 centered | direct-8 centered |
|---|---|---|---|---|---|
| (−4, 3) | 0.94 | 0.83 / 0.80 | 0.01 / −0.03 | −0.01 / 0.10 | 0.05 / 0.08 |
| (−4, 5) | 0.26–0.30 | 0.76 / 0.82 | **0.36 / 0.58** | 0.30 / 0.21 | 0.12 / 0.08 |
| (−3, 4) | 0.46–0.50 | 0.89 / 0.87 | **0.40 / 0.66** | 0.25 / 0.41 | −0.03 / 0.21 |
| (−5, 4) | 0.61 | 0.81 / 0.84 | **0.49 / 0.56** | 0.36 / 0.50 | 0.18 / 0.21 |

- **The frozen main-regime v2 transfers zero-shot to three of four
  perturbed regimes**, beating both direct-8 and the small
  regime-matched retrains (whose 20-matrix training base is too small —
  consistent with the earlier scale-convergence finding).
- **The σ=3 regime is a genuine phenomenon boundary**: with weak
  catalytic heterogeneity, heredity breaks almost always (0.94) and no
  coordinate — frozen, matched, or direct — ranks the residual
  state-local variation, even though q itself is still reliable
  (0.80–0.83). Heredity break-and-renewal as characterized here
  requires sufficient catalytic-network structure; this sharpens the
  manuscript's scope statement rather than weakening the discovery.

## Measured vs interpreted (claim-discipline note, reviewer #7)

What `JOINT_BREAK_RUN3` certifies: an inheritance break followed by
three consecutive fissions each with strict parent→daughter cosine
`H > 0.9`. It does NOT certify that those three daughters occupy one
compositional neighbourhood (adjacent similarity is not transitive, and
the growth phase between fissions is unconstrained by the flags), nor
recurrence, nor persistence beyond three fissions. Throughout this
report the measured object is therefore a **break-and-renewal event** /
**new three-fission hereditary episode**; "plastic-heredity regime
switching" is the pre-print's proposed dynamical interpretation, and —
per the section below — its coherence component is now measured and
does NOT hold at the 3-fission scale.

## Episode coherence (reviewer #7): registered upgrade gates FAILED

A coherence/distinctness criterion was registered prospectively (plan
of 2026-08-13, before any outcome was seen): coherence = episode span
similarity `H(d_u, d_{u+2}) > 0.9`; distinctness = `H(d_u, anchor) <
0.9`; upgrade gates (a) `P(coherent|joint) ≥ 0.8`, (b) frozen v2 beats
direct-8 on the coherent target with positive bootstrap lower bound,
(c) coherent-target reliability ≥ 0.7. Measured on the regenerated
v2-conf cohort (trajectory tier: 859/940 episodes; branch tier: the
full 128,000-branch campaign, with the joint-target q verified
identical to the persisted cohort).

| Quantity (cand 02 / 03) | Value |
|---|---|
| Episode span similarity, median | 0.720 / 0.728 (traj); fraction > 0.9: 0.058 / 0.067 |
| Growth-phase drift H(d_u, p_{u+1}), median | 0.890 / 0.896 |
| Episode start → old anchor, median | 0.503 / 0.533 |
| P(coherent given joint event) | **0.060 / 0.072** — gate (a) FAIL |
| P(distinct given coherent) | 0.936 / 0.921 |
| Coherent-target prevalence / reliability | 0.020, 0.320 / 0.026, 0.274 — gate (c) FAIL |
| Frozen v2 vs direct-8 on coherent target (centered) | 0.196 vs 0.083 / 0.214 vs 0.123, diff lower95 +0.059/+0.044 — gate (b) pass |

**Verdict: 2 of 3 registered gates fail in both candidates; per the
prospective rule, the "regime switching" language is demoted
permanently.** The reviewer's concern was empirically load-bearing, not
merely semantic: renewal episodes certified by adjacent inheritance
drift substantially across even three fissions (median span 0.72), and
the growth phase alone typically consumes the adjacent threshold's
allowance (median single-cycle drift 0.89 < 0.9). Heredity in this
simulator is **temporally local**: what each fission transmits
faithfully is nonetheless reshaped on the episode scale.

Two things survive and sharpen:

1. **"Plastic" stands.** Episodes start far from the old anchor (median
   similarity ~0.5; 92–94% of coherent episodes are also distinct) —
   renewal genuinely happens elsewhere, consistent with the rare
   old-return and negative anchor-gain results.
2. **This is evidence FOR the pre-print's H3** (the relevant invariant
   is hereditary capacity, not molecular identity): the system
   maintains the *capacity* for high-fidelity parent→daughter
   transmission while the transmitted composition itself drifts.
   "Break-and-renewal of hereditary capacity" is the precise claim the
   data support.

Incidentally the frozen coordinate still ranks even the rare coherent
sub-event better than direct history (gate b passed) — but with
prevalence 0.02 and reliability ~0.3 that is a weak signal, reported
for completeness, not leaned on.

## From prediction to control (preregistered intervention; H5)

The reviewer's proposed decisive test — and the pre-print's H5 — was
run with all gates preregistered (plan of 2026-08-13, before any
outcome): on a **fresh cohort** (tag `intervention-2026-08-13`, 40 new
matrices × 2 candidates, 400 landmark states), mass-preserving
single-molecule **swap edits** were selected by the FROZEN v2
coordinate (registered screening: marginal adds/removes → top-10 each
direction → exact scoring of ~200 swap combinations; no refitting).
Four arms per state — score-raising, score-lowering, unedited, random
swap — each received 64 futures under **common random streams** (common random
numbers: the arm is absent from the branch spawn key).

### Phase A — paired extremes (all gates PASS, both candidates)

| Quantity (cand 02 / 03) | Value |
|---|---|
| Arm means (up / noop / random / down) | 0.395 / 0.334 / 0.334 / 0.239 · 0.440 / 0.383 / 0.390 / 0.281 |
| **Paired up−down** | **+0.156 [0.125, 0.189] / +0.158 [0.130, 0.186]** — G1 pass |
| Ordering up > noop > down | pass / pass — G2 |
| Random−noop (specificity) | −0.0001 [−0.010, +0.010] / +0.007 [−0.006, +0.019] — G3 pass |
| Predicted vs realized shift | +0.230 vs +0.156 / +0.229 vs +0.158 (shrinkage ≈ 0.7, as expected) |
| Component decomposition (up−down) | break +0.117 / +0.121; renewal-given-break +0.162 / +0.148 |
| Noop-arm rank check (free re-confirmation) | Spearman 0.870 / 0.904 on 40 fresh matrices |

The single most important row is the **random-swap control: a random
one-molecule swap does nothing** (CIs straddle zero), while the
score-selected swap of identical magnitude moves the branch-measured
probability by ~0.16. The coordinate is not merely sitting on states
that any perturbation would move — it identifies causally special
molecular directions. The edits move both halves of the process: they
raise the break hazard and the subsequent renewal propensity together.

### Phase B — dose–response (all gates PASS, both candidates)

Six swaps per state spanning the screened predicted-shift range
(extreme arms shared with Phase A; interior arms simulated with the
same CRN streams):

- Mean within-state Spearman between predicted Δp̂ and realized Δq:
  **0.475 [0.402, 0.545] / 0.471 [0.419, 0.525]** — B1 pass.
- Pooled slope of realized on predicted shift: **0.699 [0.620, 0.768]
  / 0.733 [0.661, 0.795]** — B2 pass.

The knob is graded: intermediate predicted shifts produce intermediate
realized shifts, with attenuation ≈ 0.7 (selection shrinkage plus
frozen-model calibration error — quantified, not hidden).

### Registered interpretation boundary

This establishes: *molecular swap edits selected by the frozen
past-observable coordinate causally shift the branch-measured
break-and-renewal probability in the predicted direction and in graded
proportion, under matched futures, in this simulator.* It does not
rescue the original paper's Φ-directed interventions (different scorer;
that reconstruction failed and stays failed), and it is not chemistry.
Within those bounds, the pre-print's H5 is now supported: the
coordinate is a control knob for break-and-renewal, not only a
forecast.

## Phase C: transfer, mechanism, and closed-loop steering

All gates preregistered (plan of 2026-08-13) before any outcome.

### C2 — Knob transfer and the predicted null

Phase-A protocol re-run zero-shot (frozen home-regime v2 scorer) in
four perturbed regimes (20 fresh matrices each, 4 arms × 48 CRN
branches):

| Regime (A, σ) | up−down (02 / 03) | Registered gate | Result |
|---|---|---|---|
| (−4, 5) | +0.062 [+0.029,+0.102] / +0.053 [+0.021,+0.087] | transfer | **pass** (03 specificity blemish: random−noop +0.021, CI excludes 0) |
| (−3, 4) | +0.144 [+0.102,+0.188] / +0.170 [+0.110,+0.226] | transfer | **pass** |
| (−5, 4) | +0.170 [+0.117,+0.218] / +0.173 [+0.126,+0.220] | transfer | **pass** |
| (−4, 3) | +0.024 [−0.009,+0.064] / +0.007 [−0.032,+0.051] | **NULL predicted** | **null confirmed** |

The knob turns in every regime where the coordinate carries state-local
signal and does nothing where it doesn't — a preregistered null landing
exactly on target.

### C3 — What the knob is, physically

Exploratory sweep over all ~80,000 screened swaps (correlations of
predicted shift with per-type physical quantities): strongest add-side
correlate = the added type's **outgoing catalytic influence**
(ρ = −0.24; remove-side mirror +0.13; remove-count +0.40). Frozen rule
(recorded before any confirmatory branch ran): *swap the most
catalytically influential present molecule for the least influential
one* (rule-down mirrors). Confirmatory campaign on the home cohort
(48 CRN branches, matched spawn keys):

- rule up−down: **+0.103 [+0.077,+0.131] / +0.103 [+0.081,+0.129]** —
  direction gate pass, both candidates;
- efficiency vs the full frozen model: **0.66 / 0.65**.

Two-thirds of the model knob compresses into one interpretable scalar:
the control mechanism is **loosening or tightening the assembly's
catalytic web**. This is the symbolic/mechanistic compression the
pre-print's E02 plan listed as missing validation.

### C1 — Closed-loop steering (all gates PASS, both candidates)

Controllers {model_up, model_down, noop, random}; one frozen-v2-selected
swap per fission for 60 generations; 24 fresh matrices × 6 lineages
per controller; CRN initial streams; noop lineages verified bitwise
against plain trajectories.

| Outcome per 60 fissions (02 / 03) | model_up | noop | random | model_down |
|---|---|---|---|---|
| Break-and-renewal episodes | 3.72 / 3.94 | 2.88 / 3.11 | 2.95 / 3.15 | 0.67 / 0.76 |
| Breaks | 17.3 / 16.8 | 6.6 / 6.8 | 6.8 / 7.0 | 0.97 / 1.05 |
| Inheritance fraction | 0.712 / 0.720 | 0.891 / 0.887 | 0.887 / 0.883 | **0.984 / 0.983** |
| Longest inherited run | 19.9 / 19.1 | 33.2 / 32.3 | 33.5 / 31.2 | **57.5 / 56.3** |

Gates: episodes up−down **+3.05 [+2.27,+3.73] / +3.19 [+2.39,+3.83]**
(G1 pass); ordering up > noop > down (G2 pass); random−noop CIs include
0 (G3 pass). Sixty targeted single swaps transform the dynamics —
down-steering pins a lineage in near-perfect heredity for essentially
the whole horizon (longest run 57/60 vs 33 baseline), up-steering
forces continual plastic renewal — while sixty random swaps change
nothing.

**Relation to the original's Figure 6 / Table 1.** This is the
properly-controlled version of that protocol (repeated per-fission
edits changing replicator persistence), which did not reproduce under
the Φ scorer. The claim *shape* is hereby vindicated: a scalar signal
computed from present state can causally control hereditary persistence
over long horizons in GARD — but the working signal is the
break-and-renewal coordinate, not Φ, and the demonstration requires
the random-edit control that the original lacked.

### Registered boundary (unchanged)

Simulator-only; frozen scorer; not chemistry; no rescue of Φ-directed
claims.

## Phase D: paper-facing checks (reviewer's five items)

### D0 — Inference units and random streams (documentation)

**Resampling unit:** the catalytic matrix, in every bootstrap in this
report — states, arms, and lineages are carried as within-matrix blocks
(`run_ablation.evaluate`'s idx-map bootstrap; `run_intervention
.boot_lower`; per-matrix collapse of the six lineages in
`run_steering`). **Random streams:** intervention arms share per-branch
seeds but, once an edit changes event counts, later draws
desynchronize; arms therefore experience **common random streams
(CRN)**, not identical realized futures — wording corrected
throughout. Spawn-key domains: 0 beta, 1 initial state, 2 trajectory,
3 confirmation branches, 5 intervention branches, 6 random-arm
selection, 7 steering lineages, 8 steering random-controller edits.
Per-(fission, purpose) keying is noted as the stronger design for any
future campaign; frozen campaigns are not retrofitted.

### D1 — Registered outcome suite: the knob decomposed (all consistency asserts pass)

Six separately registered outcomes, up−down paired effects
(cand 02 / 03, 1,024 matrix bootstraps):

| Outcome | up−down effect |
|---|---|
| Break hazard | +0.117 [+0.091,+0.144] / +0.121 [+0.095,+0.145] |
| Run-3 after break (given break) | +0.162 [+0.133,+0.192] / +0.148 [+0.120,+0.180] |
| Persist-5 after renewal (given episode) | +0.143 [+0.101,+0.183] / +0.148 [+0.106,+0.187] |
| Inherited boundaries (of 12) | −0.410 / −0.437 |
| Survival | 1.000 in every arm — no effect |
| Growth updates per fission | −6.07 / −1.51 (down-arm grows with MORE updates) |

Identity-exact midpoint decomposition of the joint effect
(q = b·r; residual ≤ 1e-16): **break-hazard share 0.543 / 0.569,
renewal share 0.457 / 0.431.** Per the registered >50% criterion the
reviewer's conjecture is adjudicated in their favor — the paper-facing
framing becomes *the knob primarily controls hereditary stability, with
renewal quality moving alongside* — while the shares show it is a
54/46 split, not a dominance: renewal-given-break and five-fission
persistence move as strongly as the break hazard.

### D2 — Long-run cost panel: all 8 registered predictions PASS

Final-10-fission means under 60-generation steering
(cand 02 / 03; consistency with stored steering results asserted):

| Metric | model_down | noop | model_up |
|---|---|---|---|
| Composition entropy | 1.59 / 1.59 | 2.24 / 2.26 | 2.62 / 2.64 |
| Occupied species | 10.4 / 10.4 | 16.6 / 16.9 | 20.5 / 20.7 |
| Top-1 share | 0.50 / 0.50 | 0.34 / 0.34 | 0.24 / 0.23 |
| Catalytic throughput | 34.9 / 35.9 | 25.0 / 25.6 | 22.4 / 23.2 |
| Cross-lineage similarity | 0.87 / 0.81 | 0.54 / 0.54 | 0.50 / 0.51 |
| Extinctions | 0 | 0 | 0 |

Horizon extension: down-steered lineages hold inheritance fraction
**0.996 / 0.995** over fissions 61–120 with zero extinction.

**Answer to the "degenerate victory" question: down-steering does not
freeze the assembly — it manufactures a compotype.** The stabilized
state is concentrated (half the mass in one species), *more*
catalytically active than baseline (throughput +40%, more growth
updates per fission), convergent across independent replicate lineages
(similarity 0.87 vs 0.54), and persistent far beyond the steering
horizon. The cost of near-perfect heredity is compositional diversity —
which is precisely what this model calls a self-replicator. Up-steering
buys diversity (entropy, occupied species up) at the cost of stability.

### D3 — Controller action audit (characterization)

- **Adaptive, not repetitive:** 36–43 distinct swaps per 59-edit
  lineage; consecutive-repeat rate ≈ 0.04; cycling (undoing a recent
  swap) ≈ 0.000–0.005. The controller is not a broken record and does
  not fight itself.
- **Off-manifold exposure is moderate and honest:** ~10–12% of noop/up
  states and ~20% of down-steered states fall outside the natural-state
  PCA envelope — the manufactured compotype is somewhat outside the
  training distribution, reported as such.
- **Model-vs-rule agreement is directional, not literal:** the model's
  ADD choices match the frozen physical rule 63–66% of the time when
  stabilizing (add high-influence types), and its REMOVE choices match
  ~50% when destabilizing; exact-swap agreement is low (0.1–8%). The
  model agrees with the physics at the component level while choosing
  different specific molecules — consistent with the rule capturing
  ~65% of the model's effect.

## Phase E: steer–release–challenge — the written state is NOT an attractor

The reviewer's proposed basin test, preregistered with four candidate
verdicts and a recorded prediction (finite-basin attractor). **The
registered prediction was wrong; the verdict in both candidates is
"written-but-passive" — and only barely that.** This is the program's
second registered-prediction failure (after episode coherence), and it
completes the story rather than damaging it.

Design (registered): regenerate 96 controller-written high-heredity
states (model_down, 60 fissions); score them with the frozen v2
(precursor); release into free dynamics for 60 fissions; challenge once
(none / random-k, k ∈ {2,4,8,16} / adversarial swap); 32 free branches
× 24 fissions per arm; identical protocol on 96 matched natural states;
four-outcome classification (held / returned / mode-recovered / lost);
matrix-level bootstraps.

Results (cand 02 / 03):

- **Precursor:** written states start genuinely stabilized — frozen-v2
  risk 0.112/0.127 vs ~0.33–0.38 for natural states.
- **Release: the written composition evaporates in ~5–10 fissions.**
  Anchor similarity decays from ~0.95 to the natural drift floor
  (~0.55) almost as fast as natural states drift from their own
  anchors; composition-hold at 60 fissions is only 0.31/0.29;
  release-phase inheritance (0.909/0.928) is barely above natural
  (0.908/0.909). Mode-survival 0.60/0.56 clears the registered
  controller-maintained threshold — the state does not collapse
  instantly — but the stabilization residue is thin and transient.
- **Challenge: no basin.** Written vs natural (held + returned)
  differences straddle zero at every dose, outcomes are essentially
  dose-independent (by challenge time the drift has already left the
  anchor behind), and the basin radius is k = 0. No composition
  return, no mode-attractor signature above natural baseline.

**Correction propagated to Phase D2's framing:** the "persists to 120
fissions" result there was persistence *under continued control* (the
controller remained active through the extension). Phase E shows the
same state without the controller relaxes to baseline within tens of
fissions. The accurate sentence is: *the controller manufactures and
MAINTAINS a compotype-like state; it does not install one.*

**The unifying conclusion of the whole program, now complete and
consistent across every experiment:** natural episodes drift (coherence
span 0.72), retrospectively-identified attractors do not transfer
(the original L36–L37), written states evaporate on release (~5–10
fissions), and control works exactly as long as it is applied
(steering). In this chemistry there are no stable destinations —
neither found nor engineered. Heredity and organization are
**maintained processes, not places**. The knob is real, causal, graded,
transferable, and physically interpretable — but it is a steering
wheel, not a programmer. The reviewer's question ("holding the wheel,
or taught the system a new destination?") has a measured answer:
holding the wheel.

## Phase F: cross-clock attractor adjudication (final round)

The reviewer's Phase F design, preregistered in full (equivalence
margins, decision table, leading hypothesis), adjudicated what kind of
attractor GARD contains. Their code (Kahana–Segev–Lancet, Cell Rep.
Phys. Sci. 2023) proved to be the SAME historical GARD10 source our
candidate 02 was validated against; MATLAB-only, so the registered
feasibility gate resolved to protocol-reimplementation (near-exact).
All Phase F campaigns are seed-tagged; results in `results_f/`.

### F1 — the Kahana attractor protocol reproduces
94–97% of runs from random initial compositions converge to a
composome (similarity ≥ 0.9, median 4–5 generations) in their
configuration AND our frozen candidates. Each generation shows the
cycle: fission displaces composome similarity −0.030; growth restores
+0.033 (CIs excluding 0; below our registered +0.05 margin — reported
without adjustment) with R_Q (composition–flux alignment, their exact
formula) rising +0.14–0.16 within growth.

### F2 — the written state is composome-adjacent, and EVERYTHING drifts
Controller-written states sit nearly on the composome manifold
(atlas distance 0.08/0.09 vs 0.05/0.06 for composome members, 0.15–0.17
for ordinary states) — super-concentrated, lowest-risk versions of
natural composomes; flux-misalignment is only partial (R_Q 0.79/0.85 vs
composomes' 0.92/0.93). Decisive: one-step drift is ~+0.01/fission AWAY
from the atlas for every class — including natural composome members.

### F3/F4 — no basin on the cross-generation clock, at any delay or dose
Challenge at release delays {0, 1, 2, 5, 10, 60}: held+returned is
flat (~0.31–0.38) across ALL delays and ALL perturbation arms
(none ≈ k4 ≈ k16 ≈ fission) — no transient basin, and no
dose-dependence (the signature of basin absence). The continuous
restoring-force estimator (≈50k pairs/candidate):
**cross-generation +0.021/+0.022 per fission AWAY (margin-fail);
within-growth −0.023/−0.021 per cycle TOWARD (margin-PASS)**. Fission
perturbations heal substantially within one growth phase (Kahana's
perturbation, their result); by horizon 10 every perturbation type sits
on the same drift floor.

### F5a — assay certified on the Singh–Jain bistable positive control
Implementation quantitatively faithful (interdivision times 0.286/0.077
vs published 0.269/0.075; inheritance 0.95; both modes, 138 switches).
Gate repair documented openly (G4 constructed three times; all
iterations preserved in `f5_sj_results.json`/`f5_sj_revision.json`).
Final criteria all pass: assay hold matches true residence
(0.79/0.42 vs 0.78/0.46), small doses absorbed, large doses switch
(0.44 → 0.18). The assay shows dose-dependence exactly where real
basins exist — and showed none in GARD. Documented lesson: raw cosine
is blind to minority-species basins (0.9996 between S-J modes).

### F5b — "lost" retired: multistable redistribution
~30% of challenge branches end in the same composome, **13–22% are
captured by a DIFFERENT composome** (at t=60, cand 02: alt 0.22 >
same 0.19), ~57% are mid-transit (below threshold during the
within-growth dip). Kahana's multistability confirmed in-regime;
Phase E's "lost" was substantially transit + rival capture.

### F6 — the one true invariant: a distributional attractor
No pullback/random attractor (shared streams do not synchronize;
diffs +0.015/−0.033, CIs spanning 0). But lineages started from
DISJOINT compositions under independent noise converge to the same
MIXTURE of composomes: occupancy TV distance 0.25 within-matrix vs
0.55 across matrices (both candidates). **The persistent object in
GARD is the catalytic matrix's stationary distribution over composomes
— identity belongs to the rulebook; assemblies are transient samples.**

### Negative control (−4, 3)
Composome structure essentially absent (non-drift fraction 0.06);
the naive within-growth gain metric is uninterpretable there
(documented metric caveat).

### F7 — the attractor-aware controller: the objective was not the problem
Six controllers (v2-down, R_Q-only, composome-only, joint
(1−risk)+R_Q+atlas-sim, random, noop) steered 60 fissions then
released. The joint controller wrote the highest composome alignment
(atlas-sim 0.94); v2-down incidentally maximized R_Q best of all
(0.84/0.86 — minimizing break-risk ≈ maximizing flux alignment). On
release, EVERY controller's state decays to the same drift floor by
+10 fissions; the registered aware-vs-v2down comparison failed in both
candidates (−0.022 / +0.044, CIs spanning 0). Registered prediction
confirmed: **"steering wheel, not programmer" is maximally
strengthened — no writable objective installs persistence.**

### Decision-table adjudication (reviewer's table, frozen verbatim)
Rows that fired: **"No contradiction: phase-conditioned growth
attractor, but no fixed generational memory"** (primary);
**"dynamic/limit-cycle attractor, not a fixed composition"** (the
within-growth tube); **"multistability and basin switching"** (F5b);
**"maintained process without detectable basin"** on the
cross-generation clock (with certified positive controls). Rows
refuted: model/regime dependence; perturbation-anisotropic basin (the
anisotropy is between CLOCKS, not perturbation types); "natural
composomes return where written fail" (both drift equally);
"attractors can be deliberately written" (F7).

### Final ontology (the reviewer's expected resolution, confirmed and extended)
Attraction toward a phase-conditioned composome set during growth;
stochastic displacement at fission (near-exact cancellation:
+0.033 vs −0.030); local parent–daughter heredity; slow cross-
generation drift (+0.02/fission) that no controller objective can pin;
multistable redistribution among composomes; and one persistent
invariant — the matrix-level stationary mixture over composomes.
**The chemistry contains local restoring flows and a distributional
identity without containing a permanent self.** For origin-of-life
framing: early heredity as recurring growth-cycle reconstruction —
identity as a per-generation computation whose only durable record
lives in the environment's catalytic network.

## Novelty search (2026-08-13, post-Phase-E scope)

Web-based novelty check covering the full discovery as it now stands.
Component-wise prior art exists for: GARD compositional inheritance and
compotypes (Segré/Lancet lineage); selection response of compotypes and
serial-transfer evolution of autocatalytic sets (Markovitch & Lancet
2012; Hordijk & Steel 2014); the GARD evolvability debate (Vasas et al.
PNAS 2010 vs later replies); return-to-attractor claims for reproducing
compositions (Kahana & Lancet, Cell Rep. Phys. Sci. 2023 — abstract
level; in tension with our Phase E within our tested regime and
criteria); dynamic kinetic stability as a concept (Pross); generic
ML early-warning-signal detectors for critical transitions; committor
learning for self-organization (Jung et al. 2023); RL/optimal control
of colloidal self-assembly via external fields; minority-control
heredity in catalytic networks (Kaneko lineage); and the target
Φ-knob claim itself (Pigozzi & Levin, arXiv:2607.28250; no independent
replication found).

No prior work was found combining: (1) a frozen past-observable
state/graph/history coordinate predicting a prospectively defined
heredity break-and-renewal probability on untouched matrices; (2)
single-molecule mass-preserving scored swaps with random-edit controls
and common random streams establishing graded causal dose-response on
that probability; (3) zero-shot knob transfer across regimes with a
preregistered confirmed null at the phenomenon boundary; (4) symbolic
compression of the knob to one physical scalar (outgoing catalytic
influence) retaining ~65% efficacy; (5) properly-controlled closed-loop
per-fission steering of hereditary persistence; and (6) a preregistered
steer-release-challenge protocol with natural controls and a dose
ladder measuring basin radius = 0 for engineered states. The negative
search is not proof of priority; one key comparator (Kahana & Lancet
2023) was verified only at abstract level (paywalled), and differently
worded or unpublished work may be missed.

## Phase G: what is stabilized, what it costs, and what the dial is

The reviewer's final bounded program, preregistered in full in
**`PHASE_G.md`** (question, design, registered predictions, gates,
margins, and seed domains 20–24 written to the repo BEFORE any module
ran; per-module RESULTS appended below the frozen preregistration,
never edited above it). All campaigns seed-tagged; results in
`results_g/`.

### G1 — no competency attractor: behavior relaxes to matrix-typical
The behavioral fingerprint K(s) (8 z-frozen components: break hazards
at 4/8/12, joint risk, persistence, inherited boundaries, growth
updates, Δentropy) was tracked after perturbation for five state
classes. The registered two-target race was unambiguous: for written
states, distance to the state's OWN K(0) grows monotonically
(e.g. cand 02 model-down, none arm: 1.61 → 2.75 by t=10) while
distance to the MATRIX-TYPICAL fingerprint falls (2.31 → 1.93); the
competency-return margin fired nowhere. The assay itself is valid:
the same estimator on the Singh–Jain bistable control shows
within-basin behavioral return in 8/9 lineages (gate PASS). Registered
prediction confirmed: **what relaxes is not just composition but
behavior — toward the matrix's typical repertoire, not the engineered
state's.** Even "how it acts" is not a persistent self.

### G2 — resistance and resilience are separately steerable dials;
### the explore–consolidate controller is an informative failure
Stage A (dedicated frozen break-hazard student q_B, identical-state
arms): break-up vs break-down moves realized 6-fission break hazard by
**+0.145 [0.110, 0.182] (02) / +0.147 [0.088, 0.204] (03)**;
random ≈ noop (CIs span 0). Stage B (the clean resilience design —
the IDENTICAL post-break daughter restored across arms, frozen q_R
student): renewal-up vs renewal-down moves run3-within-8 by
**+0.026 [0.011, 0.044] / +0.027 [0.009, 0.048]** against a
near-ceiling baseline (noop 0.97/0.93), and shortens time-to-renewal
(3.35 vs 4.10 fissions). Both registered gates PASS: the chemistry has
two independently addressable dials. Stage C (explore → recover →
consolidate state machine) FAILED its registered success (miss #4 in
the ledger): staged produced FEWER distinct episodes than noop
(1.23 vs 2.60 — consolidation suppresses the natural episode turnover)
while winning on breaks (1.5 vs explore's 14.9) and on certified-
episode survival (46 vs 8.5 fissions). The failure is informative:
**stability and novelty compete for the same lineage; this controller
architecture cannot buy both at once.**

### G3 — accumulating hysteresis (registered miss #5), and the price
### of a thermostat
Pulse ladder (steer P fissions, then release): post-release half-life
of the engineered state GROWS with pulse length — 5.6 → 15.5 fissions
(02), 5.2 → 13.4 (03), saturating around P≈16; registered
accumulation test Spearman **+0.391 [+0.289, +0.492] /
+0.369 [+0.259, +0.490]**. We registered "no accumulation"; the
chemistry disagreed. This revises Phase E: the written state is still
not an attractor, but holding it longer packs it tighter and it
decays SLOWER — a graded, contentless memory of how deeply it was
written. Periodic control degrades gracefully (inheritance 0.982 at
every-fission edits → 0.899 at every-16; budget-matched random arms
flat at drift floor). Event-triggered control (edit only when frozen
risk > θ): at θ=0.35, **inheritance 0.974/0.977 with ~16–17 edits per
60 fissions — one informed correction per ~3.5 generations** is the
minimum feedback rate that maintains heredity (the
controller-information rate of the process).

### G4 — the mechanism tournament: web-tightness is the dial,
### causally, from both directions
Disagreement mining isolated edits where the frozen v2 model, the
one-scalar influence rule, and R_Q (flux alignment) make OPPOSING
predictions (4 classes, ~100 covariate-matched edits each, noise
floors registered). Only class A reached decision in both candidates
(realized Δq −0.0294 [−0.0592, −0.0005] / −0.0244 [−0.0478, −0.0043]):
v2 and the influence rule called it correctly, **R_Q called it wrong —
flux alignment is eliminated as the driver** (consistent with F7).
Transplantation: for cand 02 the same molecular edit under a foreign
matrix follows the LOCAL matrix's v2 prediction (0.71) far better than
its native effect (0.47); cand 03 tied (0.58/0.58) — partial but
directionally consistent evidence that the unit of mechanism is
relational (molecule × network), not molecular. The decisive stroke,
beta surgery: at FIXED composition, norm-matched edits to the
catalytic table that raise vs lower outgoing influence move realized
break-risk by **−0.125 [−0.208, −0.049] / −0.099 [−0.172, −0.023]**,
while Frobenius-matched random table edits do nothing (−0.016/−0.003).
Combined with every molecule-swap result: **catalytic-web tightness is
causally sufficient at both ends — change the web through the
molecules, or change it behind the molecules; heredity follows the
web.**

### G5 — internalization (exploratory): the policy is one scalar;
### the ladder is anti-monotone (registered miss #6)
Information-restricted policies vs the full v2 controller and noop
(maintenance over 60 fissions; k8 recovery; three transfer regimes).
**L0 — the memoryless one-scalar influence rule applied every
generation — recovers 92%/96% of the full model's maintenance gain**
(0.976/0.980 vs v2's 0.983/0.984 vs noop 0.900/0.889), matches it on
post-shock recovery (0.992 vs 0.990/0.995), and transfers to all
three regimes as well as v2 does (at (−5,4): 0.975/0.979 vs
0.982/0.975, noop 0.807/0.826). The registered prediction — a
monotone ladder L3 ≥ L2 ≥ L1 ≥ L0 — was wrong in the most
interesting direction: **L0 > L3 (0.68/0.64 of v2's gain, and less
portable) > L2 (0.42/0.51) > L1 (0.22/0.42)**. Event-gating on crude
local cues (a break happened; streak < 3) forfeits most of the gain,
while G3 showed the model-based risk trigger keeps 0.974 with ~16
edits — synthesis: **the knowledge of WHAT to do is one locally
computable scalar; what the trained model still owns is WHEN.** The
kinetic prototype (leave rates damped by influence percentile, no
editor; frozen sim untouched, λ=0 reproduces baseline) was a null at
the registered λ grid (all CIs span 0) — the policy is trivial to
state but must be enacted as discrete compositional edits; the hard
part of internalization is the actuator, not the knowledge.
Deviations and full tables in `PHASE_G.md`.

### Phase G synthesis and the complete prediction-miss ledger
Phase G closes the program with all four pieces a mature result
needs: the phenomenon (heredity as per-generation reconstruction),
the mechanism (catalytic-web tightness, causally sufficient from both
directions, flux-alignment eliminated), the control theory (two
independent dials; dose–response; hysteresis; a minimum feedback rate
of ~0.27 informed edits/generation; a one-scalar internalizable
policy), and the limits (no competency attractor, no permanent self,
no basin on the generational clock, stability–novelty tradeoff,
embodiment-as-kinetics null). Registered predictions that the
chemistry refused, reported as required:

| # | Where | Registered prediction | What happened |
|---|---|---|---|
| 1 | Regimes | "regime"-style language would survive | JOINT_BREAK_RUN3 is a graded process-risk, not a regime |
| 2 | Coherence | episode-coherence upgrade gates would pass | FAILED; reported as boundary |
| 3 | Phase E | written states might hold under low dose | basin radius = 0; wheel, not destination |
| 4 | G2 Stage C | staged controller beats noop on distinct episodes | consolidation suppresses novelty |
| 5 | G3 | sparse-sufficient, no accumulation | accumulating hysteresis (5.5 → ~14 fissions) |
| 6 | G5 | ladder monotone in information; kinetic helps | anti-monotone (L0 best); kinetic null |
| 7 | Phase I addendum | code-faithful Φ_R also null under heredity manipulation | robust positive coupling — implemented Φ-r is a responsive signature of hereditary organization |
| 8 | Phase K1 | Φ_R dose–response monotone in both candidates | graded in 03 only; unresolved in 02 at 24-matrix power (K2's registered atom lean also half-wrong: downward causation, not synergy, carries the signal) |
| 9 | Phase M | printed structure reads SR episodes; Φ_R reads only slow organization | inverted — Φ_R reads BOTH the episodes and the organization; the printed structure is the weaker episode-reader |
| 10 | Phase N | unbundled atoms and Φ-volatility carry weak foresight (CSD lean) | all null at validity-gated power; Φ's predictive content is matrix identity only — while the gen-clock printed structure unexpectedly works as a gauge |
| 11 | G3 (scope, via G3-ADJ) | sealed G3's hysteresis implicitly general across lineage ages | a nascent-lineage property: strong in fresh launches (+0.44..+0.57), attenuated in evolved ones (+0.14..+0.32) — resolving the cross-lab disagreement as a moderator |

## Phase H: independent occurrence replication of the strict coherent
## eight-fission episode (external clean-room cross-check)

A second independent agent ("Codex"), replicating the same pre-print
in its own clean room, reported a rare operational event: a break
followed by a distinct, mutually coherent eight-fission hereditary
episode, at ~1.81–2.11% of futures. Phase H tested whether OUR frozen
implementation generates the same event under the IDENTICAL frozen
operational definition — an occurrence experiment only (no
prediction, no intervention, no model fitting; no Codex code or data
used; the four benchmark rates entered only a post-seal descriptive
comparison). Registered event name: **STRICT_BREAK_COHERENT8_DISTINCT**.

Preregistration `STRICT8_REPLICATION_PREREGISTRATION.md`, endpoint
fixtures `test_strict8_endpoint.py` (14/14 required categories PASS),
and the sealed analysis script were hashed into
`results_strict8_occurrence/SEAL.json` BEFORE any scientific matrix
was generated; a rate-blind smoke run checked I/O and replay only.
Cohort: domain 25, fresh tag; 200 new matrices shared across both
candidates; 5 natural landmarks each (20/35/50/65/80); 128 F32
futures per restored state; prospective branch halves 0–63 / 64–127;
256,000 futures plus a complete second deterministic replay.

**Result: CONCLUSION A — phenomenon and rate numerically compatible
with the external clean room.** All 2,000 nominal states realized
(no main-path extinction losses); exact replay across all 256,000
futures (campaign hashes equal). Four primary cells (rate,
whole-matrix 95% CI, events, event-bearing matrices of 200; external
benchmark in brackets):

| cell | rate | 95% CI | events | matrices | external |
|---|---|---|---|---|---|
| 02/A | 0.01703 | [0.01256, 0.02231] | 1,090 | 121 | [0.01869] |
| 02/B | 0.01698 | [0.01278, 0.02190] | 1,087 | 127 | [0.01809] |
| 03/A | 0.02000 | [0.01522, 0.02523] | 1,280 | 125 | [0.02089] |
| 03/B | 0.01883 | [0.01413, 0.02405] | 1,205 | 125 | [0.02109] |

Every cell's CI contains its external rate; the registered gate
passes in full (all rates positive, all bootstrap lower bounds > 0,
exact replay, no rescue of any kind). The candidate ordering
reproduces too (03 > 02 in both clean rooms), and matrix breadth
matches (121–127 event-bearing matrices vs the external 119–143):
the event is broadly distributed, not a few-matrix artifact.
Descriptively, per-state event probability agrees across independent
branch halves at r ≈ 0.84/0.86 — the event rate behaves like a
stable property of the restored state (descriptive only, as
registered). Component decomposition: ~66–69% of futures break;
~50–52% then produce eight consecutive inherited boundaries;
episode-wide pairwise coherence is the sharp filter (the FIRST
eligible window passes it in only 0.17–0.26% of futures — most
positives certify at a LATER window, i.e. coherence emerges after
the post-break composition settles); median timing: break at
fission 6, qualifying run start 17, certification 24.

Registered claim boundary (verbatim): a passing result supports only
that an independent Fable clean-room GARD implementation
prospectively generates rare break-followed-by-distinct-coherent
eight-fission hereditary episodes under the frozen operational
definition. It does not establish an attractor, recurrence,
autonomous return, permanent stability, a compotype, memory,
individuality, prediction, causal control, or real chemistry.
Restored states, branch seeds, windows, and component outcomes are
archived (`strict8_units.pkl`) for a later, separately registered
study.

## Phase I: reciprocal causal bridge — reconstructed Φ-r and plastic
## heredity are independent control axes

Prediction and control are different properties: a coordinate can
fail to forecast and still supply a control gradient. Phase I asked,
in one six-arm campaign with an identical CRN action panel at every
decision point (preregistered in `PHIR_BRIDGE.md`; frozen
reconstructed Φ-r in `phir.py`, built from the target paper's own
Methods — normalized whole-minus-parts predictive information on
CLR-transformed per-update composition series, minimum-information
bipartition, Gaussian), whether the paper's causal-emergence
coordinate and our heredity coordinate move each other causally.
Registered boundary: this does not test Φ-r's replicator-onset
prediction and cannot adjudicate the authors' unavailable
implementation.

**Result: the "independent organizational axes" row of the frozen
interpretation table fired.** The campaign is valid (v2 stab-vs-
destab swings inherited fraction by +0.185/+0.174, CIs excluding 0;
random ≈ noop everywhere). But: (T1) the largest heredity
manipulation we can produce — 2 vs 13 breaks per 60 fissions —
leaves realized Φ-r unmoved in both candidates (+0.005 [−0.005,
+0.016] / −0.010 [−0.024,+0.005]); realized Φ-r is in fact slightly
NEGATIVE everywhere (−0.01 to −0.03) — no positive causal emergence
in the steady-heredity regime under this reconstruction. (T2) the
Φ-r-surrogate controller moves heredity decisively in candidate 03
(inherit +0.033 [+0.008,+0.060]; breaks −2.0; longest run +5.8, all
CIs excluding 0) but only directionally in 02, failing the
both-candidates gate — and the registered manipulation check shows
the surrogate does NOT move realized Φ-r itself, so that heredity
effect belongs to the surrogate's linearized-flow structure, not to
Φ-r. (T3) the two scorers choose nearly independent actions
(Spearman 0.07/0.09; top-choice 10–11% vs 4% chance). Full tables,
scorecard, and one pre-campaign amendment + erratum note in
`PHIR_BRIDGE.md`; suite 27/27 (three new Φ-r fixtures).

### Phase I addendum: the authors' code computes a different Φ-r —
### and THAT one responds

Inspection of the authors' public repository (pigozzif/PhiRL)
revealed their implemented Φ-r is not the printed formula: it is
Mediano et al.'s revised Φ_R — the nine ΦID atoms with synergy on
either side plus both cross-transfers (identity proven from first
principles) — unnormalized, on the two MACRO-AVERAGED halves of a
spectral minimum-information bipartition, with pointwise-MMI local
Gaussian estimation. We ported it exactly (`phir_code.py`; equality
with their code verified to ~1e-14 on all 16 atoms, bit-identical MI
matrices, 50/50 bipartition agreement), registered an addendum, and
replayed the ENTIRE frozen campaign byte-identically (replay gate:
0/576 mismatches) with both instruments. **Result: T1 PASSES under
the authors' definition, both candidates** — stabilizing heredity
raises implemented Φ_R by +0.208 [+0.134, +0.279] /
+0.256 [+0.121, +0.392] (levels strongly positive, 0.97–1.91;
random ≈ noop). The printed formula and the implemented one are
nearly uncorrelated across lineages (r ≈ 0.09/0.13). Revised
adjudication: **the implemented Φ-r is a responsive signature of
hereditary organization — causally downstream of the heredity dial —
while the printed formula is decoupled; the reverse direction
(Φ-r as controller) remains untested for lack of a working
code-Φ_R-steering rule.** Registered miss #7: we leaned null, and
the sign (organization, not transition) refutes the original Phase I
lean as well. The text-vs-code divergence itself (own-future vs
whole-future parts, the redundancy repair, normalization) is
documented for the field in `PHIR_BRIDGE.md`. [Erratum 2026-08-18:
close-up inspection of the typeset page shows the printed formula
is UNNORMALIZED — the denominator in our reading was a
text-extraction artifact (the "Where I(X_t, X_{t+1})…" definition
line stacking under the display equation). Measured retroactively
via the stored Phase K atom decomposition, the true typeset formula
is ALSO null under the heredity manipulation (stab − destab
−0.006 [−0.031, +0.018] / −0.015 [−0.042, +0.011]); no conclusion
changes — both the typeset and extracted readings are unresponsive,
and only the code's Φ_R responds. Full three-reading taxonomy and
adjudication in `PHIR_BRIDGE.md`.]

## Phase J: the signature confirms prospectively; the gauge is not
## a controller (2× scale, fresh matrices)

Phase J (`PHIR_CONFIRM.md`, domain 28, sealed with source hashes
before the campaign; 48 fresh matrices × 2 candidates × 2 reps ×
6 arms; suite 30/30) closed the two gaps the addendum left. **C1 —
prospective confirmation: PASS both candidates.** On matrices no
analysis had ever touched, stabilizing heredity raises implemented
Φ_R by +0.155 [+0.075, +0.233] / +0.178 [+0.103, +0.259] — the
addendum's re-measurement discovery is now a fully preregistered,
prospectively confirmed effect. **C2 — the reverse direction:** a
probe-rollout controller steering by the actual implemented Φ_R
(no surrogate; 2-fission CRN probes over a shared 12-swap panel)
moved the gauge weakly and only reliably in candidate 03 (C2a
+0.065 [+0.006, +0.123]; 02 ns), and **heredity did not follow
anywhere** (C2b null on all co-primaries in both candidates —
informative in 03, where the gauge demonstrably moved). Candidate 03
thus realizes the frozen table's registered pattern in full:
**responsive signature, not a controller — confirmed
prospectively.** A striking corollary: even where Φ_R-steering
works, it moves Φ_R only ~1/3 as far as the heredity dial does —
the best causal-emergence controller found is the heredity
controller. One honest deviation: C4 specificity was violated in a
single cell (candidate 03 heredity, random − noop
−0.017 [−0.033, −0.003]) — at 2× power a small real cost of random
per-fission editing becomes detectable; the primary contrasts are
between equal-edit-budget arms and are untouched, and the
program-wide "random ≈ noop" background claim is now
scale-qualified. Registered predictions: C1 hit, C2b's null hit
(the signature hypothesis survived its designed refutation), C2a
half-right.

## Phase K: dose, decomposition, prediction, robustness — the
## signature is downward causation

Phase K (`PHIR_DOSE.md`, domain 29, sealed; suite 31/31) completed
chapter 5 with four registered pieces. **K1 (dose–response):**
heredity's own dose curve passes decisively in both candidates
(signed-dose Spearman +0.584/+0.569); Φ_R's dose curve is graded in
candidate 03 (+0.274 [+0.107, +0.427]) but unresolved in 02
(+0.080 [−0.038, +0.203]) — ledger miss #8, partial. **K2 (the
headline):** replaying Phase J byte-exactly (gate: 0 mismatches)
with the full 16-atom decomposition shows the heredity response is
carried by the **downward-causation atoms (synergy→unique:
+0.577/+0.504, both CIs excluding 0)** while pure
synergy-persistence responds NEGATIVELY (−0.384/−0.298) and
part-to-part transfers fall: organized assemblies route information
top-down; destabilized ones churn it sideways. The authors' own
alternative summary "emergence" (synergy + causation) responds in
both candidates with the tightest intervals of any quantity tested
(+0.193 [+0.146, +0.247] / +0.207 [+0.144, +0.272]). **K3 (natural
prediction; both registered predictions hit):** Φ_R predicts future
breaks at the MATRIX level only (overall −0.330/−0.271, CIs
excluding 0; matrix-centered null) — the signature reads present
organization and carries no detectable state-level foresight; v2
dominates overall; the centered comparison is underpowered by
design (2 lineages/matrix) for all predictors and does not revise
the core replication's state-level transfer. **K4 (robustness):**
sign preserved in all six variant×candidate cells; aggregation and
the paper-text drop-last fix are immaterial; removing CLR shrinks
the effect in candidate 02 — the magnitude's dependence on the GARD
paper's own registered CLR preprocessing is disclosed prominently.

## Phase L: the paper-faithful instrument — the page, implemented
## verbatim, contains no heredity coupling

Phase L (`PHIR_PAPER.md`, sealed; suite 32/32) implemented the
typeset Methods page exactly as printed — unnormalized formula,
MULTIVARIATE bipartition blocks (the page never macro-averages),
CLR + drop-last, no z-scoring — and measured it on byte-exact
replays of Phase J's four arms (replay gate 0/768). Levels are
strongly negative everywhere (−8.9 to −11.0), as registered. The
coupling question fails coherently: the primary-window contrast
passes marginally in candidate 02 (+1.26 [+0.08, +2.43]) but is
null in 03, REVERSES SIGN in 02's own secondary window, is negative
in 03's secondary window, and the specificity control is violated
in 03 (random − noop −2.12 [−3.04, −1.26] — random editing moves
the instrument more than informed editing does). With near-zero
correlation to the stable code instrument (r ≈ 0.18/0.09), this is
an estimator dominated by 198-dimensional Gaussian-MI noise, not a
physical coupling. **The chapter's instrument map is now complete:
extracted reading — null; typeset-on-macro — null; typeset
multivariate (paper-faithful) — no coherent coupling; the authors'
implemented Φ_R — responsive, prospectively confirmed, dose-graded,
carried by downward causation. The plastic-heredity connection
exists in exactly one reading of Φ-r: the one the authors
implemented, not any reading of the one they printed.**

## Phase M: does the emergence gauge read the "self-replicating"
## state? In our chemistry, yes — sharpening a cross-agent mystery

Motivated by the paper-replication agent's mirror finding (in their
GARD rebuild the PRINTED formula reads the self-replication state,
78/100 vs the paper's 73/100, while PhiRL's Φ_R fails at 31/100 —
the exact inverse of our intervention results), Phase M
(`PHIR_SR.md`, sealed; suite 33/33) ran BOTH comparison styles on
identical data: byte-exact Phase J replays (gate 0/768), sliding
3-generation windows, four instruments, SR = inside an inherited run
of length ≥ 5. Results: SR occupancy tracks the heredity dial
(0.89/0.90 stabilized vs 0.68/0.63 destabilized — "self-replicating"
and "high heredity" are operational kin); the consistency gate
reproduces every established chapter result on the same windows; and
**within-lineage, Φ_R READS the SR state in both candidates
(ΔSR +0.099 [+0.055, +0.149] / +0.138 [+0.094, +0.188])** while the
printed structure is the weaker reader (null in 02) and synergy is
NEGATIVE during SR. Our registered "different variance components"
resolution is refuted (ledger miss #9): in this chemistry Φ_R reads
both the slow organization and the fast episodes. The named user
hypothesis (detection stronger under strong heredity) is supported
in candidate 03, null in 02, with a registered-caveat ceiling effect
(stabilized arms are ~90% SR, starving the contrast). Consequence
for the mystery: the other agent's GARD failure of Φ_R cannot be an
intrinsic property of the formula — the remaining suspects are the
SR detector definition (our high-occupancy "unbroken run" vs their
rarer locked-compotype episodes), their GARD configuration, or which
formula actually produced the paper's figures — now precise
questions for the authors.

## Phase N: the foresight round — gauge proven, predictor closed,
## and the clock was load-bearing all along

Phase N (`PHIR_FORESIGHT.md`, domain 30, sealed; suite 34/34)
repaired every weakness of the earlier prediction test and added the
sharpest one. **N1** (16 fresh matrices × 12 lineages each — real
within-world power): the registered validity gate passes decisively
(v2 matrix-centered Spearman +0.428 [+0.271, +0.558] /
+0.289 [+0.165, +0.414]), and against that standard EVERY Φ variant
is centered-null in both candidates — the scalar, the unbundled
components (causation, emergence), the printed structure,
Φ-volatility, Φ-trend, and both generational-clock variants — with
all residual-on-v2 correlations null too. The matrix-level channel
replicates (high-Φ worlds break less). **N2** (event-locked early
warning, 303/316 pre-break events vs matched deep-run controls): no
instrument shifts in the currently-inherited generations
immediately before a break — the critical-slowing-down lean missed
(ledger miss #10, with the causation/emergence and volatility leans).
**N3** (replay gate 0 mismatches): the generational-clock Φ_R is a
responsive gauge (+0.059/+0.053), and — bonus finding — **on the
generational substrate even the PRINTED structure becomes a working
gauge in both candidates (+0.026 [+0.004, +0.049] /
+0.039 [+0.018, +0.060])**: the printed formula's chapter-long
deadness was partly a wrong-clock problem, not purely a formula
problem. Final adjudication: Φ is a robust GAUGE of present
hereditary organization across formulas, clocks, and comparison
styles — and a predictor of nothing beyond which world it is in;
every foresight door this program could construct is now closed.

## G3-ADJ: the hysteresis disagreement resolved — depth-of-writing
## memory belongs to nascent lineages

The external lab's G3 replication found much weaker hysteresis
(+0.141/+0.108, CIs spanning 0) than our sealed +0.391/+0.369. A
source audit ruled out the estimand (their statistic STRENGTHENS our
sealed data to +0.571/+0.443) and the edit selector (70% agreement,
regret ~0.002), leaving launch state and edit/anchor convention. The
preregistered 2×2 factorial (`G3_ADJUDICATION.md`, domain 32,
sealed; replay gate: the fresh×conv-A cell reruns the SEALED G3 on
its original seed keys and reproduced all 672 rows exactly) decides
it: **launch state dominates.** Fresh nascent launches show strong
hysteresis under BOTH conventions (+0.44 to +0.57); naturally
evolved generation-60 launches are attenuated in all four cells
(+0.14 to +0.32) — and their pulse-1 half-lives START at ~12–15
fissions, already at the plateau fresh lineages only reach after
long holding. The external result falls almost exactly on our
natural-launch cells: **both labs were right about their own launch
conditions; the disagreement was a moderator, not an error.**
Registered consequence (ledger #11): G3's finding is scope-revised —
the chemistry's depth-of-writing memory is primarily a property of
young, unformed lineages; an evolved lineage sits near its imprint
ceiling. (Weak residual hysteresis in evolved lineages remains real:
three of four natural cells exclude zero.)

## Artifacts

Full intervention-program documentation — registered designs, seed
architecture, gates, and complete result tables for Phases A, B, C1–C3,
and D1–D3 — is consolidated in **`INTERVENTIONS.md`**.

- Code: `sim.py`, `features.py`, `cohort.py`, `models.py`,
  `run_dev.py`, `run_conf.py`, `analyze.py` (this directory).
- Frozen models: `results/frozen_models.pkl`
  (SHA-256 `9b58549ed5ee0193…`), dev summary `results/dev_summary.json`.
- Confirmation data: `results/conf_data.pkl`; metrics
  `results/confirmation_metrics.json`; campaign hash
  `c6cfebcee19671c1…` (identical across both campaigns).
- Figures: `results/figures/` (1x) and `results_5x/figures/` (5x):
  `fig_rank_transfer.png`, `fig_calibration.png`,
  `fig_process_prevalence.png`, `fig_reliability.png`.
- 5x records: `results_5x/`; 25x records: `results_25x/`
  (each: `confirmation_metrics.json`, `frozen_models.pkl`,
  `conf_data.pkl`, `figures/`).
- Runtime (12 workers): 1x — dev 4 s, confirmation 2 × ~72 s; 5x —
  dev 20 s, confirmation 2 × ~5.5 min; 25x — dev 160 s,
  confirmation 2 × ~23 min.
