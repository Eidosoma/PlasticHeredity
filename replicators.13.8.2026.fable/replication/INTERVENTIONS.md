# The intervention program: from prediction to control

Consolidated documentation of all intervention experiments in the
clean-room replication — registered designs, seed architecture, gates,
and results. Every experiment used the FROZEN v2 coordinate
(`results_v2/frozen_models_v2.pkl`, SHA-256 `0420cb49a2077bd2…`)
without refitting, and every gate was preregistered in the session
plan file before any outcome was observed.

Chronology: Phase A/B (one-shot control, dose–response) → Phase C
(transfer + predicted null, mechanism extraction, closed-loop
steering) → Phase D (outcome decomposition, long-run costs, controller
audit — the reviewer's five paper-facing checks).

---

## Common machinery

### Edit space
Mass-preserving single-molecule **swaps**: remove one molecule of type
i (requires n_i ≥ 1), add one of type j ≠ i. History features (the
direct-8 block) are held at the state's values — an instantaneous swap
changes neither history nor mass. Score of an edited state =
`registry_v2.predict_v2` on (X9_state, `graph_state_195(n_edited, β)`).

### Registered screening (`run_intervention.screen_swaps`)
Score all single adds (100) and single removes (≤ #present); take the
top-10 in each direction; exactly score the ~100 up-up and ~100
down-down swap combinations; select the extremes. The random arm draws
a uniformly random legal swap from its own stream. All screened swap
tables and scores are persisted in
`results_intervention/selections.pkl`.

### Common random streams (CRN)
Branch b of every arm uses spawn key `(5, cand_i, m, lm, b)` — the arm
is absent from the key, so all arms share per-branch seeds. Once an
edit changes the event count, later draws desynchronize; arms
experience **common random streams, not identical realized futures**
(wording per reviewer check D0). Steering lineages use
`(7, cand_i, m, rep)` (controller absent); steering random-controller
edits use `(8, cand_i, m, rep, f)`; random-arm selection uses
`(6, cand_i, m, lm)`.

### Inference
The catalytic matrix is the resampling unit in every bootstrap; states,
arms, and lineages travel as within-matrix blocks (or are collapsed to
per-matrix means first, in steering). Bootstraps: 2,048 draws (A/B),
1,024 (C/D).

### Seed tags (all domain-separated, 256-bit)
| Experiment | Tag |
|---|---|
| Phase A/B, C3 Part 2, D1 | `intervention-2026-08-13` |
| C2 | `knob-A{A}_S{sigma}-2026-08-13` (per regime) |
| C1, D2/D3 | `steering-2026-08-13` |

---

## Phase A — paired extremes (`run_intervention.py`)

**Design.** Fresh cohort: 40 matrices × 2 candidates, landmarks
{20, 35, 50, 65, 80} → 400 post-fission states. Arms per state:
up (max predicted p̂), down (min), noop, random; 64 CRN branches each;
outcome `JOINT_BREAK_RUN3` plus break and renewal-given-break
components.

**Registered gates** (both candidates): G1 mean paired (q_up − q_down)
> 0 with bootstrap lower bound > 0; G2 ordering up > noop > down; G3
|random − noop| < 0.25 × (up − down).

**Results (cand 02 / 03): ALL GATES PASS.**
- Arm means: up 0.395/0.440, noop 0.334/0.383, random 0.334/0.390,
  down 0.239/0.281.
- up − down: **+0.156 [0.125, 0.189] / +0.158 [0.130, 0.186]**.
- random − noop: −0.0001 [−0.010, +0.010] / +0.007 [−0.006, +0.019] —
  a random swap of identical magnitude does nothing.
- Predicted vs realized shift: +0.230 vs +0.156 / +0.229 vs +0.158
  (winner's-curse shrinkage ≈ 0.7, as registered-expected).
- Noop-arm rank check (free re-confirmation of the frozen coordinate
  on 40 new matrices): Spearman 0.870 / 0.904.

## Phase B — dose–response (`run_intervention.py`, auto-run on A pass)

**Design.** Per state, six swaps at quantiles {0, 20, 40, 60, 80,
100}% of the screened predicted-shift distribution (endpoints shared
with Phase A); 64 CRN branches per interior arm.

**Registered gates**: (i) mean within-state Spearman(predicted Δp̂,
realized Δq) > 0 with bootstrap lower bound > 0; (ii) pooled OLS slope
> 0 with lower bound > 0.

**Results: ALL GATES PASS.** Within-state Spearman **0.475
[0.402, 0.545] / 0.471 [0.419, 0.525]**; pooled slope **0.699
[0.620, 0.768] / 0.733 [0.661, 0.795]**. The knob is graded;
attenuation ≈ 0.7 = selection shrinkage + frozen-model calibration
error.

## Phase C2 — knob transfer + predicted null (`run_knob_transfer.py`)

**Design.** Phase-A protocol, zero-shot frozen home-regime scorer, in
four perturbed regimes: (A, σ) ∈ {(−4,5), (−3,4), (−5,4)} (transfer
registered-expected) and **(−4,3) with a registered NULL prediction**
(the regime probe had shown no state-local signal there). 20 matrices,
landmarks {35, 65}, 4 arms × 48 CRN branches per regime.

**Results** (80 states per regime; up−down with 1,024
matrix-bootstrap CIs):
| Regime | up−down (02 / 03) | Gate | Result |
|---|---|---|---|
| (−4,5) | +0.062 [+0.029,+0.102] / +0.053 [+0.021,+0.087] | transfer | pass (blemish: cand 03 random−noop +0.021, CI excludes 0 — knob real but less specific) |
| (−3,4) | +0.144 [+0.102,+0.188] / +0.170 [+0.110,+0.226] | transfer | pass |
| (−5,4) | +0.170 [+0.117,+0.218] / +0.173 [+0.126,+0.220] | transfer | pass |
| (−4,3) | +0.024 [−0.009,+0.064] / +0.007 [−0.032,+0.051] | **null** | **null confirmed** |

Arm means per regime (up / noop / random / down):
| Regime | Candidate 02 | Candidate 03 |
|---|---|---|
| (−4,5) | 0.115 / 0.086 / 0.080 / 0.053 | 0.130 / 0.098 / 0.119 / 0.077 |
| (−3,4) | 0.329 / 0.286 / 0.290 / 0.184 | 0.395 / 0.336 / 0.340 / 0.226 |
| (−5,4) | 0.510 / 0.428 / 0.441 / 0.340 | 0.551 / 0.494 / 0.509 / 0.378 |
| (−4,3) | 0.269 / 0.254 / 0.256 / 0.245 | 0.291 / 0.282 / 0.300 / 0.284 |

## Phase C3 — mechanism extraction + rule controller (`run_mechanism.py`)

**Part 1 (exploratory, labeled).** Pooled Spearman between each
screened swap's predicted shift and per-type physical quantities:
add_in_boost −0.136, **add_out_infl −0.241**, add_self_cat −0.043,
rem_in_boost +0.159, rem_out_infl +0.127, rem_self_cat +0.039,
rem_count +0.395.

**Frozen rule** (recorded in JSON before any Part 2 branch ran):
quantity = out_infl (outgoing catalytic influence, c_t = Σ_i x_i β_it),
orientation −. rule_up = *remove the most catalytically influential
present type, add the least influential type*; rule_down mirrors.

**Part 2 (confirmatory).** Home cohort, arms {rule_up, rule_down,
noop} × 48 CRN branches (Phase A spawn-key domain). Gate: rule up−down
> 0 with bootstrap lower bound > 0.

**Results: PASS.** rule up−down **+0.103 [+0.077,+0.131] / +0.103
[+0.081,+0.129]**; efficiency vs the full model **0.66 / 0.65**.
Two-thirds of the model knob compresses into one physical scalar.

**Orientation clarification (added 2026-08-15, after an external
replication query; no code or result changed).** The rule's quantity
is the OUTGOING (source/catalyst) influence, computed in code as
`out_infl = x @ beta`, i.e. c_t = Σ_i x_i β[i,t] = (βᵀx)_t — how
strongly type t catalyzes the molecules currently present. It is NOT
the incoming support (β x)_t = Σ_k β[t,k] x_k received by type t.
Index convention: the kinetics use c = β n with bn = 1 + c/N, so
β[i,j] = how much the presence of j boosts i (row = target, column =
catalyst). Caution: `sim.py` uses the letter c for the INCOMING boost
vector β n, while the C3 docstring writes the OUTGOING quantity as
c_t — same symbol, opposite orientation; read the code, not the
letter. The two rankings are essentially uncorrelated under the
frozen β distribution (mean Spearman ≈ 0 across states), so a
transposed implementation behaves as a weak proxy (~+0.01) rather
than reproducing the rule effect (~+0.10). Fingerprint for external
replicators: the correctly oriented rule agrees with the frozen v2
predictor's edit direction in ≈99.5% of states. This quantity and
orientation are used identically in `run_mechanism.py`,
`run_g4_tournament.py` (disagreement classes and beta surgery), and
`run_g5_internal.py` (L0/L3).

## Phase C1 — closed-loop steering (`run_steering.py`)

**Design.** Fresh cohort: 24 matrices × 2 candidates; controllers
{model_up, model_down, noop, random}; 6 replicate lineages each;
horizon 60 fissions; after every fission the model controllers score
the ~140 marginal edits and apply the (best-remove, best-add) swap for
their sign; noop lineages verified bitwise against plain trajectories.

**Registered gates** (primary outcome = certified break→3-run episode
count): G1 paired up−down > 0 with bootstrap lower bound > 0; G2
ordering up > noop > down; G3 |random − noop| CI includes 0.

**Results (cand 02 / 03): ALL GATES PASS.**
| Outcome per 60 fissions | model_up | noop | random | model_down |
|---|---|---|---|---|
| Episodes | 3.72 / 3.94 | 2.88 / 3.11 | 2.95 / 3.15 | 0.67 / 0.76 |
| Breaks | 17.3 / 16.8 | 6.6 / 6.8 | 6.8 / 7.0 | 0.97 / 1.05 |
| Inheritance fraction | 0.712 / 0.720 | 0.891 / 0.887 | 0.887 / 0.883 | 0.984 / 0.983 |
| Longest run | 19.9 / 19.1 | 33.2 / 32.3 | 33.5 / 31.2 | 57.5 / 56.3 |

Episodes up−down **+3.05 [+2.27,+3.73] / +3.19 [+2.39,+3.83]**;
random−noop CIs include 0. This is the properly-controlled analog of
the original paper's Figure 6/Table 1 protocol (which did not
reproduce under the Φ scorer): the claim shape is vindicated with a
different signal and the controls the original lacked.

## Phase D1 — registered outcome suite (`run_d1_outcomes.py`)

**Design.** Regenerate Phase A (same seeds/arms; joint-event arm means
consistency-asserted against stored values) capturing six registered
outcomes per branch.

**Results, candidate 02** (arm means up / noop / random / down, then
paired up−down with 1,024 matrix-bootstrap CI):
| Outcome | up | noop | random | down | up−down [CI] |
|---|---|---|---|---|---|
| Break hazard | 0.497 | 0.449 | 0.451 | 0.380 | +0.117 [+0.091,+0.144] |
| Run-3 after break | 0.769 | 0.710 | 0.713 | 0.612 | +0.162 [+0.133,+0.192] |
| Persist-5 after renewal | 0.725 | 0.650 | 0.674 | 0.579 | +0.143 [+0.101,+0.183] |
| Inherited boundaries | 10.813 | 10.970 | 10.975 | 11.223 | −0.410 [−0.532,−0.288] |
| Survival | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| Growth updates / fission | 63.48 | 65.64 | 65.68 | 69.55 | −6.07 [−8.20,−4.11] |

**Results, candidate 03:**
| Outcome | up | noop | random | down | up−down [CI] |
|---|---|---|---|---|---|
| Break hazard | 0.544 | 0.498 | 0.500 | 0.423 | +0.121 [+0.095,+0.145] |
| Run-3 after break | 0.784 | 0.742 | 0.752 | 0.637 | +0.148 [+0.120,+0.180] |
| Persist-5 after renewal | 0.749 | 0.681 | 0.704 | 0.600 | +0.148 [+0.106,+0.187] |
| Inherited boundaries | 10.709 | 10.875 | 10.854 | 11.146 | −0.437 [−0.569,−0.325] |
| Survival | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| Growth updates / fission | 16.13 | 16.62 | 16.58 | 17.64 | −1.51 [−2.10,−1.02] |

(Growth-update magnitudes differ between candidates by construction:
candidate 02 counts single-molecule events, candidate 03 counts
vector-Poisson steps.)

**Identity-exact decomposition** (q = b·r, midpoint convention,
residual ≤ 1e-16): break-hazard share **0.543 / 0.569**. Per the
registered >50% criterion, the reviewer's stability-primary conjecture
is adjudicated in their favor — framing: *the knob primarily controls
hereditary stability* — with the caveat that it is a 54/46 split.

## Phase D2 — long-run cost panel (`run_d2d3_steering_audit.py`)

**Design.** Deterministic regeneration of the steering campaign
(outcome means consistency-asserted) with extended logging + a
120-fission extension ({model_down, noop} × 2 reps). Eight predictions
registered from the C3 physics.

**Results: ALL 8 PREDICTIONS PASS (both candidates).** Final-10-fission
means (cand 02 / 03):
| Metric | model_down | noop | model_up |
|---|---|---|---|
| Entropy | 1.59 / 1.59 | 2.24 / 2.26 | 2.62 / 2.64 |
| Occupied species | 10.4 / 10.4 | 16.6 / 16.9 | 20.5 / 20.7 |
| Top-1 share | 0.50 / 0.50 | 0.34 / 0.34 | 0.24 / 0.23 |
| Catalytic throughput | 34.9 / 35.9 | 25.0 / 25.6 | 22.4 / 23.2 |
| Cross-lineage similarity | 0.87 / 0.81 | 0.54 / 0.54 | 0.50 / 0.51 |
| Extinctions | 0 | 0 | 0 |

Extension: down-steered inheritance fraction **0.996 / 0.995** over
fissions 61–120, zero extinction. **Conclusion: down-steering
manufactures a compotype** — concentrated, catalytically hotter than
baseline, convergent across independent lineages, persistent — not a
degenerate frozen state. The cost of near-perfect heredity is
compositional diversity.

## Phase D3 — controller action audit (same script)

Per-controller, per-candidate detail:

| Quantity (cand 02 / 03) | model_up | model_down |
|---|---|---|
| Distinct swaps per 59-edit lineage | 42.5 / 41.9 | 35.9 / 36.2 |
| Consecutive-repeat rate | 0.037 / 0.037 | 0.046 / 0.043 |
| Cycling rate (reverses a swap ≤ 3 back) | 0.003 / 0.005 | 0.000 / 0.000 |
| Rule agreement — exact swap | 0.001 / 0.003 | 0.055 / 0.080 |
| Rule agreement — add side | 0.003 / 0.004 | 0.628 / 0.663 |
| Rule agreement — remove side | 0.497 / 0.507 | 0.115 / 0.151 |
| Out-of-envelope state fraction | 0.118 / 0.100 | 0.198 / 0.207 |

Reference envelope: [0.5%, 99.5%] per-dimension bounds of the frozen
v2 PCA coordinates over the v2 confirmation cohort's natural
post-fission states (noop lineages sit at 0.114 / 0.107; random at
0.103 / 0.083). Reading: the controller is adaptive (no broken-record
or self-fighting behavior); the stabilized end of the state space is
moderately extrapolated; and the model agrees with the frozen physical
rule at the component level in the direction that matters for each
controller — the stabilizing controller's ADD choices are
rule-consistent (adding high-influence types), the destabilizing
controller's REMOVE choices are (removing high-influence types) —
while selecting different specific molecules.

## Phase E — steer–release–challenge (`run_release_challenge.py`)

**Design.** 96 controller-written states (model_down, 60 fissions,
regenerated from the steering tag) + 96 matched natural states; frozen
v2 precursor scoring; 60-fission free release (spawn domain 9);
challenge arms {none, random-k for k ∈ {2,4,8,16}, adversarial swap}
(perturbation streams domain 11) × 32 branches × 24 fissions (domain
10); registered four-outcome classifier (held / returned /
mode-recovered / lost; departure < 0.7, return > 0.9 sustained ≥ 3,
mode = final-6 inheritance ≥ 5/6 and top-1 ≥ 0.45); challenge anchor =
release-end composition (registered refinement); matrix bootstraps
(1,024). Registered prediction on record: finite-basin attractor —
**the prediction failed.**

**Results (cand 02 / 03): VERDICT written-but-passive (both).**
- Precursor risk: 0.112 / 0.127 (natural ≈ 0.33–0.38) — genuinely
  stabilized at release.
- Release: anchor similarity decays ~0.95 → ~0.55 within ~5–10
  fissions (natural drift floor); composition-hold at 60 fissions
  0.31 / 0.29; mode-survival 0.60 / 0.56; release inheritance
  0.909 / 0.928 vs natural 0.908 / 0.909.
- Challenge: written (held+returned) 0.17–0.19 / 0.21–0.23 vs natural
  0.20–0.21 / 0.24–0.26 at every dose; all difference CIs straddle 0;
  outcomes dose-independent; **basin radius k = 0**.
- Consequence for D2's framing: the 120-fission "persistence" was
  under continued control; released states relax to baseline. The
  controller maintains the compotype-like state; it does not install
  one. Unifying conclusion: in this chemistry, heredity and
  organization are maintained processes, not places.

## Phase F7 — attractor-aware controller (`run_f7_attractor_controller.py`)

**Design.** Six controllers (v2-down; R_Q-only; composome-alignment-
only; joint = (1−risk)+R_Q+atlas-sim, registered equal weights; random;
noop), per-fission marginal swaps, 60 steering fissions, 24 matrices ×
2 candidates × 2 reps (lineage streams domain 16, controller absent
from the key), then 60 release fissions. Registered comparison: best
attractor-aware controller must exceed v2-down's own-anchor similarity
at release+10 by ≥ 0.10 with matrix-bootstrap CI > 0. Registered
prediction on record: no controller achieves durable hold.

**Results (cand 02 / 03): prediction CONFIRMED, comparison FAILED both
candidates.**
- Written-state properties: joint achieves the highest composome
  alignment (atlas-sim 0.936/0.915); v2-down the lowest risk
  (0.111/0.129) and — notably — the highest R_Q (0.838/0.864),
  exceeding even direct R_Q-steering (0.736/0.750): minimizing
  break-risk ≈ maximizing flux alignment.
- Release: every controller's state decays to the same drift floor by
  +10 fissions (anchor@10 spans 0.53–0.63 across all six controllers);
  aware-vs-v2down at +10: −0.022 [−0.125,+0.085] / +0.044
  [−0.044,+0.129].
- Conclusion: the objective was not the problem; no writable objective
  installs cross-generation persistence. "Steering wheel, not
  programmer," maximally strengthened.

---

## Phase G2 Stage C — staged explore–recover–consolidate controller (`run_g2_resist_resil.py`)

**Design (preregistered in PHASE_G.md).** State machine over the
steering loop: EXPLORE (break-up edits via the frozen q_B student)
until a break is observed; RECOVER (renewal-up via frozen q_R) until
run3 certified; CONSOLIDATE (break-down) thereafter. Arms
{always-stabilize, always-explore, staged, random, noop}; 60 fissions;
24 × 2 × 2 lineages. Registered success = all three, both candidates:
staged > noop on compositionally distinct new episodes; staged <
always-explore on breaks; staged episodes outlive always-explore's
after certification.

**Results: registered FAILURE (prediction-miss ledger #4), two of
three gates pass.** Staged beats explore on breaks (1.5 vs 14.9
(02); 1.9 vs 15.4 (03)) and on certified-episode survival (46 vs 8.5;
44 vs 8.2 fissions), but produces FEWER distinct episodes than noop
(1.23 vs 2.60; 1.31 vs 2.71): consolidation suppresses natural episode
turnover. Stability and novelty compete for the same lineage under
this architecture — programmed exploration bought nothing over the
chemistry's own drift.

## Phase G3 — periodic and event-triggered maintenance (`run_g3_halflife.py`)

**Design (preregistered).** (i) Pulse ladder: model-down steering for
P ∈ {1,2,4,8,16,32,60} fissions then 60 free; registered accumulation
test = Spearman(P, post-release persistence), matrix bootstrap.
(ii) Periodic: one model-down edit every k ∈ {1,2,4,8,16} fissions,
budget-matched random arms. (iii) Event-triggered: edit only when
frozen v2 risk > θ ∈ {0.15, 0.25, 0.35}. Registered adjudication:
continuous-required / sparse-sufficient / accumulating-hysteresis;
registered prediction was sparse-sufficient with NO accumulation.

**Results: ACCUMULATING-HYSTERESIS, both candidates (ledger miss #5).**
- Half-life of the engineered state grows with pulse length:
  5.6 → 15.5 fissions (02), 5.2 → 13.4 (03), saturating ≈ P=16;
  Spearman +0.391 [+0.289,+0.492] / +0.369 [+0.259,+0.490].
- Periodic: inheritance 0.982 (k=1) → 0.899 (k=16), graceful; random
  arms flat (0.884–0.897) at every budget.
- Event-triggered: θ=0.35 sustains inheritance 0.974/0.977 with
  15.9/17.2 edits per 60 fissions (~0.27 edits/generation; ~9
  excursions) — the minimum feedback rate for maintained heredity
  with a working risk sensor.

## Phase G5 — internalized-controller ladder (`run_g5_internal.py`, EXPLORATORY)

**Design (preregistered; one documented deviation on L3 features, see
PHASE_G.md).** Policies {L0 memoryless influence rule; L1 = L0 gated
on last-boundary-broken; L2 = L0 gated on streak < 3; L3 = frozen
depth-3 decision trees distilled from v2 choices on dev states;
v2-down; random; noop} on 60-fission steered lineages, CRN across
arms and conditions. Readouts: maintenance, k8-recovery (perturbation
at fission 30), three transfer regimes. Kinetic appendix: leave rates
damped by 1/(1 + λ·influence-percentile), λ ∈ {0.1, 0.3}, no editor.

**Results (cand 02 / 03): registered miss #6 — the ladder is
ANTI-monotone.**
- L0 recovers **0.92 / 0.96** of v2's maintenance gain over noop
  (0.976/0.980 vs 0.983/0.984, noop 0.900/0.889), equals it on
  recovery (0.992/0.992 vs 0.990/0.995), and transfers to all three
  regimes at v2's level (worst regime: 0.975/0.979 vs noop
  0.807/0.826).
- L3 0.68/0.64 of the gain and degrades out-of-regime; L2 0.42/0.51;
  L1 0.22/0.42; random ≈ noop (CIs span 0).
- Kinetic prototype: NULL at registered λ (all CIs span 0); λ=0
  control reproduces the frozen baseline exactly.
- Reading: the policy content is one locally computable scalar; model
  value is in timing (compare G3's 16-edit risk-triggered result);
  soft retention bias cannot replace discrete edits — the actuator,
  not the knowledge, is the internalization bottleneck.

## Phase I — reciprocal Φ-r bridge (`run_phir_bridge.py`, `phir.py`)

**Design (preregistered in PHIR_BRIDGE.md).** Six arms on identical
24-swap CRN panels each fission (ph_stab, ph_destab, phir_max,
phir_min, random, noop), 24 fresh matrices × 2 candidates × 2 reps ×
60 fissions, domain 27. Realized Φ-r measured on concatenated
per-update composition series of fissions 41–60 (downstream, never
used for selection); surrogate Φ-r (linearized growth flow) used
only for selection, with a registered manipulation check.

**Results.** Validity PASS (+0.185/+0.174 inherit swing); random
null clean. T1 (heredity → Φ-r): NULL both candidates — realized
Φ-r (−0.01..−0.03 everywhere) is insensitive to a 6× break-rate
manipulation. T2 (surrogate → heredity): passes all three
co-primaries in 03 only → fails the both-candidates gate; the
manipulation check shows the surrogate does not move realized Φ-r,
so the 03 effect is not attributable to Φ-r. T3: near-orthogonal
action choice (ρ ≈ 0.08). Adjudication: **independent organizational
axes** (constrains only this frozen reconstruction). Unexplained
lead: the linearized-flow surrogate carries a weak heredity-relevant
direction nearly orthogonal to the v2 knob.

**Phase J (`run_phir_confirm.py`, PHIR_CONFIRM.md; domain 28,
sealed).** 2× prospective test on 48 fresh matrices, six arms,
shared 12-swap CRN panels, probe-rollout Φ_R controller. C1
(heredity → Φ_R) PASS both candidates (+0.155/+0.178, CIs excluding
0) — prospective confirmation of the addendum. C2a gauge-steering
partial (03 only, +0.065); C2b (Φ_R → heredity) NULL everywhere —
in 03 the full registered "responsive signature, not a controller"
pattern realized. C4 violated in one cell (03 heredity, random −
noop −0.017): random-editing cost detectable at 2× power; primary
contrasts equal-budget, untouched; background claim
scale-qualified. Suite 30/30; SEAL.json with source hashes.

**Phase K (`run_phir_dose.py`, PHIR_DOSE.md; domain 29, sealed).**
Dose ladder (11 arms × 5 cadences), atom decomposition of Phase J
(replay gate exact), natural-prediction cohort, robustness
appendix. Φ_R dose-graded in 03 only (miss #8); decomposition:
downward-causation atoms carry the response (+0.577/+0.504),
synergy-persistence negative, authors' "emergence" summary responds
in both candidates (+0.193/+0.207); natural prediction matrix-level
only (centered null); sign robust across instrument variants, CLR
magnitude-sensitivity in 02 disclosed. Suite 31/31.

**Addendum (`run_phir_code_addendum.py`, `phir_code.py`).** The
authors' public code implements a DIFFERENT quantity (revised Φ_R,
nine-atom ΦID sum on macro-averaged Fiedler halves; port verified to
~1e-14 against their repo). Byte-identical replay of the whole
campaign (replay gate 0/576): **T1 PASSES under the implemented
definition, both candidates** (+0.208 [+0.134,+0.279] /
+0.256 [+0.121,+0.392]; levels positive 0.97–1.91; random null;
text-vs-code correlation ≈ 0.1). Implemented Φ-r = responsive
signature of hereditary organization, causally downstream of the
heredity dial; not a demonstrated controller (no working
code-Φ_R-steering rule exists). Ledger miss #7.

---

## Registered interpretation boundary (applies to everything above)

These experiments establish causal control of the break-and-renewal
probability (primarily via hereditary stability) by coordinate-selected
molecular edits, **in this simulator**, under the frozen v2 scorer.
They do not rescue the original paper's Φ-directed intervention
claims, and they are not chemistry.

## File map

| Experiment | Script | Results |
|---|---|---|
| A + B | `run_intervention.py` | `results_intervention/` (JSON, selections.pkl, arm + dose figures) |
| C2 | `run_knob_transfer.py` | `results_knob_transfer/` |
| C3 | `run_mechanism.py` | `results_mechanism/` |
| C1 | `run_steering.py` | `results_steering/` |
| D1 | `run_d1_outcomes.py` | `results_d1/` |
| D2/D3 | `run_d2d3_steering_audit.py` | `results_d2d3/` (JSON, cost-panel figure) |
| E | `run_release_challenge.py` | `results_release/` (JSON, release-trace + challenge-dose figures) |
| F1–F7 | `run_f1_kahana.py`, `run_f2_states.py`, `run_f3_f4.py`, `run_f5_sj.py`, `run_f5b_f6.py`, `run_f7_attractor_controller.py` | `results_f/` |
| G1–G5 | `run_g1_competency.py`, `run_g2_resist_resil.py`, `run_g3_halflife.py`, `run_g4_tournament.py`, `run_g5_internal.py` | `results_g/` (preregistration + results appendix: `PHASE_G.md`; external-replication clarifications: C3 orientation note above, G4 beta-surgery random-arm audit note in `PHASE_G.md`) |
| I | `run_phir_bridge.py`, `phir.py`, `phir_code.py`, `run_phir_code_addendum.py` | `results_phir_bridge/` (preregistration + results + addendum: `PHIR_BRIDGE.md`) |
| J | `run_phir_confirm.py` | `results_phir_confirm/` (preregistration + results: `PHIR_CONFIRM.md`; SEAL.json) |
| K | `run_phir_dose.py` | `results_phir_dose/` (preregistration + results: `PHIR_DOSE.md`; SEAL.json) |
| L | `run_phir_paper.py`, `phir_paper.py` | `results_phir_paper/` (preregistration + results: `PHIR_PAPER.md`; SEAL.json) |
| M | `run_phir_sr.py` | `results_phir_sr/` (preregistration + results: `PHIR_SR.md`; SEAL.json) |
| N | `run_phir_foresight.py` | `results_phir_foresight/` (preregistration + results: `PHIR_FORESIGHT.md`; SEAL.json) |
| G3-ADJ | `run_g3_adjudication.py` | `results_g3_adjudication/` (preregistration + results: `G3_ADJUDICATION.md`; SEAL.json) |

Validation: `test_validation.py` (24 checks; includes swap legality,
CRN stream pairing, deterministic selection, steering-callback
neutrality, extended-outcome unit cases, decomposition identity).
Reproduction: each script is deterministic from its seed tag; regenerated
campaigns assert equality with stored results where applicable.

Runtimes (12 workers): Phase A ≈ 4.6 min, Phase B ≈ 6.5 min, C2 ≈ 4.5
min total (4 regimes), C3 ≈ 4 min, C1 ≈ 11 min, D1 ≈ 4.6 min,
D2/D3 ≈ 26 min (incl. the 120-fission extension and per-fission PCA
logging), E ≈ 13 min. Validation suite: 19 checks, all passing.
