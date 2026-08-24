# PHASE G — Preregistration (written 2026-08-14, BEFORE any module ran)

This document is the durable, repo-resident preregistration for the
final bounded program (reviewer's G-proposal, full scope per user
decision). Every design element, prediction, gate, equivalence margin,
and trim below is fixed before outcomes are observed. Results are
appended per module under "RESULTS" headings as each completes; nothing
above a RESULTS heading may be edited afterward.

Standing machinery (unchanged): frozen v2 models
(`results_v2/frozen_models_v2.pkl`, SHA-256 `0420cb49…`) untouched;
new task-specific predictors trained on development data and frozen
before confirmation; fresh seed domains per module; candidates 02/03
never pooled; catalytic matrices are the bootstrap/inference unit
(1,024 draws unless stated); common random streams where applicable;
exact replay from seed tags; validation suite green throughout.

Adopted do-NOT-run list (reviewer's, verbatim in spirit): no further
extreme-arm up/down replications; no longer continuous-control runs;
no more objective-weight combinations; no same-clock anchor-return
tests; no unmatched dose ladders.

Spawn-domain registry additions: 20 G1 fingerprints/carriers; 21 G2
(sub-keyed A/B/C); 22 G3; 23 G4; 24 G5.

---

## G1 — Competency-attractor test

**Question.** Does BEHAVIOR return after perturbation even though
composition does not?

**Fingerprint.** K(s) with 8 components, each estimated from
independent branch ensembles: q_break at horizons {4, 8, 12}; q_joint
(break+run3, horizon 12); q_persist5 (given episode); E[inherited
boundaries of 12]; E[growth updates per fission]; E[entropy(final) −
entropy(s)]. Component z-standardization frozen from a development
fingerprint cohort (12 dev matrices × landmark states). Behavioral
distance = Euclidean in z-space.

**States** (24 steering matrices × 2 candidates, deterministic
regeneration): natural (noop fission 60), model-down written,
model-up written, composome-aligned written (F7 comp_only), ordinary
mid-lineage control (noop fission 30).

**Protocol.** K(0) from branch half A (16 branches, domain 20). Arms
{none, k8-swap}; 3 carrier trajectories per arm; K(t) at recovery
times t ∈ {1, 2, 5, 10, 24} from 16 half-B branches launched from the
carrier state at t.

**Two-target race (registered crux).** For each atypical state
(model-down/up, comp-aligned), compare d[K(t), K(0)] against
d[K(t), K_matrix] where K_matrix is the matrix-typical fingerprint
frozen from dev lineages. Competency attractor requires: (i)
d-to-K(0) at t = 10 ≤ 0.5 × its post-perturbation peak, AND (ii)
K(t) closer to K(0) than to K_matrix at t = 10, matrix-bootstrap CI
excluding the tie.

**Registered prediction.** Relaxation to the matrix-typical
fingerprint wins; a state-specific competency return would be a major
surprise (and the strongest possible support for H3).

**Positive control (gate).** The same behavioral assay on the
Singh–Jain model (fingerprint: mode flag, X4 at division,
interdivision time) must show within-basin behavioral return after
small perturbations; if it does not, G1 conclusions are suspended.

## G2 — Resistance vs resilience

**Stage A (resistance).** Frozen student q_B(s) = P(break within
6 | s), registry-v2 architecture, trained on regenerated 25x dev rows.
Confirmation (fresh states, 24 × 2 matrices, domain 21): arms
{break-up, break-down, random, noop} by marginal screening on q_B;
48 CRN branches; outcome = break hazard (horizon 6). Gates: paired
up−down > 0 with bootstrap lower bound > 0 in both candidates;
|random − noop| CI includes 0.

**Stage B (resilience — clean causal design).** Frozen student
q_R(s) = P(run3 within 8 | post-break daughter s), trained on dev
post-break states. Confirmation: harvest fresh post-break daughters
(the first break in confirmation lineages); restore the IDENTICAL
post-break state across arms {renewal-up, renewal-down, random,
noop}; 48 CRN branches; outcomes: run3 (primary), run5,
time-to-renewal, new-episode anchor similarity, inherited boundaries.
Gates as Stage A on run3. This supersedes D1's conditioned
decomposition with an unconditional recovery test.

**Stage C (staged controller).** Policies {always-stabilize
(break-down each fission), always-explore (break-up), staged
(break-up until a break occurs → renewal-up until run3 certified →
break-down thereafter; re-arm on later breaks), random, noop};
60 fissions; 24 × 2 × 2 lineages (domain 21 sub-key). Registered
success requires ALL THREE in both candidates: (1) staged > noop on
compositionally distinct new episodes (episode start < 0.9 cosine to
the previous episode's start); (2) staged < always-explore on total
breaks; (3) staged episodes survive longer post-certification than
always-explore's.

**Registered prediction.** A and B both pass (the knob decomposes);
Stage C passes (1) and (2); (3) is the uncertain one.

## G3 — Control half-life and minimum feedback rate

**Pulse ladder.** Model-down for {1, 2, 4, 8, 16, 32, 60} fissions,
then release to 60; traces of risk, inheritance, R_Q, atlas
similarity, entropy, top-1, throughput; per-pulse post-release
half-life of anchor similarity. Registered accumulation test:
Spearman(pulse length, post-release persistence) with bootstrap CI —
a positive CI-excluded correlation = hysteresis (flagged live; would
revise Phase E's characterization).

**Periodic control.** One model-down edit every {1, 2, 4, 8, 16}
fissions over 60, plus budget-matched random arms; maintained
inheritance vs edit budget.

**Event-triggered control.** Score frozen risk every fission; edit
only when q̂ > {0.15, 0.25, 0.35}; report inheritance maintained,
edits used, threshold excursions → the edits-per-maintained-generation
curve.

**Registered adjudication.** Continuous-correction-required (missing
1–2 edits destroys the effect) / sparse-sufficient (1 edit per 4–8
fissions preserves ≥ 0.95 inheritance) / accumulating hysteresis
(positive pulse-persistence correlation).
**Registered prediction:** sparse-sufficient, with no accumulation
(half-life independent of pulse length ≈ 5–10 fissions).

## G4 — Mechanism disagreement tournament

**Mining (dev matrices only, frozen before outcomes).** Score
candidate swaps with Δv2 (frozen), Δrule (outgoing-influence score),
ΔR_Q; select covariate-matched sets (removed-type abundance, baseline
risk, cosine displacement) for the four disagreement classes:
A (v2↓ rule↓ R_Q↑), B (v2↓ rule↑ R_Q↓), C (v2↑ rule↓ R_Q↑),
D (v2↑ rule↑ R_Q↓), where ↓ = predicts stabilization. Target ≈ 100
edits per class per candidate (report shortfalls).

**Branch test.** 32 CRN branches × 12 fissions per edit vs the
unedited state (domain 23); realized Δq per class; the winning
predictor is the one whose sign matches realized Δq in the classes
where it disagrees with the others (majority, with bootstrap CIs).

**Transplantation.** For 40 native edits per candidate: native
(β_A, s_A); same (i, j) on the most composition-similar dev state
under β_B; that β_B state unedited; a β_B-native edit. Registered
relational claim: realized sign follows the native-matrix v2
prediction computed AT the applied state, not molecule identity.

**Beta surgery.** At fixed composition: norm-matched β-edge
modifications raising / lowering / preserving the outgoing-influence
quantity + random-edge controls of equal total Frobenius change;
32 branches × 12 fissions. Sufficiency gate: influence-raising vs
influence-lowering surgery separates realized q with bootstrap
CI > 0 while random surgery of equal norm does not.

**Registered interpretation table** (reviewer's): influence wins →
compact physical mechanism; R_Q wins → flux-alignment mechanism; v2
retains residual after matching → higher-order motifs/history;
effects follow matrix → the causal unit is relational
(network × state). **Registered prediction:** influence wins the
add-side, abundance/removal effects split, v2 retains a minority
residual; transplantation follows the matrix; surgery separates.

## G5 — Internalized-controller ladder (EXPLORATORY, labeled)

Information-restricted policies on the steering loop (domain 24),
all compared against full v2-down and noop on: maintenance
(inheritance over 60 fissions), recovery (k8 perturbation at fission
30), and generalization (the three transfer regimes):

- L0: memoryless local rule (frozen C3 influence rule).
- L1: L0 gated by one bit (edit only after a non-inherited boundary).
- L2: L0 with run-length modulation (edit only when trailing
  inherited run < 3).
- L3: distilled transparent policy — depth-3 decision tree on local
  features (abundance, influence, R_Q, streak) imitating frozen-v2
  edit choices on dev matrices; frozen before confirmation.

**Kinetic internalization prototype (model extension, appendix).**
Traced-growth variant with leave rates multiplied by
1/(1 + λ · influence_percentile(type)) for λ ∈ {0.1, 0.3} — retention
physics embodying L0 without any editor. Same readouts. Frozen sim
paths untouched.

**Registered prediction.** The ladder is monotone (L3 ≥ L2 ≥ L1 ≥ L0
on maintenance), L3 recovers a large fraction of v2-down's
maintenance (plausibly ≥ 0.7 of the inheritance gain over noop), and
the kinetic prototype shows a real but weaker stabilization —
"a chemistry can embody a crude version of its own controller."

---

# RESULTS (appended per module as completed; nothing above edited)

## G1 RESULTS (2026-08-14)

S-J behavioral positive control: within-basin behavioral return 0.89
(n = 9) — PASS (gate honored). One analysis-stage repair before any
result was seen: NaN-robust z-scaling and masked distances (persist5 is
undefined in episode-free ensembles); raw units now persisted before
analysis (`results_g/g1_units.pkl`).

**Two-target race (k8 arm, t = 10): NO competency return in any
class or candidate — registered prediction CONFIRMED.** Candidate 02:
model_down d(K0) 2.86 vs d(matrix) 2.09 (diff −0.77 CI [−1.32, −0.25]);
model_up −1.17 [−1.59, −0.77]; comp_aligned −0.40 [−0.70, −0.08] — all
decisively matrix-typical. Candidate 03: model_down −0.92
[−1.47, −0.35]; model_up and comp_aligned statistical ties (CIs span
0). Distance to the state's own K(0) GROWS over recovery time
(e.g., model_down 2.2 → 2.9) while distance to the matrix-typical
fingerprint stays flat or falls. Behavior relaxes to the rulebook's
signature; the chemistry retains no state-specific competency memory —
the behavioral counterpart of the F6 distributional attractor, and the
adjudication of H3's strongest form: hereditary capacity is
matrix-level, not state-level.

## G2 RESULTS (2026-08-14)

Students: q_B on 37.5k rows (prev 0.34/0.36), q_R on ~4k post-break
rows (prev 0.85) — frozen in `results_g/g2_students.pkl`.

**Stage A (resistance): PASS both candidates.** Break hazard bu 0.352 /
0.374 vs bd 0.207 / 0.227; up−down +0.145 [+0.110, +0.182] / +0.147
[+0.088, +0.204]; random ≈ noop. Resistance is a clean standalone
knob.

**Stage B (resilience, shared-break-state design): PASS both
candidates — with a decisive reinterpretation.** run3: ru 0.982 / 0.970
vs rd 0.956 / 0.943 (up−down +0.026 [+0.011, +0.044] / +0.027
[+0.009, +0.048]); time-to-renewal 3.35 vs 4.10 / 3.51 vs 4.13.
Renewal after a break is NEAR-CEILING (~0.93–0.98 in every arm,
including random and noop): recovery is nearly a constant of the
chemistry, and the knob can only shave its tail and timing. This
sharpens D1's 54/46 decomposition: the conditional-renewal share there
largely reflected WHICH trajectories break; under the clean design,
**the controllable margin is almost entirely resistance** — the knob is
a stability dial; renewal takes care of itself.

**Stage C (staged controller): registered gate FAIL (both
candidates) — informative failure, prediction partially missed
(program's 4th recorded miss).** Gates: fewer-breaks-than-explore PASS
(1.5 vs 14.9 / 1.6 vs 15.2); longer-survival-than-explore PASS (46–48
vs 5–9 fissions); more-distinct-episodes-than-noop FAIL (staged 1.2
vs noop 2.6–3.0). Mechanistic account: each component works — explore
forces breaks, recover certifies an episode, consolidate locks it
(survival 46–48) — but consolidation is so effective that the frozen
one-cycle state machine never re-arms (breaks stop coming), while noop
drifts through 5–6 natural break-renewal cycles. The staged policy
achieves "one controlled renewal, then near-permanent stability"; to
beat noop on episode COUNT it would need a registered re-explore
trigger — a policy-design property, not a chemistry limit. Reported as
FAIL per the frozen gates; no post-hoc modification.

## G3 RESULTS (2026-08-14)

**Registered verdict: ACCUMULATING-HYSTERESIS — in both candidates;
the registered prediction (sparse-sufficient, no accumulation) MISSED
(program's 5th recorded miss), in the scientifically richer
direction.**

- Pulse ladder: post-release half-life (first anchor-H < 0.7) grows
  with steering duration — 1-fission pulse ≈ 5.2–5.6 fissions;
  8-fission ≈ 9.5–12.4; 16–60-fission ≈ 13–15.5, saturating around
  ~14. Spearman(pulse, half-life) +0.391 [+0.289, +0.492] / +0.369
  [+0.259, +0.490]. **Steering accumulates: the chemistry retains a
  quantitative trace of how long it was held** (deeper-written =
  more concentrated = slower-drifting), tripling the decay half-life
  at saturation. This REVISES Phase E's flat characterization: still
  no permanence, but written depth is a graded, persistent property.
- Periodic control: inheritance 0.982 (every fission) → 0.973/0.966
  (every 2) → 0.944/0.934 (every 4) → 0.917/0.908 (every 8) vs
  budget-matched random floor ≈ 0.89. Graceful degradation; the 0.95
  sparse-sufficiency bar lands between periods 2 and 4.
- **Event-triggered control is the efficiency winner:** at threshold
  q̂ > 0.35, inheritance 0.974/0.977 with only ~16–17 edits per 60
  fissions — ≈ 0.27 corrective edits per maintained generation,
  ~3.5× cheaper than every-fission control for nearly the same
  maintenance. The controller-information rate is quantified: with a
  working risk sensor, one targeted molecule swap every ~3.5
  generations sustains ≈ 0.975 heredity.

## G4 RESULTS (2026-08-14)

Mining: classes A/C/D filled (100 each per candidate); B scarce
(48–49) — v2-stabilizing edits that the influence rule calls
destabilizing are rare, consistent with rule ≈ v2 alignment.

- **Tournament:** only class A reached decision (CI excluding 0):
  realized Δq −0.0294 [−0.0592, −0.0005] / −0.0244 [−0.0478, −0.0043]
  — stabilizing, as v2 AND the influence rule predicted, and
  OPPOSITE to R_Q's prediction. **Flux alignment (R_Q) loses its
  disagreement; v2 and the influence rule remain tied.** Classes
  B/C/D undecided (CIs span 0) — when the mechanisms disagree,
  realized effects shrink, indicating they are partially redundant
  proxies of one underlying quantity.
- **Transplantation:** candidate 02 follows the LOCAL v2 prediction
  (0.71) over the native edit's effect (0.47) — relational
  (network × state) evidence; candidate 03 ties (0.58/0.58).
  Partial support, one candidate decisive.
- **Beta surgery: SUFFICIENCY PASS, both candidates — the program's
  cleanest mechanism result.** At FIXED composition, multiplicatively
  tightening the present-present catalytic edges lowers
  break-and-renewal q (raise 0.292 vs lower 0.417/0.391; raise−lower
  −0.125 [−0.208, −0.049] / −0.099 [−0.172, −0.023]) while
  norm-matched random surgery does nothing (−0.016/−0.003).
  **Catalytic-web tightness is causally sufficient: changing the
  rulebook alone, without touching one molecule, moves heredity in
  the predicted direction.** The dial's physical referent is
  confirmed at both the composition level (Phase A/C3 swaps) and the
  network level (surgery).

## G5 RESULTS (2026-08-14; EXPLORATORY as registered)

Campaigns: home 24 matrices × 2 candidates × 2 reps × 7 policies ×
{maintenance, k8-recovery} (CRN streams shared across arms and
conditions); transfer 3 regimes × 12 matrices × 4 policies; kinetic
prototype 24 matrices × λ ∈ {0, 0.1, 0.3}. L3 trees distilled from
frozen-v2 choices on regenerated 25x dev states and frozen to
`g5_trees.pkl` before any confirmation lineage ran. Raw units
persisted (`g5_home_units.pkl`, `g5_aux_units.pkl`). Suite 24/24.

**Registered deviation (documented).** The preregistration listed L3
tree features as "abundance, influence, R_Q, streak". As implemented,
L3 is two depth-3 per-type trees (remove-side, add-side) on local
per-type features: abundance share, out-influence percentile, in-boost
percentile, presence. R_Q was dropped after G4 eliminated it as a
driver; streak is not a per-type quantity (streak-gating is exactly
policy L2). No other deviations.

**Maintenance / recovery (inheritance over 60 fissions; after k8 at
fission 30), cand 02 / 03:**

| policy | maint 02 | k8 02 | maint 03 | k8 03 | frac-of-v2 (02/03) |
|---|---|---|---|---|---|
| noop | 0.900 | 0.923 | 0.889 | 0.913 | — |
| random | 0.898 | 0.912 | 0.893 | 0.905 | ≈0 (null) |
| L0 influence rule | 0.976 | 0.992 | 0.980 | 0.992 | **0.92 / 0.96** |
| L1 (edit after break) | 0.918 | 0.923 | 0.928 | 0.923 | 0.22 / 0.42 |
| L2 (streak < 3) | 0.935 | 0.944 | 0.938 | 0.944 | 0.42 / 0.51 |
| L3 distilled tree | 0.956 | 0.973 | 0.950 | 0.962 | 0.68 / 0.64 |
| v2_down (full model) | 0.983 | 0.990 | 0.984 | 0.995 | 1.00 |

All L0/L2/L3−noop contrasts have matrix-bootstrap CIs excluding 0 in
both candidates and both conditions; L1's recovery contrast spans 0;
random−noop spans 0 everywhere (control null clean).

**Transfer regimes (maintenance, 12 matrices each):** L0 matches
v2-down everywhere — (−4,5): 0.989/0.997 vs 0.994/0.997; (−3,4):
0.988/0.981 vs 0.992/0.985; (−5,4): **0.975/0.979 vs 0.982/0.975
against noop 0.807/0.826**. L3 degrades out-of-regime (0.881/0.899 at
(−5,4)): the distilled imitation is less portable than the physical
rule it approximates.

**Kinetic prototype (registered appendix):** λ=0 reproduces the
frozen-sim baseline (0.892/0.888 ≈ noop). Effects at λ=0.1/0.3 are
small and CIs span 0 (02: −0.010 [−0.049,+0.033], +0.015
[−0.007,+0.042]; 03: +0.019 [−0.006,+0.046], +0.017 [−0.009,+0.041]).
**No detectable stabilization at the registered λ grid** —
directionally positive in 3 of 4 cells but not resolvable at this
scale. Reported as a null.

**Registered-prediction adjudication (miss #6).** The registered
prediction had three parts. (a) "Ladder monotone, L3 ≥ L2 ≥ L1 ≥ L0"
— **WRONG, anti-monotone: L0 > L3 > L2 > L1.** The memoryless
one-scalar rule applied every generation beats every
information-added variant; event-gated versions (L1/L2) lose most of
the gain. (b) "L3 recovers ≥ 0.7 of v2's gain" — near-miss
(0.68/0.64). (c) "Kinetic prototype shows real but weaker
stabilization" — not confirmed (null at registered λ).

**Interpretation (exploratory, within the registered boundary).**
The controller's knowledge of WHAT to do is almost fully internalized
by one physical scalar computed locally from the current composition
(remove least-connected, add most-connected): 92–96% of the trained
model's maintenance gain, equal recovery, and full regime
portability. What the model still owns is WHEN: G3's risk-triggered
controller kept 0.974 inheritance with ~16 edits/60 fissions, while
crude local triggers (L1/L2) at similar sparsity collapse toward
noop — a good sensor buys sparseness; no sensor demands constant
action. And embodying the rule as a soft retention bias in the
kinetics (no editor) does nothing measurable at small λ: in this
chemistry the policy is trivial to know and cheap to state, but it
must be enacted as discrete compositional edits — the hard part of
internalization is the actuator, not the knowledge.

## G4 BETA-SURGERY RANDOM-ARM AUDIT NOTE (appended 2026-08-15, after an
## external replication query; no code, seeds, or results changed)

Confirmed design (as run): delta = 0.5 multiplicative; raise =
present-present block × 1.5; lower = ÷ 1.5; random arm norm-matched
to the RAISE arm's realized block change (0.5 · ||β[P,P]||_F);
24 matrices × 2 candidates, one generation-60 state each, 16 CRN
branches per arm, horizon 12. The arms are multiplicatively symmetric
(± log 1.5 in log-space, positivity- and shape-preserving) and
therefore Frobenius-asymmetric (lower/raise block-change ratio
exactly 2/3); the registered contrast raise−lower does not require
Frobenius equality, and the random arm was matched to the larger arm.
The random−none null was reported descriptively (matrix-bootstrap CI
crossing zero); no TOST equivalence test was run, and no larger
surgery cohort exists (the 200-matrix cohort is Phase H,
occurrence-only).

Post-hoc deterministic audit of the random arm (exact regeneration of
all 48 surgery units from their seeds; no new futures): achieved/
requested Frobenius ratio median 0.916, range 0.030–1.000; distinct
changed edges median 355 of 361 drawn (~1.7% duplicate draws,
resolved last-write-wins, not cumulative); norm-matching scale factor
median ×21.8, so roughly the negative half of perturbed entries hits
the positivity clip at 1e-12 (~110 entries/unit), which is the
dominant cause of the norm shortfall. Characterization: the random
control is an UNSTRUCTURED-LOCATION, APPROXIMATELY norm-matched null
(random edges drawn over the full 100×100 matrix, of which the
present-present block is ~3.6%), not a strictly audited same-block
null. The raise/lower causal result is unaffected; the recommended
stricter control for replications is a sign-shuffled random surgery
WITHIN the present-present block with post-clip norm audit.

Related instruction erratum (external): a third-party instructions
document specified a 0.05 · ||β[P,P]|| Frobenius perturbation; no
such number exists in this repo's code or registration (likely
δ = 0.5 misread as 5%). Small-dose runs under that spec are a new
point on the surgery dose axis, not a replication of the registered
δ = 0.5 intervention.
