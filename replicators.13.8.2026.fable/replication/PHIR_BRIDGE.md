# Phase I — Reciprocal causal bridge between reconstructed Φ-r and
# plastic heredity
# (preregistered 2026-08-17, BEFORE any campaign lineage ran)

Candidate chapter for the current pre-print. One fresh six-arm
campaign asking whether the paper's causal-architecture coordinate
(Φ-r, causal emergence) and our plastic-heredity coordinate are
coupled CONTROL directions — prediction and control being different
properties.

**Registered boundary (verbatim from the design):** this experiment
does not test whether Φ-r predicts replicator onset and cannot
adjudicate the unavailable target implementation. A negative
cross-effect constrains only OUR frozen reconstructed Φ-r
(implementation mismatch remains possible); a positive cross-effect
is informative even before the authors' code is available.

## Frozen reconstructed Φ-r (`phir.py`)

From the target paper's Methods:
Φ_r = [I(X_t;X_{t+1}) − Σ_parts I(X^part_t;X_{t+1})] / I(X_t;X_{t+1})
on centered-log-ratio-transformed relative molecular compositions,
with the parts given by the minimum-information bipartition (MIB),
Gaussian estimation throughout. Registered reconstruction choices
(the paper is not fully prescriptive): top-k = 8 species by mean
count over the analyzed window; pseudocount 0.5 before the relative/
CLR transform; drop the last CLR component (the paper's full-rank
fix); MIB = argmin over all bipartitions of the retained components
of I(A;B)/min(|A|,|B|), instantaneous, Gaussian; ridge 1e-8 on
covariances; Φ_r undefined (NaN, excluded) if the window has < 20
snapshots or total predictive information < 1e-9.

**Empirical Φ-r (outcome measure):** computed on the composition
snapshot recorded after EVERY molecular update (each Gillespie event
for candidate 02, each vector-Poisson step for candidate 03 — the
paper's own time base, "ntot molecular steps"), concatenated across
the measurement window into one per-lineage series, one Φ-r per
lineage. [Amended pre-campaign, before any lineage ran: the first
draft registered per-growth-phase series on a unit-mass grid; a
pilot fixture showed candidate 03 growth completes in ~12 Poisson
steps, below the registered 20-sample minimum, making per-phase
estimates undefined. The concatenated per-update series is closer to
the paper's stated substrate and was adopted before the campaign.]
Evaluated ONLY on trajectories downstream of the edits and never
used to select them (non-circularity requirement).

**Surrogate Φ-r (edit-selection score only):** Φ_r of the one-step
Gaussian linearization of the expected growth flow at the candidate
state: J_ab = KF·ρ·(1+β_ab) − KB·(δ_ab·bn_a + n_a·(β_ab·N − c_a)/N²)
over the top-k present species; A = I + τJ with τ = 0.5/ρ(J);
X_0 ~ N(0,I), X_1 = A·X_0 + ε, ε ~ N(0, 0.1²·I); MIB on cov(X_1);
same Φ_r formula. Whether the surrogate actually moves empirical
Φ-r is itself a registered manipulation check, not an assumption.

## Campaign (domain 27, tag "phir-bridge-2026-08-17")

Fresh cohort: 24 new matrices × 2 candidates × 2 reps; 60 steering
fissions; one edit after every fission. **Identical legal action set
across arms:** at every decision point a CRN panel of 24 legal
mass-preserving swaps (remove a present type, add another) is drawn
from an arm-independent stream; every controller chooses from the
SAME panel. Growth/fission streams are also arm-independent (CRN).

| arm | selection rule on the shared panel |
|---|---|
| ph_stab | frozen v2: minimize predicted break-and-renewal risk |
| ph_destab | frozen v2: maximize predicted risk |
| phir_max | maximize surrogate Φ-r of the edited state |
| phir_min | minimize surrogate Φ-r |
| random | uniform pick from the panel |
| noop | no edit (also logs both scorers on the panel for T3) |

Outcomes: Plastic-H primary = inherited-boundary fraction, break
count, longest inherited run over the 60 fissions; F12 secondary
(joint break-and-run3 over fissions 49–60; secondary because
successful stabilization suppresses its required break). Realized
Φ-r = mean per-phase empirical Φ-r over fissions 41–60 (20 traced
growth phases, post-burn-in, downstream of all prior edits).

## Registered tests (matrix bootstrap, 2,048 draws, seed 13;
## candidates never pooled; all five must be reported)

- T1 Plastic-H → Φ-r: ph_stab − ph_destab on realized Φ-r; CI
  excluding 0 in both candidates = pass (either sign).
- T2 Φ-r → Plastic-H: phir_max − phir_min on inherited fraction
  (break count and longest run co-primary); CI excluding 0 in both
  candidates = pass (either sign).
- T3 action agreement: on noop states, Spearman between (−v2 risk)
  and surrogate Φ-r over each 24-swap panel; plus top-choice
  agreement rate. Descriptive.
- T4 random specificity: random − noop ≈ 0 on both outcome families
  (CIs spanning 0 required for experiment validity).
- T5 candidate agreement: any claimed direction must pass separately
  in both candidates.
- Validity arm: ph_stab − ph_destab must move heredity (the known
  Phase C1 effect); if it does not, the campaign is invalid and no
  Φ-r conclusion is drawn.
- Manipulation check: phir_max − phir_min on realized Φ-r (is the
  surrogate a working handle on the empirical quantity at all?).

## Registered predictions

- Validity arm passes (strong prior from C1/G5).
- T1 lean: DESTABILIZING raises realized Φ-r (Φ-r as a
  reorganization/transition signature rather than a stability
  signature); registered two-sided.
- T2 lean: null (the surrogate gradient is weak relative to the v2
  knob); registered two-sided.
- T4: null required.
- T3: weak agreement (|ρ| < 0.3) expected.

## Interpretation table (frozen verbatim)

Both cross-directions pass → Φ-r and plastic heredity are coupled
architecture- and process-level control coordinates. Plastic moves
Φ-r but Φ-r edits don't control heredity → Φ-r is a responsive
signature, not a sufficient controller. Φ-r edits control heredity
but Plastic edits don't move Φ-r → reconstructed Φ-r contains an
actionable direction not captured by the Plastic-H score. Neither →
independent organizational axes (constrains only this frozen
reconstruction). Effects reversed from lean → Φ-r marks transition/
reorganization rather than stable heredity.

---

# RESULTS (appended 2026-08-17; nothing above edited)

Campaign 517 s on 12 workers; 576 lineages; raw units in
`results_phir_bridge/phir_bridge_units.pkl`. Suite 27/27 at launch.

## Arm table (candidate 02 / 03)

| arm | inherit | breaks | longest run | realized Φ-r |
|---|---|---|---|---|
| ph_stab | 0.966 / 0.964 | 2.1 / 2.2 | 49.4 / 47.4 | −0.0156 / −0.0324 |
| ph_destab | 0.781 / 0.790 | 13.2 / 12.6 | 21.8 / 23.4 | −0.0204 / −0.0228 |
| phir_max | 0.900 / 0.897 | 6.0 / 6.2 | 33.1 / 32.5 | −0.0168 / −0.0257 |
| phir_min | 0.884 / 0.865 | 6.9 / 8.1 | 31.1 / 26.6 | −0.0117 / −0.0197 |
| random | 0.907 / 0.900 | 5.6 / 6.0 | 33.6 / 33.2 | −0.0190 / −0.0281 |
| noop | 0.907 / 0.901 | 5.6 / 5.9 | 34.8 / 32.6 | −0.0151 / −0.0347 |

## Registered tests

- VALIDITY (ph_stab − ph_destab, inherit): **+0.185 [+0.135,+0.235] /
  +0.174 [+0.128,+0.219] — PASS both.** The campaign is valid.
- T4 random − noop: CIs span 0 on both outcome families in both
  candidates — PASS (edit-specificity intact).
- **T1 Plastic-H → Φ-r: FAIL (null), both candidates.** ph_stab −
  ph_destab on realized Φ-r: +0.0047 [−0.0050,+0.0156] /
  −0.0096 [−0.0241,+0.0050]; signs inconsistent, CIs span 0. A
  0.17–0.19 causal swing in inherited fraction (2 vs 13 breaks)
  leaves realized Φ-r unmoved.
- **T2 Φ-r-surrogate → Plastic-H: passes in candidate 03 only; T5
  (both candidates) therefore FAILS the registered gate.** 03:
  inherit +0.0326 [+0.0080,+0.0604], breaks −1.96 [−3.63,−0.48],
  longest run +5.81 [+1.55,+10.62] — all three co-primaries. 02:
  directionally consistent, CIs graze/include 0 (+0.0160
  [−0.0010,+0.0333]; −0.96 [−2.00,+0.06]; +2.02 [−2.18,+6.48]).
- **Manipulation check: FAIL — the surrogate does not move realized
  Φ-r** (−0.0051 [−0.0135,+0.0031] / −0.0060 [−0.0186,+0.0061]). So
  the 03 heredity effect above cannot be attributed to Φ-r; it is a
  property of the surrogate (the linearized-growth-flow structure),
  not of the measured quantity.
- T3 agreement: Spearman 0.074 / 0.089; top-choice 9.7% / 11.2%
  (chance 4.2%; n=528 panels each) — the two scorers select nearly
  independent actions, as registered (|ρ| < 0.3).

## Adjudication under the frozen interpretation table

The row that fires is **"neither cross-effect passes → independent
organizational axes"**, with the registered caveat that a negative
constrains only THIS frozen reconstruction, not the authors'
implementation. Supporting observations, both exploratory:
(1) realized Φ-r is slightly NEGATIVE everywhere (−0.01 to −0.03):
under this reconstruction the steady-heredity regime shows no
positive causal emergence at all, and Φ-r is insensitive to the
largest heredity manipulation we can produce; (2) the surrogate
contains a weak heredity-relevant control direction (decisive in 03,
directional in 02) that is nearly orthogonal to the v2 knob in
action space (T3) and NOT mediated by realized Φ-r (manipulation
check) — an unexplained lead, explicitly not a Φ-r effect.

## Prediction scorecard

T1 lean (destabilization raises Φ-r): not confirmed — null with
inconsistent signs. T2 lean (null): held in 02, refuted in 03.
T4 null: confirmed. T3 weak agreement: confirmed. Registered
boundary reiterated: this experiment does not test whether Φ-r
predicts replicator onset and cannot adjudicate the unavailable
target implementation.

**Erratum note (2026-08-17, append-only):** the Campaign section
above retains a stale pre-amendment sentence ("mean per-phase
empirical Φ-r … 20 traced growth phases"). The outcome actually
computed is the amended registered definition (Empirical Φ-r
paragraph): ONE Φ-r per lineage on the concatenated per-update-step
series of fissions 41–60. The amendment predates the campaign; the
stale sentence was an editing oversight, left in place to preserve
the append-only record.


[Filing repair 2026-08-18: the ADDENDUM registration and port-equality record below were written on 2026-08-17, BEFORE the addendum campaign ran, but a working-directory slip appended them to a stray repo-root copy of this file instead of this one. They are restored here in their correct chronological position, content unmodified; the stray copy is deleted. The registration's timing claim is unaffected: the addendum campaign log (2026-08-17) postdates the misfiled write.]
---

# ADDENDUM (registered 2026-08-17, AFTER the campaign above and BEFORE
# any addendum measurement ran): code-faithful Φ-r re-measurement

**Trigger.** Inspection of the authors' public repository
(github.com/pigozzif/PhiRL, `information.py`/`main.py`) shows their
implemented Φ-r is NOT the formula printed in the paper's Methods
(and reconstructed above). Three text-vs-code discrepancies: (i) the
code's quantity decomposes against parts predicting their OWN
futures, not the whole's; (ii) it adds back double-redundancy —
algebraically it is exactly Mediano et al.'s revised Φ_R, the sum of
the nine ΦID atoms with synergy on either side plus the two
cross-transfers (identity verified from first principles:
Φ_WMS = Σ(those 9 atoms) − rtr, so Φ_R = Φ_WMS + rtr = the 9-atom
sum, matching the code's PHIR_ATOMS set); (iii) it is unnormalized.
Pipeline differences besides the formula: spectral (Fiedler-vector)
minimum-information bipartition of the lag-1 pairwise-MI graph over
ALL surviving channels (not exhaustive top-8); each half then
MACRO-AVERAGED to a single scalar; pointwise (local) Gaussian
entropies with pointwise-MMI redundancy and Möbius inversion on the
16-atom lattice. The Phase I conclusions above therefore constrain
the PRINTED formula; this addendum re-measures the same frozen
campaign with the CODE-faithful instrument.

**Registered code-faithful Φ-r (`phir_code.py`).** Per-lineage
per-update composition series (identical substrate and window as
above) → pseudocount 0.5 → relative → CLR (the GARD paper's
preprocessing) → drop channels with std < 1e-8 → z-score per channel
(their `preprocess_data`) → lag-1 Gaussian pairwise-MI matrix,
alpha=1 (no significance masking), symmetric averaged lagged
correlation (their `mutual_information_matrix_fast`) → Fiedler
bipartition with 1e-6 noise floor, halves = strictly positive /
strictly negative entries (their `minimum_information_bipartition`;
implemented with a deterministic dense-eigendecomposition Fiedler
vector; equality with their networkx path verified on real windows
before the campaign, result recorded below) → macro-average halves →
pointwise 2×2 ΦID with local Gaussian entropies and pointwise-MMI
redundancy, Möbius inversion (their `local_phi_id`; lattice
re-derived from the product order and checked against their
`phi_lattice_22.pickle`) → Φ-r = mean over timepoints of the 9-atom
local sum (their `local_phi_r`; aggregation to the mean is our
registered choice — the repo saves the raw local vector).

**Design.** Exact deterministic replay of all 576 Phase I lineages
(same seeds, same panels, same selections — selection scorers
untouched). REPLAY GATE: replayed heredity outcomes and text-formula
Φ-r must equal the stored units exactly; the only new quantity is
code-Φ_R on the same recorded series. Registered tests, same
machinery (matrix bootstrap 2,048 draws, seed 13, candidates never
pooled): T1code = ph_stab − ph_destab on code-Φ_R; MANIPcode =
phir_max − phir_min on code-Φ_R; T4code = random − noop on code-Φ_R.
Descriptive: per-arm code-Φ_R levels (sign!), correlation of text vs
code measures across lineages.

**Registered predictions.** Code-Φ_R will be substantially POSITIVE
(the +rtr term restores the large double-redundancy of two
macro-averaged halves of one assembly). T1code lean: null again
(two-sided, lower confidence than the original registration).
T4code: null required. No prediction on MANIPcode (the surrogate was
built for the text formula).

**Boundary unchanged:** even this code-faithful measurement uses OUR
port of their public code with two registered choices (CLR upstream
per the GARD paper's text; mean aggregation), and cannot adjudicate
their private GARD-paper pipeline.

**Port-equality verification (run before the addendum campaign,
2026-08-17).** Against the authors' own `information.py` (networkx
path, their `phi_lattice_22.pickle`): lattice descendant sets
identical (16/16 nodes); all 16 pointwise atom vectors equal to
≤ 1.4e-14 over 20 random trials; local Φ-r vectors equal to
≤ 9.8e-15; lag-1 MI matrices bit-identical; Fiedler bipartitions
identical (as unordered pairs) on 50/50 block-structured matrices.

## ADDENDUM RESULTS (appended 2026-08-17; nothing above edited)

Replay 560 s; **REPLAY GATE PASS: 0 of 576 units mismatch** the
stored campaign on heredity outcomes and text-formula Φ-r — the
experiment is byte-identical; only the instrument changed. Raw units
`results_phir_bridge/phir_code_units.pkl`; suite 29/29.

**Levels (registered prediction confirmed):** code-Φ_R is strongly
POSITIVE everywhere — 1.70–1.91 (cand 02), 0.97–1.23 (cand 03) —
versus the text formula's −0.01..−0.03 on the SAME series. The two
instruments are nearly uncorrelated across lineages (Pearson
+0.086 / +0.129): they are different quantities in practice, not
just on paper.

**T1code (heredity → code-Φ_R): PASS, BOTH candidates.**
ph_stab − ph_destab: **+0.2084 [+0.1336, +0.2785] /
+0.2555 [+0.1212, +0.3921]** — consistent signs, CIs far from zero.
Stabilizing heredity RAISES the authors' implemented causal
emergence; destabilizing lowers it. T4code: random − noop spans 0 in
both candidates (edit-specific). MANIPcode: the (text-formula-
targeted) surrogate does not move code-Φ_R (+0.012 / −0.011, CIs
span 0) — consistent with the near-zero text↔code correlation.

**Revised adjudication (supersedes the main-campaign row, which
holds for the PRINTED formula only).** Under the authors'
implemented definition, the frozen interpretation table's row is:
**"Plastic-H interventions move Φ-r, but Φ-r edits do not control
heredity: Φ-r is a responsive signature, not a sufficient
controller"** — with one precision: the reverse direction is
UNTESTED rather than failed, because no working code-Φ_R-maximizing
controller exists (the surrogate targeted the text formula and moves
neither heredity consistently nor code-Φ_R). Summary across both
instruments: the printed formula is causally decoupled from heredity
(flat, negative, redundancy-dominated); the implemented Φ_R is
causally DOWNSTREAM of hereditary organization — the first causal
coupling of a ΦID quantity to hereditary stability in this system —
and the coupling is invisible under the paper's printed equation.

**Prediction scorecard (addendum).** "Code-Φ_R positive" —
confirmed. "T1code lean null" — **WRONG (program prediction-miss
#7):** the chemistry answered with a robust positive coupling, and
with the OPPOSITE sign to the original Phase I lean (Φ-r tracks
organization/stability, not transition). Boundary unchanged: our
port of their public code, two registered choices (CLR upstream,
mean aggregation); their private GARD-paper pipeline remains
unadjudicated.

## TYPESET-FORMULA ERRATUM AND ADJUDICATION (appended 2026-08-18;
## nothing above edited)

Close-up inspection of the paper's typeset equation (page image, not
text extraction) shows the printed definition is the UNNORMALIZED
difference Φr = I(X_t, X_{t+1}) − Σᵢ I(Xⁱ_t, X_{t+1}) — no fraction
bar. The "denominator" in our Methods reading was a text-extraction
artifact: "Φʳ =" sits on its own centered line, the difference on
the next, and the following left-margin sentence "Where" +
centered I(X_t, X_{t+1}) (the term being defined) stacks under the
equation in extracted text, manufacturing a phantom denominator.
The taxonomy of "Φ-r" is therefore THREE readings of one page:

1. TYPESET: unnormalized whole minus parts-to-whole-future
   (a Rosas-style Ψ-numerator with the whole as macro).
2. EXTRACTED: the same numerator divided by I(X_t;X_{t+1}) — the
   phantom-denominator reading; this is what the frozen `phir.py`
   faithfully implemented and what all "text-formula" results above
   measured.
3. CODE: revised Φ_R (9-atom ΦID sum, parts-to-own-future, +rtr,
   macro-averaged Fiedler halves) — the responsive instrument.

**Adjudication of the TYPESET formula (no new futures).** On the
two-part macro pipeline, the typeset quantity is an exact linear
combination of the stored Phase K atoms: I(X;X′) − I(m1;X′) −
I(m2;X′) = Σ(synergy-source atoms) − Σ(redundancy-source atoms).
Computed per lineage from `results_phir_dose/phase_k_units.pkl`
(Phase J replay, replay-gate-exact): arm means are small and
negative everywhere (stab −0.033/−0.018; destab −0.027/−0.003;
noop −0.029/−0.016), and the stab − destab contrast is NULL in both
candidates: **−0.0059 [−0.0305, +0.0183] / −0.0152 [−0.0416,
+0.0111]** (matrix bootstrap, 4,096 draws, seed 19, 48 matrices).

**Consequences.** No scientific conclusion changes: the typeset and
extracted readings are both flat, slightly negative, and
unresponsive to the strongest available heredity manipulation —
normalization only rescaled a null — while the authors' code
quantity remains the sole responsive instrument. Phase I's
conclusion stands with the instrument relabeled ("printed formula"
= both the typeset and extracted readings, now each measured
directly). For future replicators: anyone reconstructing Φ-r from
the PDF via text extraction will inherit the phantom denominator;
anyone reconstructing from the typeset page will get the
unnormalized difference; neither is what the code computes. The
three-way divergence is recorded for the authors' attention.
