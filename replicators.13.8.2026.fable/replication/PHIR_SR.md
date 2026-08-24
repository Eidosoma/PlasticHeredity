# Phase M — self-replication-state reading versus organizational
# reading
# (preregistered 2026-08-18, BEFORE any measurement ran)

Motivating external observation (the paper-replication agent's
result, used as motivation only, never as a fitting target): in
their rebuild of the paper's own analysis, the PRINTED formula reads
the "currently self-replicating" (SR) state well (78/100 runs
positive vs the paper's 73/100) while PhiRL's Φ_R fails at it
(31/100) — the exact mirror of our chapter-5 result, where Φ_R
responds to heredity interventions and the printed formula does not.
Phase M tests whether this split is real WITHIN ONE SYSTEM by
running both comparison styles on the same lineages:

- BETWEEN-ARM (our style): does the instrument differ across
  stabilized vs destabilized lineages?
- WITHIN-LINEAGE (the paper's style): does the instrument read
  higher DURING self-replicating stretches than outside them, inside
  a single lineage?

**Registered named hypothesis (the user's):** where heredity is
strong, replicator detection should work — SR discrimination should
be present and STRONGER in stabilized arms than destabilized ones.

**Registered candidate resolution (ours):** the two formulas read
different variance components — Φ_R reads slow between-lineage
organizational tightness; the printed (redundancy-sensitive)
structure reads fast within-lineage SR episodes. If so: within-
lineage ΔSR favors the printed-structure instrument while the
between-arm contrast favors Φ_R, on identical data — dissolving the
mystery without contradiction.

## Design

Byte-exact replay of Phase J's ph_stab, ph_destab, random, noop arms
(domain 28 untouched; 768 lineages) with per-update recording over
ALL 60 fissions and per-fission record markers. REPLAY GATE: code-
Φ_R on the fissions-41–60 sub-record must equal stored Phase J
values exactly.

Per lineage, sliding 3-generation windows (center g = 2…59, step 1):
one pointwise ΦID on the macro-averaged Fiedler halves of each
window's update series (the tractable pipeline; window T ≈ 36–160),
yielding per window: Φ_R (9-atom sum), PRINTED-STRUCTURE typeset
quantity (source-synergy atoms minus source-redundancy atoms — the
exact typeset formula on the macro pipeline; the full multivariate
variant is infeasible at window length and Phase L showed it
noise-dominated), bare synergy (sts), and "emergence" (synergy +
causation). Windows with < 20 update steps are NaN.

SR labels: the paper never states its SR metric (registered caveat);
house standard: fission g is SR iff it lies inside a maximal run of
consecutive inherited boundaries (H > 0.9) of length ≥ L, with
L = 5 primary and L ∈ {3, 8} secondary. A window inherits its center
generation's label.

## Registered tests (matrix bootstrap, 4,096 draws, seed 29;
## candidates never pooled)

- **M1 (within-lineage SR reading, per instrument):** ΔSR = mean
  over SR windows − mean over non-SR windows, per lineage (defined
  only when both classes exist); arm noop, L = 5 primary. CI
  excluding 0 = that instrument reads the SR state.
- **M2 (the named hypothesis):** ΔSR in ph_stab vs ph_destab, per
  instrument: positive difference = SR detection strengthens with
  heredity strength.
- **M3 (consistency):** between-arm window-mean Φ_R contrast
  (ph_stab − ph_destab) must reproduce Phase J's C1 sign, and the
  printed-structure between-arm contrast its Phase-K null — the
  same data must reproduce the chapter's established results.
- Descriptive: SR occupancy per arm and L; per-window instrument
  correlations; random − noop specificity on ΔSR.

## Registered predictions

- SR occupancy: much higher in ph_stab (long inherited runs ARE the
  SR state under this label — the operational kinship of "self-
  replicating" and "high heredity" is itself a registered
  observation of this module).
- M1 lean: the printed-structure instrument shows ΔSR > 0 (its
  redundancy-penalizing reading aligns with episodic SR); Φ_R's ΔSR
  weak or null (it reads slow organization, not fast episodes).
  Registered two-sided for all instruments.
- M2 (user's hypothesis): lean POSITIVE for whichever instrument
  passes M1.
- M3: must pass, else the module is invalid.

## Boundary

The SR label is our reconstruction (the paper's metric is unstated);
the printed-structure instrument is the typeset formula on the
tractable macro pipeline, not the infeasible multivariate one; the
external agent's numbers motivated the design and play no role in
any gate. Sealed: this file, `run_phir_sr.py`, `phir_code.py`,
`run_phir_confirm.py`, `run_phir_dose.py`, `sim.py`, `cohort.py`
(SHA-256 in `results_phir_sr/SEAL.json`).

---

# RESULTS (appended 2026-08-18; nothing above edited)

Replay 316 s; **REPLAY GATE PASS (0 mismatches)**; raw units
`results_phir_sr/phir_sr_units.pkl`; suite 33/33 at seal.

## SR occupancy (registered observation confirmed)

Under the L=5 label, SR occupancy tracks the heredity dial exactly:
ph_stab 0.89/0.90 > noop 0.82/0.80 > ph_destab 0.68/0.63.
"Currently self-replicating" and "high heredity" are operationally
kin under this labeling — but note occupancy is HIGH (0.8 natural):
our label reads "not currently broken," which may be laxer than the
paper's rarer locked-compotype episodes (registered caveat, now
quantified).

## M3 (consistency gate): PASS in full

On the same windows, the chapter's established results reproduce:
between-arm Φ_R +0.131 [+0.101,+0.163] / +0.153 [+0.123,+0.184]
(Phase J's sign); printed-structure between-arm null (+0.007/+0.009);
synergy negative; emergence positive (all matching K2). Module valid.

## M1 (within-lineage SR reading, natural lineages):
## Φ_R READS THE SR STATE — registered lean WRONG (ledger miss #9)

- **Φ_R: ΔSR positive in BOTH candidates** (+0.099 [+0.055,+0.149] /
  +0.138 [+0.094,+0.188]).
- emergence: +0.043 [−0.003,+0.091] (02, marginal) /
  +0.143 [+0.092,+0.197] (03).
- printed-structure: null in 02 (−0.003), modest positive in 03
  (+0.049 [+0.014,+0.081]).
- synergy: NEGATIVE during SR in both (−0.083/−0.044, CIs
  excluding 0).

Our registered "different variance components" resolution is
REFUTED in this system: Φ_R reads BOTH the slow between-arm
organization AND the fast within-lineage episodes. The
printed-structure quantity is the weaker episode-reader here, not
the stronger one.

## M2 (the named hypothesis): PARTIAL

SR detection stronger under strong heredity: candidate 03 YES for
Φ_R (+0.072 [+0.026,+0.116]) and emergence (+0.064 [+0.022,+0.107]);
candidate 02 null (Φ_R −0.028 [−0.091,+0.035]) and REVERSED for the
printed structure (−0.043 [−0.080,−0.004]). Interpretive caveat:
in stab arms SR occupancy reaches 0.89–0.90, so non-SR windows are
scarce — ceiling/class-imbalance effects bias ΔSR estimation
against the stabilized arms, which the registration did not
anticipate.

## What this does to the cross-agent mystery

The mirror-split does NOT reproduce inside our system: here the
code's Φ_R is a competent reader of the self-replicating state in
the paper's own comparison style. Therefore the paper-replication
agent's GARD finding (Φ_R fails at 31/100 while the printed formula
succeeds at 78/100) cannot be explained by any intrinsic inability
of Φ_R to read episodic states. The unresolved difference must live
in: (i) the SR detector (our high-occupancy "unbroken run" label vs
their presumably rarer locked-compotype episodes — the one
quantified divergence), (ii) their GARD configuration/dynamics, or
(iii) which formula actually generated the paper's figures. These
are now precise questions for Dr. Pigozzi rather than testable
unknowns on our side.
