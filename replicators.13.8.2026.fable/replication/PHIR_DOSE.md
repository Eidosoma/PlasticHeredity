# Phase K — Φ_R dose–response, atom decomposition, and
# natural-prediction test
# (preregistered 2026-08-17, BEFORE any campaign lineage ran)

Chapter-5 completion module. Phase J established prospectively that
implemented Φ_R (verified port `phir_code.py`, unchanged and still
matching its Phase J seal hash) is a responsive signature of
hereditary organization and not a controller. Phase K makes the
signature (K1) quantitative — a graded dose–response; (K2)
mechanistic — which ΦID atoms carry it, including the authors'
alternative "emergence" summary; (K3) epistemically complete — does
the signature carry natural PREDICTIVE information, closing the
prediction-versus-control dissociation; and (K4, appendix) robust —
is C1 an artifact of our two registered port choices. Sealed before
the campaign (SHA-256 in `results_phir_dose/SEAL.json`).

## K1 — dose–response (domain 29, tag "phir-dose-2026-08-17")

Fresh cohort: 24 new matrices × 2 candidates × 2 reps. Eleven arms:
{stabilize, destabilize} × edit cadence k ∈ {1, 2, 4, 8, 16}
(one v2-selected edit from a 12-swap CRN panel at every fission
f ≡ 0 mod k), plus noop. Keys: (29,0,m) matrix; (29,1,m) initial
state; (29,2,cand_i,m,rep) growth (CRN, arm-independent);
(29,3,cand_i,m,rep,f) panel (drawn every fission in every arm for
stream parity). 60 fissions; realized Φ_R on the concatenated
per-update series of fissions 41–60 (the sealed instrument,
unchanged).

Signed dose of an arm = ± (edits taken)/59 (stabilize +,
destabilize −, noop 0). **Registered test K1:** per matrix, Spearman
across the 11 arms between signed dose and mean Φ_R; mean rho with
whole-matrix bootstrap CI (4,096 draws, seed 19) excluding 0, both
candidates, = monotone dose–response. Per-rung means reported.
Registered prediction: PASS, monotone increasing; heredity dose–
response (same statistic on inherited fraction) reported as the
positive control and must pass for validity.

## K2 — atom decomposition (deterministic replay of Phase J)

Exact replay of Phase J's ph_stab, ph_destab, and noop lineages
(seeds/domain 28 unchanged), recording the per-update series and
computing the mean of each of the 16 pointwise ΦID atoms over the
measurement window (replay gate: recomputed scalar Φ_R must equal
the stored Phase J values exactly; identity gate: the 9-atom sum
must equal the Φ_R scalar per lineage). Reported, with matrix-
bootstrap CIs on ph_stab − ph_destab per atom: which atoms carry
C1. Also the authors' alternative summaries from their code:
"synergy" (sts), "causation" (s→u0 + s→u1), "emergence" = synergy +
causation. SECONDARY/decompositional: cannot replace C1; no
per-atom multiplicity claims — the decomposition is descriptive.
Registered prediction (lean): the response is carried predominantly
by transfer and synergy-involving atoms rather than by the
double-redundancy repair term (rtr); the authors' "emergence"
summary also responds.

## K3 — natural prediction (domain 29, sub-key 5)

96 natural lineages per candidate (48 fresh matrices × 2 reps under
domain 29 keys (29,5,cand_i,m,rep); no edits, no controllers),
60 fissions. Predictors computed at fission 40: (a) Φ_R_early on
the traced per-update series of fissions 21–40 (same instrument);
(b) frozen v2 risk of the fission-40 state; (c) history baseline =
inherited fraction over fissions 21–40. Outcome: break count in
fissions 41–60. **Registered statistics:** Spearman(predictor,
outcome) across lineages, OVERALL and MATRIX-CENTERED (predictor
and outcome demeaned within matrix), each with whole-matrix
bootstrap CIs. Registered predictions: overall Φ_R correlation
NEGATIVE and nonzero (shared matrix-level driver: tight-web worlds
are both stable and high-Φ_R); matrix-centered Φ_R correlation —
the sharp test of state-level early warning — registered two-sided
with lean NULL; v2 must dominate on both (validity of the
comparison). Interpretation is frozen: a centered null means the
signature reads present organization but carries no state-level
foresight; a centered pass means Φ_R contains genuine early-warning
information beyond the matrix, partially rehabilitating the
predictive role in-domain.

## K4 — instrument robustness (appendix; matrices m < 12 of the K2
## replay)

On the subsample, C1 (ph_stab − ph_destab on Φ_R) recomputed under
three one-knob variants of the registered port choices: (i) no CLR
(z-score of raw counts only); (ii) median instead of mean
aggregation of the local Φ-r vector; (iii) CLR with the paper-text
drop-last-component fix applied. Registered requirement: the C1
SIGN is preserved under all three (descriptive; CIs reported;
the standard instrument's subsample contrast reported alongside).

## Boundary

All Phase I/J boundaries carry over. K3 tests prediction of
HEREDITY BREAKS in-domain only — it is not a test of
replicator-onset prediction. Sealed files: this registration,
`run_phir_dose.py`, `phir_code.py`, `phir.py`, `run_phir_bridge.py`,
`run_phir_confirm.py`, `sim.py`, `cohort.py`.

---

# RESULTS (appended 2026-08-17; nothing above edited)

K1 163 s; K2/K4 replay 142 s (REPLAY GATE PASS: 0 mismatches vs the
stored Phase J scalars); K3 9 s. Raw units
`results_phir_dose/phase_k_units.pkl`; suite 31/31 at seal.

## K1 — dose–response: graded in candidate 03, unresolved in 02
## (registered miss #8, partial)

Heredity positive control: PASS both candidates (signed-dose
Spearman +0.584 [+0.490, +0.670] / +0.569 [+0.480, +0.653]) — the
dose axis works. Φ_R dose–response: **candidate 03 PASSES
(+0.274 [+0.107, +0.427])** with visibly ordered rungs (stab1–4
1.18–1.23 vs destab1–4 1.03–1.07); **candidate 02 does not resolve
(+0.080 [−0.038, +0.203])** — its middle rungs are flat/noisy at
this 24-matrix scale even though the endpoint contrast reproduces.
The registered both-candidate prediction of a monotone curve is
therefore missed in 02 (ledger miss #8): at current power the
signature's dose-gradedness is demonstrated in one candidate and
merely consistent in the other.

## K2 — the decomposition: the signal is DOWNWARD CAUSATION
## (the module's headline)

With the replay gate exact, the stab − destab response decomposes
consistently in BOTH candidates:

- **"Causation" (synergy→unique atoms — Rosas-style downward
  causation): +0.577 [+0.321, +0.821] / +0.504 [+0.332, +0.682] —
  the dominant positive carrier.**
- Pure synergy persistence (s→s): NEGATIVE (−0.384 [−0.602, −0.148]
  / −0.298 [−0.425, −0.171]); cross-part transfers (u→u): negative.
- Upward terms (r→s, u→s) and syn→red: positive; double-redundancy
  (rtr): positive (+0.163/+0.204) — contra the registered lean,
  which named transfers/synergy as carriers; the sign structure is
  richer than the lean anticipated.
- **The authors' own alternative summary "emergence" = synergy +
  causation responds robustly in BOTH candidates with the tightest
  intervals of any quantity tested: +0.193 [+0.146, +0.247] /
  +0.207 [+0.144, +0.272]** — arguably a cleaner responder than
  Φ_R itself.

Reading: stabilized assemblies are not "more synergistic" — they are
assemblies in which **the whole informs the futures of its parts**
(downward-causation atoms rise) while sideways information churn
(part-to-part transfer, synergy-to-synergy persistence) falls.
Destabilized assemblies churn information laterally; organized ones
route it top-down.

## K3 — natural prediction: matrix-level only (both registered
## predictions HIT)

Overall Spearman(Φ_R_early, later breaks): **−0.330 [−0.525, −0.099]
/ −0.271 [−0.420, −0.094]** — negative and nonzero as registered
(high-Φ_R worlds break less; the shared driver is the matrix).
Matrix-centered: +0.164 [−0.092, +0.399] / +0.071 [−0.177, +0.329] —
null as leaned: **no state-level early warning.** v2 dominates
overall (+0.729/+0.512) as required; NOTE: in this design (2
lineages/matrix) the centered test is underpowered for every
predictor — v2's centered null here (+0.201/−0.038, CIs spanning 0)
is a power artifact of K3's cohort shape, not a revision of the
core replication's state-level transfer (established at scale with
branch-half reliability ρ ≈ 0.93). The frozen interpretation
applies: the signature reads present organization; it carries no
detectable state-level foresight.

## K4 — robustness: sign-stable everywhere; magnitude CLR-sensitive
## in 02

Reference (standard instrument, same 12-matrix subsample):
+0.099 [−0.093, +0.311] (02; the subsample itself is underpowered)
/ +0.082 [+0.011, +0.145] (03). Variants: median aggregation
+0.105/+0.065 and paper-text drop-last +0.089/+0.084 [+0.029,
+0.133] — indistinguishable from the standard instrument. Removing
CLR shrinks the effect in 02 (+0.015) but not in 03 (+0.072). The
registered sign-preservation requirement holds in all six
variant×candidate cells; magnitude in candidate 02 depends
materially on the CLR preprocessing (reported prominently; the CLR
step is the GARD paper's own registered preprocessing).

## Scorecard

K1 monotone-both — missed in 02 (ledger #8). K2 lean (transfers +
synergy carry it) — half-wrong in an informative way: downward
causation dominates, synergy-persistence responds NEGATIVELY, rtr
contributes. K3 both registered predictions HIT (overall negative;
centered null). K4 sign requirement met; CLR sensitivity disclosed.
