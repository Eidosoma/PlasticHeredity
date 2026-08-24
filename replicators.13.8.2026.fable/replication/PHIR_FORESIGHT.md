# Phase N — the foresight round: is any Φ variant a predictor?
# (preregistered 2026-08-18, BEFORE any campaign lineage ran)

Closes the predictor-versus-gauge question with the three known
weaknesses of the K3 test repaired and one new test added:

- POWER: K3 had 2 lineages/matrix — the matrix-centered comparison
  was underpowered for every predictor including v2. N1 uses 12
  lineages/matrix.
- CANCELLATION: the Φ_R scalar sums atoms that move in opposite
  directions (K2); N1 tests the unbundled components (downward
  causation; emergence) as predictors in their own right.
- CLOCK: all prior instruments run on within-growth micro-steps;
  N1/N3 add a GENERATIONAL-clock Φ (the same macro ΦID pipeline on
  the daughter-composition series across fissions).
- EVENT-LOCKED EARLY WARNING (new, sharpest): does any instrument
  shift in the generations JUST BEFORE a break, compared against
  matched currently-inherited control windows — foresight beyond
  current state, the critical-slowing-down test done properly.

## N1 — powered natural prediction (domain 30,
## tag "phir-foresight-2026-08-18")

Fresh cohort: 16 new matrices × 12 lineages (reps) × 2 candidates =
384 natural 60-fission lineages, fully traced (per-update record
with per-fission markers; daughters retained). Keys: (30,0,m) beta;
(30,1,m) initial state; (30,2,cand_i,m,rep) growth.

Predictors, all computed from fissions ≤ 40 only:
- phiR_scalar: code-Φ_R on the concatenated 21–40 update series.
- causation, emergence, printed: same window, atom components (the
  M/K machinery).
- phi_volatility: std over sliding 3-generation-window Φ_R values
  with centers in 22–39 (critical-slowing-down candidate).
- phi_trend: OLS slope of the same window series.
- gen_phiR / gen_printed: the macro ΦID pipeline applied to the
  daughter-composition series of fissions 1–40 (T = 40 generations).
- v2_risk (frozen model at the fission-40 state) and hist (inherited
  fraction 21–40): benchmarks.

Outcome: break count in fissions 41–60. Statistics: Spearman
(predictor, outcome), OVERALL and MATRIX-CENTERED, whole-matrix
bootstrap (4,096 draws, seed 31). Secondary: centered Spearman of
each Φ variant against the outcome residualized on v2 within matrix
("adds anything beyond v2?").

**Validity gate (registered): v2_risk matrix-centered must be
positive with CI excluding 0 at this power; if it is not, the design
is declared underpowered and NO Φ null is interpreted.**

## N2 — event-locked early warning (same cohort)

Per-window instruments (3-generation sliding windows, the Phase M
machinery) over all centers g = 2…59. CASE windows: centered at
t−2 for each break at fission t that was preceded by ≥ 5 consecutive
inherited fissions (the window spans t−3…t−1: currently-inherited
generations immediately before a break). CONTROL windows: centers c
inside inherited runs with run-position ≥ 5 and no break within
c+1…c+3 (deep-run, no imminent break). Per matrix: mean(case) −
mean(control) per instrument {phiR, printed, synergy, emergence},
plus local volatility (std of the three preceding windows' Φ_R).
Matrix bootstrap; a CI excluding 0 in both candidates = genuine
pre-break signal beyond current inheritance state.

## N3 — is the generational-clock Φ a gauge too? (replay)

Byte-exact replay of Phase J ph_stab/ph_destab/noop (daughters only;
no tracing needed), gen-clock instruments on daughters 1–60
(T = 60). Between-arm contrast (matrix bootstrap): does gen_phiR
respond to the heredity dial like micro-clock Φ_R does? Replay gate:
recomputed heredity outcomes must equal stored Phase J values.

## Registered predictions

- N1 validity: v2 centered POSITIVE (must). hist centered positive.
- N1 leans: phiR_scalar centered null (as in K3); causation and
  emergence weakly positive (uncancelled organization signal);
  volatility positive (CSD theory); trend two-sided; gen-clock
  two-sided, no lean. All two-sided.
- N2 leans: level instruments null (no foresight beyond state);
  volatility positive (fluctuations rise before transitions). The
  synergy component two-sided (M1 showed it NEGATIVE during SR —
  a pre-break RISE would be coherent with reorganization onset).
- N3 lean: responsive (organization reading robust to clock).

## Boundary

In-domain break prediction only — not replicator-onset adjudication.
All instruments are our verified ports/pipelines; per-window and
gen-clock quantities use the tractable macro pipeline throughout.
Sealed: this file, `run_phir_foresight.py`, `run_phir_sr.py`,
`phir_code.py`, `run_phir_confirm.py`, `run_phir_dose.py`, `sim.py`,
`cohort.py` (SHA-256 in `results_phir_foresight/SEAL.json`).

---

# RESULTS (appended 2026-08-18; nothing above edited)

N1/N2 cohort 74 s (384/384 complete lineages; 303/316 pre-break
events per candidate); N3 replay 101 s, **REPLAY GATE PASS (0
mismatches)**. Raw units `results_phir_foresight/
foresight_units.pkl`; suite 34/34 at seal.

## N1 — powered natural prediction: VALIDITY GATE PASS, and every
## Φ variant is null where v2 is strong

The registered validity gate passes decisively: v2 matrix-centered
Spearman **+0.428 [+0.271, +0.558] / +0.289 [+0.165, +0.414]** —
the design has real within-world power (even the bare history
baseline shows a centered signal in candidate 03). Against that:
EVERY Φ variant's centered correlation is null in BOTH candidates —
phiR_scalar +0.020/−0.058; causation −0.094/−0.005; emergence
−0.059/−0.008; printed −0.046/−0.082; volatility −0.033/+0.026;
trend; gen_phiR −0.084/+0.002; gen_printed — all CIs spanning 0.
The matrix-level channel replicates (overall correlations negative
and significant for phiR/emergence/gen_phiR: high-Φ worlds break
less). Residual-on-v2: all null — no Φ variant adds anything beyond
the frozen state coordinate. Registered leans: phiR-null HIT;
causation/emergence weak-positive and volatility-positive leans
MISSED (ledger miss #10).

## N2 — event-locked early warning: null across the board

With 303/316 pre-break events against matched deep-run controls, no
instrument shifts in the currently-inherited generations
immediately before a break: phiR −0.088 [−0.209, +0.008] /
−0.051 [−0.166, +0.032] (directionally lower pre-break — the gauge
sagging with organization — but not significant); printed, synergy,
emergence null; volatility +0.013 [−0.007, +0.032] /
+0.002 [−0.017, +0.024] — the critical-slowing-down lean also
MISSED (folded into miss #10). No foresight beyond current state.

## N3 — the generational clock: the gauge property is
## clock-robust, and the PRINTED structure comes alive as a gauge

gen_phiR responds to the heredity dial in both candidates
(+0.0587 [+0.0364, +0.0806] / +0.0532 [+0.0294, +0.0769]) —
registered lean HIT. **Bonus finding: gen_printed ALSO responds in
both candidates (+0.0263 [+0.0044, +0.0494] /
+0.0389 [+0.0175, +0.0603])** — on the generational substrate, even
the typeset formula's structure becomes a working (if weaker)
gauge. The clock was a load-bearing variable all along: the printed
formula's chapter-long deadness was partly a substrate problem, not
only a formula problem.

## Adjudication — the question closed

Gauge: YES, robustly — across formulas (Φ_R, emergence, and on the
generational clock even the printed structure), across clocks, in
both comparison styles (Phases J, M, N3). Predictor: NO — at
validity-gated power, with the scalar unbundled, the clock
re-pointed, and the event-locked test run, every Φ variant carries
zero state-level foresight and adds nothing beyond the frozen state
coordinate; its entire natural predictive content is matrix
identity. The one door left open in principle — that some
information-theoretic functional sees trouble coming — is now
closed for every variant this program could construct.
