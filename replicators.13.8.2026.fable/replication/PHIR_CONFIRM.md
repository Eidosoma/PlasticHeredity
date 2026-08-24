# Phase J — Φ_R signature-versus-controller test (2× scale,
# prospective)
# (preregistered 2026-08-17, BEFORE any campaign lineage ran)

Chapter-5 companion to Phase I (`PHIR_BRIDGE.md`). The Phase I
ADDENDUM discovered — by re-measurement of a frozen campaign — that
the authors' IMPLEMENTED Φ_R (code-faithful port `phir_code.py`,
verified to ~1e-14 against github.com/pigozzif/PhiRL) responds
causally to the heredity dial, while the paper's printed formula
does not. Two limitations remained: (i) the discovery was a
re-measurement, not a prospective confirmation on fresh matrices;
(ii) the reverse direction (Φ_R as a CONTROLLER of heredity) was
untested because no working Φ_R-steering rule existed. Phase J
closes both at 2× the Phase I scale, on completely fresh matrices,
with a probe-rollout Φ_R controller. This experiment is designed for
independent external replication (sealed hashes, fresh seed domain,
self-contained registration).

## Design (domain 28, tag "phir-confirm-2026-08-17")

Fresh cohort: **48 new matrices** × 2 candidates × 2 reps (2× Phase
I's 24). Spawn keys: (28,0,m) matrix; (28,1,m) initial state;
(28,2,cand_i,m,rep) growth/fission stream (CRN, arm-independent);
(28,3,cand_i,m,rep,f) action panel; (28,4,…,f) random-arm pick;
(28,5,…,f) probe stream (identical for every candidate edit at a
decision — CRN across edits). 60 steering fissions; one edit after
every fission; measurement window = fissions 41–60, concatenated
per-update composition series (the addendum's registered substrate).

**Identical legal action set:** a CRN panel of 12 mass-preserving
swaps per decision, shared by all six arms:

| arm | selection rule on the shared panel |
|---|---|
| ph_stab / ph_destab | frozen v2: minimize / maximize predicted risk |
| phiR_max / phiR_min | maximize / minimize PROBED code-Φ_R: apply the edit, run a 2-fission probe (CRN probe stream), compute code-Φ_R on the probe's update series; NaN probes never selected |
| random | uniform pick |
| noop | no edit |

The probe estimates the ACTUAL implemented quantity (no surrogate);
it is noisy by construction (probe series ≈ 100 update steps for
cand 02, ≈ 24 for cand 03) — the manipulation check below measures
whether it works at all. Probes never touch the real growth stream.

Outcomes: heredity (inherited fraction, breaks, longest inherited
run) and realized code-Φ_R (one value per lineage, `phir_code.py`
unchanged); text-formula Φ-r retained descriptively.

## Registered tests (matrix bootstrap, 4,096 draws, seed 17;
## candidates never pooled; both candidates required for any claim)

- **C1 (prospective confirmation, primary):** ph_stab − ph_destab on
  realized code-Φ_R > 0 with CI excluding 0, both candidates.
- **C2a (manipulation check):** phiR_max − phiR_min on realized
  code-Φ_R; CI excluding 0 both candidates = the gauge is steerable.
- **C2b (reverse causal test, primary):** phiR_max − phiR_min on
  inherited fraction (breaks, longest run co-primary); CI excluding
  0 both candidates = Φ_R supplies a heredity control direction.
- **C3 (validity):** ph_stab − ph_destab must move heredity;
  campaign invalid otherwise.
- **C4 (specificity):** random − noop CIs must span 0 on both
  outcome families.

## Frozen adjudication table

- C1 pass, C2a pass, C2b null → **"responsive signature, not a
  controller" confirmed prospectively** (the addendum's reading).
- C1 pass, C2b pass → Φ_R and heredity are reciprocally coupled
  control coordinates (upgrade beyond the addendum).
- C1 pass, C2a fail → the gauge is not steerable at probe
  resolution; C2b uninformative; the signature claim stands.
- C1 fail → the addendum finding does not confirm prospectively;
  report as fragility (scale/matrix dependence), no rescue.

## Registered predictions

- C1: PASS (strong lean — the addendum effect was +0.21/+0.26 with
  CIs far from zero; fresh matrices and 2× scale should reproduce).
- C2a: uncertain, lean weakly positive (probe noise may swamp the
  per-edit signal, especially cand 03's ~24-step probes).
- C2b: lean NULL — the "downstream signature" hypothesis predicts
  that pushing the gauge does not move the mechanism. This is the
  theory-deciding cell.
- C3, C4: must pass (validity).

## Boundary

All Phase I / addendum boundaries carry over verbatim: the
instrument is our verified port of the authors' public code (two
registered choices: CLR upstream, mean aggregation); nothing here
tests replicator-onset prediction or adjudicates the authors'
private GARD pipeline. Sealed before the campaign: this file,
`run_phir_confirm.py`, `phir_code.py`, `phir.py`, `sim.py`,
`cohort.py` (SHA-256 in `results_phir_confirm/SEAL.json`).

---

# RESULTS (appended 2026-08-17; nothing above edited)

Campaign 898 s on 12 workers; 1,152 lineages; raw units
`results_phir_confirm/phir_confirm_units.pkl`; suite 30/30 at seal.

## Arm table (candidate 02 / 03)

| arm | inherit | breaks | longest run | code-Φ_R |
|---|---|---|---|---|
| ph_stab | 0.935 / 0.939 | 3.9 / 3.7 | 41.2 / 41.5 | 1.856 / 1.177 |
| ph_destab | 0.809 / 0.790 | 11.5 / 12.6 | 24.1 / 22.3 | 1.702 / 0.999 |
| phiR_max | 0.885 / 0.883 | 6.9 / 7.0 | 31.2 / 31.7 | 1.791 / 1.099 |
| phiR_min | 0.872 / 0.877 | 7.7 / 7.4 | 30.9 / 31.0 | 1.762 / 1.034 |
| random | 0.878 / 0.861 | 7.3 / 8.4 | 31.8 / 29.2 | 1.783 / 1.085 |
| noop | 0.886 / 0.878 | 6.8 / 7.3 | 32.7 / 30.7 | 1.829 / 1.062 |

## Registered tests

- **C1 (prospective confirmation): PASS, BOTH candidates.**
  ph_stab − ph_destab on code-Φ_R: **+0.1548 [+0.0749, +0.2333] /
  +0.1781 [+0.1030, +0.2591].** The addendum discovery confirms on
  48 completely fresh matrices at 2× scale, under full prospective
  registration. (Effect modestly smaller than the addendum's
  +0.21/+0.26 — ordinary shrinkage.)
- **C2a (gauge steerable?): PARTIAL — 03 only.** phiR_max − phiR_min
  on code-Φ_R: +0.0297 [−0.0579, +0.1104] (02, ns) /
  **+0.0650 [+0.0057, +0.1227] (03, passes)**. The both-candidates
  gate fails; the probe controller moves the gauge weakly and only
  reliably in candidate 03. Notably, even where it works it moves
  Φ_R only ~1/3 as far as the heredity dial does (C1): the best
  Φ_R controller found so far is the heredity controller.
- **C2b (does pushing the gauge move heredity?): NULL, both
  candidates, all three co-primaries** (inherit +0.0125
  [−0.0017, +0.0267] / +0.0057 [−0.0080, +0.0188]; breaks and
  longest-run CIs all span 0). Informative in candidate 03, where
  the gauge demonstrably moved (C2a) and heredity did not follow;
  uninformative in 02 per the frozen table (gauge not demonstrably
  moved there).
- **C3 (validity): PASS both** (+0.1255 [+0.0995, +0.1552] /
  +0.1493 [+0.1191, +0.1818]).
- **C4 (specificity): VIOLATED IN ONE CELL, reported as registered.**
  random − noop spans 0 everywhere except candidate 03 heredity:
  −0.0170 [−0.0325, −0.0030] — at 2× power, a small real cost of
  random per-fission editing becomes detectable (prior phases at
  half this power found random ≈ noop). Consequence assessment: C1
  and C2a/C2b are contrasts between arms with EQUAL edit budgets and
  are internally controlled; the violation does not touch them, but
  it is flagged prominently and the "random ≈ noop" background claim
  is now scale-qualified program-wide.

## Adjudication under the frozen table

Candidate 03 realizes the full registered pattern "C1 pass, C2a
pass, C2b null → **responsive signature, not a controller, confirmed
prospectively**." Candidate 02 realizes "C1 pass, C2a fail → gauge
not steerable at probe resolution; C2b uninformative; the signature
claim stands." No cell anywhere supports the rival "coupled control
coordinates" reading. Combined chapter-5 conclusion: **the authors'
implemented causal emergence is a prospectively confirmed,
causally downstream signature of hereditary organization; it is not
a control handle — and the most effective way to raise it is not to
aim at it but to stabilize heredity.**

## Prediction scorecard

C1 pass — predicted, confirmed. C2a lean weakly positive —
half-right (03 only). C2b lean NULL — **predicted and confirmed**
(the signature hypothesis survived its designed refutation test).
C4 — violated in one cell; documented above, no rescue applied.
