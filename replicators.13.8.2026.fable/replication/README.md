# Replication: plastic-heredity break-and-renewal in reconstructed GARD

Scope: replicates ONLY the positive discovery of the pre-print — the
untouched-confirmed, past-observable process-risk coordinate for
plastic-heredity break-and-renewal (the L44/L53/L54-analog chain). It does
not attempt the paper-facing PhiID reconstruction, Figure 5 prediction,
or intervention analyses, which the pre-print itself reports as
unsupported.

The primary `JOINT_BREAK_RUN3` result establishes a break followed by
renewed adjacent parent–daughter inheritance; it does not establish one
mutually coherent new regime. Later adversarial phases made that
distinction explicit, and Phase H independently confirmed only the
occurrence of the stricter coherent eight-fission endpoint.

## Pipeline

1. `sim.py` — GARD engine with kinetics pinned to the historical GARD10
   source (`ModelingOriginsofLife/GARD` @ `86dff632`): `Kf=1e-2`,
   `Kb=1e-4`, uniform `rho=1/100`, `bn = 1 + (beta n)/N`, `NG=100`,
   `nmin=40`, `nmax=80`, `beta_ij = exp(-4 + 4 Z)`. Two candidate
   contracts: `02` = historical categorical events, exact-size fission,
   hypergeometric equal split, first-daughter continuation; `03` =
   paper-described vector-Poisson exposure, overshoot allowed,
   binomial(0.5) fission, uniformly selected daughter.
2. `features.py` — nine direct history/phase variables, a 195-coordinate
   molecule-label-permutation-invariant graph/state block, beta-only
   matrix features; the prospectively fixed target `JOINT_BREAK_RUN3`
   (within 12 fissions: an inheritance break `H<=0.9`, then a run of 3
   consecutive inherited fissions starting strictly after the break);
   the seven process outcomes of the plasticity decomposition.
3. `run_dev.py` — development stage (dev seed domain): 40 matrices x 2
   candidates x 100 fissions; per-fission training rows with realized
   F12 outcomes; trains and freezes four students per candidate
   (prior, direct-9, beta-only, full = PCA-12 of the 195 block + 9
   direct, ridge logistic `C=0.1`).
4. `run_conf.py` — untouched confirmation (new 256-bit seed domain):
   40 new matrices, 80 trajectories, five restored post-fission states
   per trajectory at fissions 20/35/50/65/80 (400 states), 64
   independent F12 branches per state split into halves of 32 before
   outcomes (25,600 branches), regenerated exactly in a second campaign
   (replay gate). Frozen models applied without refitting.
5. `analyze.py` — reliability, overall and matrix-centered rank
   transfer, branch log loss, q-Brier, 4,096 whole-matrix bootstraps,
   512 whole-matrix permutations, process probabilities, figures.

## Registered reconstruction choices (where the pre-print is not fully
prescriptive)

- The exact inventory of the 195 graph/state coordinates is not
  enumerated in the pre-print; this replication registers its own
  195-coordinate permutation-invariant inventory (`features.py`).
- Candidate `02`/`03` semantics are reconstructed from the pre-print's
  registry menu (exposure/daughter/trim branches) and the pinned
  historical source; the pre-print does not disclose its two contracts.
- Process-outcome operationalizations (`resume2`, `old_return` with a
  0.7 departure gate, `repeat`) are registered in `features.py`.
- Development cohort size (40 matrices x 2 candidates) is a compute
  choice; the confirmation campaign matches the paper scale exactly
  (40 matrices, 80 trajectories, 400 states, 25,600 branches, halves
  of 32, second exact campaign).

Run order: `python3 run_dev.py && python3 run_conf.py && python3 analyze.py`

## Beyond the core replication

The full adversarial program that followed (reviewer-driven Phases
A–G) is documented in three places: **`REPORT.md`** — every campaign,
gate, and result, including the final ontology and the complete
six-entry prediction-miss ledger; **`INTERVENTIONS.md`** — registered
designs and result tables for every intervention/controller
experiment; **`PHASE_G.md`** — the durable preregistration of the
final bounded program (G1–G5), with per-module RESULTS appended below
the frozen registration; **`STRICT8_REPLICATION_PREREGISTRATION.md`**
— Phase H, the sealed external occurrence replication of the strict
coherent eight-fission episode (domain 25; endpoint fixtures in
`test_strict8_endpoint.py`; seal hashes in
`results_strict8_occurrence/SEAL.json`); and **chapter 5, the Φ-r
program (Phases I–N)** — six preregistrations with results appended
below their frozen registrations: `PHIR_BRIDGE.md` (I: reciprocal
bridge + code-faithful addendum + typeset erratum),
`PHIR_CONFIRM.md` (J: prospective 2× confirmation,
signature-vs-controller), `PHIR_DOSE.md` (K: dose–response, atom
decomposition, natural prediction, robustness), `PHIR_PAPER.md`
(L: the paper-faithful instrument), `PHIR_SR.md` (M: SR-state
reading), `PHIR_FORESIGHT.md` (N: the foresight round). Instruments:
`phir.py` (extracted-text reading), `phir_code.py` (verified port of
the authors' public PhiRL implementation; no external dependencies),
`phir_paper.py` (typeset-verbatim reading). Hand-off index for
external replication: `CODEX_HANDOFF_CH5.md`. Validation:
`python3 test_validation.py` (34 checks) and
`python3 test_strict8_endpoint.py` (14 fixtures). All campaigns are
deterministic from documented seed tags (domains 0–25 and 27–30; 26
is used by the separate `strict-8-paper-preparations/` side study).
