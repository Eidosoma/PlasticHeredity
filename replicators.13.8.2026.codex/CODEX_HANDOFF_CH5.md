# Chapter 5 (Φ-r program, Phases I–N) — external replication hand-off

For the independent replicating agent. Everything below is
self-contained in this directory; nothing depends on our
conversation history. Run `python3 test_validation.py` first
(34 checks must pass) from this directory.

## Instruments (the four readings of one Methods page)

| file | quantity | provenance |
|---|---|---|
| `phir.py` | normalized whole-minus-parts (parts→whole future, /I_tot) | the TEXT-EXTRACTED reading — the denominator is a PDF-extraction artifact (see PHIR_BRIDGE.md erratum) |
| `phir_paper.py` | unnormalized difference, MULTIVARIATE MIB blocks, CLR+drop-last | the TYPESET page verbatim |
| `phir_code.py` | revised Φ_R: 9-atom ΦID sum on macro-averaged Fiedler halves | port of the authors' public PhiRL repo (github.com/pigozzif/PhiRL), verified vs upstream to ~1e-14 (record in PHIR_BRIDGE.md); NOTE: PhiRL is the companion RL paper's code, not the GARD paper's (still private) |
| atom decomposition (`run_phir_dose.phi_atoms`, `run_phir_sr.window_scores`) | all 16 atoms + "synergy"/"causation"/"emergence" summaries | same pipeline, unbundled |

No external dependencies beyond numpy/scipy/scikit-learn. Verify
your own port against the PhiRL repo independently; do not trust
ours.

## Modules, seals, and the numbers to shoot at

Every module: preregistration ABOVE the `--- RESULTS` line (frozen),
results below, SHA-256 source seals in each results dir (Phase I
predates the sealing practice; its instruments are hashed in the
J–N seals, and its addendum records the port-equality verification).
Matrix = bootstrap unit throughout; candidates 02/03 never pooled.

- **I — `PHIR_BRIDGE.md`** (domain 27): six-arm bridge. Text-Φr
  flat/negative, unresponsive (T1 null both candidates); v2
  validity swing +0.185/+0.174. ADDENDUM (byte-exact replay,
  gate 0/576): code-Φ_R responds **+0.208 [+0.134,+0.279] /
  +0.256 [+0.121,+0.392]**; text↔code correlation ≈ 0.1. Erratum:
  typeset formula (unnormalized) also null via atom identity
  (−0.006/−0.015, CIs span 0).
- **J — `PHIR_CONFIRM.md`** (domain 28, 48 fresh matrices, 2×):
  prospective confirmation **C1 +0.1548 [+0.0749,+0.2333] /
  +0.1781 [+0.1030,+0.2591]**; probe-rollout Φ_R controller moves
  the gauge only in 03 (C2a +0.065), heredity follows nowhere (C2b
  null) → responsive signature, not a controller. Known deviation:
  C4 random−noop = −0.017 in 03 heredity (real small cost of random
  editing at this power).
- **K — `PHIR_DOSE.md`** (domain 29): dose–response graded in 03
  only (+0.274 [+0.107,+0.427]; 02 unresolved). Atom decomposition
  (replay gate exact): **"causation" carries the response
  (+0.577/+0.504); synergy-persistence NEGATIVE (−0.384/−0.298);
  "emergence" responds both candidates (+0.193/+0.207)**. Natural
  prediction: matrix-level only. Robustness: sign stable; no-CLR
  shrinks magnitude in 02.
- **L — `PHIR_PAPER.md`**: typeset-verbatim multivariate instrument
  (replay gate 0/768): levels −8.9..−11.0; no coherent coupling
  (sign flips across windows/candidates; L2 specificity violated in
  03) — noise-dominated 198-dim estimator.
- **M — `PHIR_SR.md`**: SR-state reading (SR = inside inherited run
  ≥ L, L=5 primary; occupancy 0.89/0.90 stab vs 0.68/0.63 destab).
  Within-lineage **Φ_R reads SR: ΔSR +0.099 [+0.055,+0.149] /
  +0.138 [+0.094,+0.188]**; printed-structure weaker (null in 02);
  synergy negative during SR. Consistency gate reproduces J/K on
  the same windows.
- **N — `PHIR_FORESIGHT.md`** (domain 30): validity gate v2
  centered **+0.428/+0.289** (must reproduce or your design is
  underpowered — do not interpret Φ nulls without it); ALL Φ
  variants centered-null and residual-null; event-locked pre-break
  test null (303/316 events); volatility null. Gen-clock gauges
  respond: gen_phiR +0.059/+0.053 and **gen_printed +0.026/+0.039
  (the typeset structure works as a gauge on the generational
  substrate)**.

## Replay-gate chain (your strongest tool)

J's campaign is the anchor: I-addendum, K2, L, M, and N3 are all
byte-exact replays of it, each gated on equality with
`results_phir_confirm/phir_confirm_units.pkl`. If you reproduce J
from its seed recipe (domain 28, tag "phir-confirm-2026-08-17",
keys in PHIR_CONFIRM.md), every downstream module is a deterministic
re-measurement you can verify the same way.

## Registered misses relevant to this chapter (honesty ledger)

#7 (I-addendum lean), #8 (K1 dose in 02; K2 atom lean), #9 (M
variance-components resolution), #10 (N unbundling + volatility
leans). Full ledger in REPORT.md.

## Boundary (carry it verbatim)

Nothing here adjudicates the GARD paper's private pipeline or its
replicator-onset prediction claim; PhiRL is the companion RL
paper's public code; the SR label (M) is our reconstruction of an
unstated metric; registered choices forced by the page's silence
(pseudocount, ridge, MIB search, windows, aggregation) are listed
per module.
