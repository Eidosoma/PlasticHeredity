# G3-ADJ — launch-state × convention factorial
# (preregistered 2026-08-18, BEFORE any campaign lineage ran)

Adjudicates the cross-laboratory G3 hysteresis disagreement. The
source audit measured and ruled out the estimand difference (the
external per-matrix estimand STRENGTHENS our sealed result:
+0.571/+0.443) and the selector difference (70% action agreement,
regret ~0.002), leaving two credible explanations: (1) LAUNCH STATE —
our sealed G3 launches from fresh `matrix_and_init` states (40
occupied types, entropy 3.69, throughput ~22, risk 0.58–0.66) while
the external design launches from naturally evolved generation-60
states (~17 types, entropy ~2.2, throughput ~2,900, risk 0.35–0.39);
(2) EDIT/ANCHOR CONVENTION — ours edits after fissions 1…P−1 and
anchors the unedited post-fission daughter; the external convention
(implemented here from its DESCRIPTION only; no external code
consulted) edits after every pulse fission including P and anchors
the post-edit state.

## Design (domain 32, tag "g3-adj-2026-08-18")

2×2 factorial, everything else the sealed G3 machinery: model-down
`marginal_swap` edits, pulses P ∈ {1, 2, 4, 8, 16, 32, 60}, release
F60, anchor-similarity half-life t07 (first crossing < 0.7, censored
at 61), 24 matrices × 2 candidates × 2 reps.

- Launch FRESH: `matrix_and_init` under the sealed steering tag —
  and the fresh×conv-A cell is run with the SEALED G3 seed keys
  ((22, 0, cand_i, m, rep), fresh generator per pulse call,
  replicating the sealed restart-per-pulse behavior), so it doubles
  as a REPLAY GATE: its t07 values must equal the sealed
  `results_g/g3_units.pkl` pulse rows exactly.
- Launch NATURAL: the generation-60 daughter of the sealed domain-7
  steering-tag lineage (the same states Phase G4 used), one state
  per (matrix, candidate), shared by both reps. REGISTERED SCOPE
  NOTE: the history clock still restarts at launch in ALL cells
  (hs empty) — this factorial isolates the STATE; the clock
  component of the audit's differences #1/#6 is deliberately held at
  our convention everywhere and remains a residual suspect only if
  natural-launch cells stay strong.
- Convention A: edits after fissions 1…P−1; anchor = unedited
  post-fission daughter after fission P (sealed G3).
- Convention B: edits after fissions 1…P (including after P); anchor
  = the post-edit state, which also launches the release.
- Non-gate cells use fresh streams (32, cell_id, cand_i, m, rep),
  fresh generator per pulse call (mirroring the sealed quirk).

## Registered analysis

Primary estimand (the cleaner one, per the audit): average reps
within matrix × pulse, one seven-point Spearman per matrix, mean
over matrices, 4,096 whole-matrix bootstrap draws (seed 41). The
sealed-pooled estimand reported descriptively. Per-cell per-pulse
half-life tables reported.

## Registered predictions and frozen reading

- Replay gate must PASS (else the module is invalid).
- Fresh×A reproduces strong hysteresis (≈ +0.57/+0.44 under this
  estimand — the sealed recomputation).
- LEAN: launch state is the moderator — both fresh cells strong,
  both natural cells attenuated (CIs including 0 or markedly
  smaller), convention secondary. Frozen reading by outcome:
  - Launch dominates → hysteresis is a property of NASCENT lineages
    (a registered SCOPE REVISION of G3's finding; ledger entry —
    the sealed G3 interpretation implicitly claimed generality).
  - Convention dominates (fresh×B attenuated, natural×A strong) →
    anchor-semantics artifact; G3's finding survives as stated for
    its convention, and the disagreement is definitional.
  - Both matter → report the interaction; no rescue, no pooling.
  - All four cells strong → neither explains the external result;
    the disagreement moves to configuration/scale questions.

## Boundary

Our lab only; the external convention is implemented from its
written description; no external code or data is consulted or used
as a gate. Sealed files: this registration,
`run_g3_adjudication.py`, `run_g3_halflife.py`, `run_steering.py`,
`sim.py`, `cohort.py` (SHA-256 in
`results_g3_adjudication/SEAL.json`).

---

# RESULTS (appended 2026-08-18; nothing above edited)

Campaign 1,368 s; **REPLAY GATE PASS: 0 of 672 fresh×conv-A rows
differ from sealed `results_g/g3_units.pkl`** — the gate cell IS the
sealed experiment, reproduced end-to-end. Raw units
`results_g3_adjudication/g3_adj_units.pkl`; suite 35/35 at seal.

## The 2×2 (per-matrix-mean Spearman, 4,096 whole-matrix draws)

| cell | cand 02 | cand 03 |
|---|---|---|
| fresh × conv A (sealed G3) | **+0.571 [+0.456, +0.677]** | **+0.443 [+0.246, +0.616]** |
| fresh × conv B | **+0.542 [+0.454, +0.626]** | **+0.515 [+0.354, +0.651]** |
| natural × conv A | +0.136 [−0.055, +0.318] | +0.190 [+0.011, +0.380] |
| natural × conv B | +0.315 [+0.119, +0.519] | +0.192 [+0.017, +0.351] |

Half-life tables confirm the mechanism: fresh launches start at
P1 ≈ 4–6 fissions and climb to ~13–16; natural launches START at
P1 ≈ 12–15 — already at the long-pulse plateau before any steering.

## Adjudication under the frozen reading

**Launch state dominates; convention is secondary.** Both fresh
cells are strong under both conventions (+0.44 to +0.57); all four
natural cells are markedly attenuated (+0.14 to +0.32, three of four
weakly excluding zero, one including it). The registered lean is a
HIT. Per the frozen reading, this is a **registered SCOPE REVISION
of G3** (ledger entry #11): accumulating hysteresis is primarily a
property of NASCENT, unformed lineages — depth-of-writing matters
most while the self is still being formed; an already-evolved
lineage sits near its imprint ceiling, where additional holding buys
little. The convention factor produces no consistent effect (B ≈ A
on fresh states; B mildly higher than A in one natural cell).

## Cross-laboratory resolution (descriptive)

The external result (+0.141 [−0.016, +0.291] / +0.108
[−0.039, +0.249], evolved-launch design) falls almost exactly on our
natural-launch cells (+0.136/+0.190 under the matching convention A).
**Both laboratories were right about their own launch conditions;
the disagreement was a moderator, not an error.** The weak residual
hysteresis in evolved lineages (three natural cells excluding zero)
suggests the external design was underpowered for a small real
effect rather than measuring a null. No sealed result on either side
requires correction; G3's interpretation is scope-qualified as
above.
