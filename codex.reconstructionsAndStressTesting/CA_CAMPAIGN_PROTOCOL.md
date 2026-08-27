# Preregistered E23/E24 clean-room CA campaign

Date frozen: 2026-08-20, after wiring-only smoke validation and before the
pilot or full reference outcomes were opened.

## Purpose and evidence boundary

This campaign closes E23 with a genuinely independent ECA truth atlas, then
tests the retained E24 Life-family result and its known activity-clock weakness.
Sibling prose and retained data tables are permitted. Sibling source, tests,
scripts, executable seeds, and serialized implementation artifacts are not.
The E20 GARD–ECA–XENO bridge and Rosetta placement are excluded.

The retained E24 CSV supplies only the frozen 1,024-rule sample and comparison
targets. Simulated outcomes never read retained endpoint columns. Because the
four hash-density Life boards and trajectory RNG were not disclosed, primary
Life results are independent gate-level evidence unless a later code-free
trace makes exact replay possible.

## Frozen simulator

- B/S rules use nine survival bits plus B1–B8 birth bits on a synchronous
  periodic Moore-8 grid.
- Primary grid 16×16; process flip 0.002; copy error 0.005; minimum four
  sweeps; 32 completed generations.
- Activity is realized post-noise state change. Timeout or an empty terminal
  board is death. Composition is the unit-normalized sum of the fifteen live
  2×2 censuses over one generation, observed before copy.
- Break-by-8 and strict coherent-eight use the golden-reconciled E19 boundary
  and anchor conventions.
- Each of eight launches contributes at most one form: the 0.75-mass support
  of the mean last completed composition across all futures that broke.
- Primary launches are glider, blinker, toad, block, and exact-density
  SHA-256-ranked boards at 0.10, 0.20, 0.30, and 0.40, shared across rules.
- One shrinking PCG64 stream is derived per condition/rule/launch. Work is
  checkpointed per rule under the full contract digest.

## Frozen campaigns and gates

All 1,024 rules run with eight launches × 64 futures at activity budgets 48,
256, and 1,024, holding the 64-sweep maximum fixed. A deterministic,
outcome-blind 128-rule subset then runs:

- budget 1,024 at 128- and 256-sweep maxima;
- matched 16×16 and 32×32 grids at budget 4N and max 256, 32 futures;
- a second 0.10–0.40 launch draw and a broad 0.20/0.40/0.60/0.80 ensemble at
  budgets 48 and 1,024, 32 futures.

Registered E24 gates retain their original definitions: smoothness ≥ 2,
heavy-tail share ≥ 0.35, Life top-decile plus strict in [0.005, 0.5], and a
capable-rule median clock strictly inside the family IQR. New gates are:

1. `family_laws_robust`: smoothness and heavy tail pass at all three full
   budgets.
2. `clock_rescued`: at budget 1,024 there are at least 20 capable rules, clock
   IQR width is at least one sweep, and capable median lies strictly inside it.
3. `life_remains_maintainer`: Life stays top-decile by library size but below
   0.005 strict at every full budget.
4. `launch_robustness`: alternate-launch strict and break ranks correlate at
   least 0.70 with primary and preserve both atlas-law verdicts.
5. `scale_stability`: matched 16×16/32×32 strict ranks correlate at least 0.70.

For E23, a new 16-seed × 128-future truth atlas uses namespace
`rulial-evo-truth-v1`. The top eight dev-census forms define 64 deterministic
start/target trials. Planning is shortest-path search to dev holders; the
random control walks for at most 16 edits; the oracle routes to truth holders.
All endpoints are scored only on truth. GPS passes when planned success beats
random and reaches at least 60% of oracle success.

Every failure or reversal is reported. No stress condition may replace the
frozen budget-48 result, and no early outcome may cancel a scheduled condition.
