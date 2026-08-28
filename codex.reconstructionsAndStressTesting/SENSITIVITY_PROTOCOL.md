# Frozen protocol — semantic reconciliation round 1

Date frozen: 2026-08-19  
Design name: `overnight`  
Output: `results/sensitivity-round-1/`

## Purpose and status

The first clean-room run reproduced the broad ECA/Life observer story but not
the numerical ECA atlas.  This is a development-only reconciliation campaign,
not a new confirmation.  It tests whether choices left unresolved by the prose
record explain that mismatch before requesting executable-contract details from
Fable.

No sibling source, tests, scripts, executable configuration, or seed material
may be opened.  The permitted comparison target remains the retained JSON/CSV
result record.

## Frozen semantic axes

The campaign crosses:

- launch anchor: prepared seed or first completed generation;
- launch preparation: 0, 1, 4, 16, or 64 deterministic sweeps, or one
  noiseless activity-gated generation;
- seed ensemble: expected-half hash, exact-half, or density-stratified;
- process noise: before every rule step, after every rule step, or once at
  generation termination;
- activity count: realized stored-row changes or deterministic rule changes;
- monochrome death: terminal only, immediately on a realized row, on a
  realized row after the four-sweep floor, or immediately before noise when
  the deterministic rule output is monochrome; and
- observed daughter: terminal before copy error or offspring after copy error.

The terminal-once activity duplicates are canonicalized, leaving 1,440 unique
cells.  Density-stratified seeds and terminal-once process noise are marked
stress-test-only and cannot supply the accepted setting.

## Stages

The 88 canonical orbits are split 44/44 by alternating retained strict rate
within Wolfram class; one class-2 rule is deterministically moved to balance
the halves.  The split and complete design are written with a digest before any
scientific cell starts.

1. Screen all 1,440 cells on development rules at 4 seeds x 8 futures.
2. Promote at most 24 Pareto/frontier cells and run development at 16 x 32.
3. Freeze four finalists and expose the holdout at 16 x 128.
4. Run the best two accepted-or-leading cells on all 88 rules with a fresh
   16 x 128 seed block.
5. Propagate the best setting through atlas, phase, particle, and evolution;
   separately rescore Life ensemble-form pooling without resimulating a setting
   grid.

Every completed cell is an atomic checkpoint.  The runner is resumable and
maintains a machine-readable `STATUS.json`.

## Frozen strong-match bar

Holdout acceptance requires strict/break Spearman at least 0.75/0.85,
strict/break MAE at most 0.05/0.10, survival/clock Spearman at least 0.75/0.80,
raw class-4 strict at most 0.005, and retained class-3 separation.

Fresh all-rule acceptance tightens strict/break Spearman to 0.80/0.90 and MAE
to 0.04/0.08; at least four reference champions must enter the top ten; rules
13, 28, 156, and 172 must each fall to strict at most 0.05; and class break
medians must lie within 0.10 of the retained medians.

The overall round succeeds only if the downstream phase, particle, Life, and
evolution parity checks also pass.  Otherwise the runner writes
`FABLE_QUESTIONS.md`, including the leading settings and requests for minimal
code-free golden traces.

## Pre-run diagnostic

A deliberately underpowered, read-only probe established the priority of the
launch axis before this protocol was frozen.  Replacing the prepared-seed
anchor with the first completed generation reduced strict-rate MAE from about
0.14 to 0.02–0.03.  It did not recover the rule ranking.  The largest false
champions also have radically different survival: rules 13 and 172 survive 32
generations in the first clean-room run but only about 2.5 and 0.5 generations
in the retained table.  Launch anchoring and death timing are therefore primary
axes; the exhaustive cross-product remains frozen rather than narrowing after
this observation.

