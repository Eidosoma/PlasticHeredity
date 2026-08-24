# PX6 procedural amendment 001 — archived PX2 state-key namespace

Date: 2026-08-22

This amendment supersedes PX6 registration
`50d7e122ea37da68cbdcf716b235ebf4c5d4305c15f7b5e81fd4abf43ac43576`
before any PX6 result was produced.

## Failure and timing

The first detached analysis launch stopped while reading the archived PX2
table. The PX6 dataset contract named its within-matrix state key `landmark`,
but the sealed PX2 table calls that same event-locked state key `break_step`.
Pandas raised `KeyError: 'landmark'` before the correction grid, bootstrap,
randomization, classification, report, or result directory was produced.

The failed registration, launch record, and traceback are retained. No PX6
effect size or classification existed when this amendment was written.

## Authorized correction

For PX2 only, replace the nonexistent within-state column name `landmark` with
the archived column name `break_step`. This preserves the registered
operation: pair RENEWAL_UP and RENEWAL_DOWN within each event-locked restored
state, then average those paired effects within catalytic matrix.

No scientific quantity changes:

- the five archived input files and their hashes are unchanged;
- the high and low arms are unchanged;
- the full-base and redundancy columns are unchanged;
- the correction grid remains `(0, 0.25, 0.5, 0.75, 1)`;
- all bootstrap, randomization, matrix-block, and classification rules remain
  unchanged;
- no new matrix, future, outcome, or favorable correction is introduced.

## Validation correction

The original unit tests checked synthetic contracts but not the column names
of all five real archived tables. Before re-registration, validation must now
read only each CSV header and require every registered matrix, arm, grouping,
within-state, filter, base, and redundancy column to exist. It must explicitly
require PX2's within-state key to be `break_step`.

## Claim boundary

PX6 remains a forensic robustness analysis of the material full-block
statistic. It cannot rescue the public nine-atom Phi-r result, correct PX3's
sample-support instability, or support consciousness, agency, or life.
