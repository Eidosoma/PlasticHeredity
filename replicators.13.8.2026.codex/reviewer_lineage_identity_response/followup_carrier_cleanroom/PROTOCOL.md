# Registered clean-room carrier protocol

## Question and cohorts

The intervention asks whether adding a bounded inherited lineage register can
turn strict-B texture into transmitted information rather than a shared rule
destination. Matrices 11, 54, and 63 are engineering-only. The other 47 frozen
strict-capable matrices form the untouched primary cohort under simulator
candidates 02 and 03. The first strict bank row is the single-form target. For
the multiform challenge, each candidate/rule uses the least-similar pair of
strict rows from distinct lineages with cosine similarity at most 0.85; only
rules eligible under both candidates enter the shared cohort. Pair choice and
all masks are frozen before carrier outcomes are generated.

## Carrier

The carrier is a bounded 100-entry molecule-indexed vector `m`. A composition
writer converts adult-parent abundance to a centered signal and divides by its
largest absolute component. A beta-only mask retains the `k` entries with the
largest total incoming-plus-outgoing catalytic influence. The matched random
mask has the same size. The read field is

`rho_i(m) = exp(mu m_i) / sum_j exp(mu m_j)`.

This replaces only the uniform reservoir factor during growth. Beta, Poisson
events, overshoot, fission, and daughter selection remain unchanged. With a
zero carrier or disabled reader the adapter calls the frozen simulator directly
and must reproduce it bitwise.

After adult growth, the inherited register is

`m' = d m + (1-d) writer(parent)`, where `d = 2^(-1/L)`.

Ideal copying has no additional damage. Nominal copying multiplies the register
by 0.95, independently drops each active coordinate with probability 0.02, adds
Gaussian noise with sigma 0.05, reapplies the mask, and clips to [-1, 1]. State
and carrier randomness use separate semantic seed streams.

## Engineering calibration and frozen selection

The engineering grid crosses `k={8,16,32,100}`, `L={1,4,8}`, `mu={0.5,1,2}`,
and ideal/nominal copying. Eight F32 futures are run for correct, zero,
opposite-history, shuffled-history, stranger/correct, and stranger/zero arms in
both candidates on all three engineering rules.

A calibration setting passes only when both candidates have native-correct
terminal strict-8 probability at least 0.50, gains of at least 0.20 over zero
and opposite carrier, at least 0.10 over shuffled carrier, and stranger
installation gain of at least 0.20. At most two settings are frozen before
confirmation: (1) the smallest nominal passing setting ordered by `k`, `mu`,
then `L`; and (2) the strongest passing setting by the minimum candidate score.
Duplicate selections collapse to one.

## Confirmation

The primary F64 arms are native/correct, matched stranger/correct,
stranger/zero, stranger/shuffled, native/zero, native/shuffled, reader disabled,
founder writer disabled, descendant renewal disabled, erase after generation 2,
erase plus exact carrier rescue at generation 3, unmodified no-carrier, and a
matched random mask where applicable. The two-form cohort runs reciprocal
state/carrier combinations A/A, A/B, B/A, and B/B.

Readouts are target arrival, terminal coherent strict-8 capture at F16/F32/F64,
capture anywhere by those horizons, occupancy, maximum residence, departure and
re-entry, extinction, nearest-A/B carrier-origin accuracy, carrier crossover,
and final-carrier decoding.

## Gates

Whole-matrix bootstrap intervals use 10,000 resamples and equal rule weighting;
candidates are reported separately.

- Constructive carrier memory: in both candidates native/correct F32 terminal
  capture lower bound >0.30; correct-minus-zero point >=0.20 and lower bound
  >0.10; positive superiority to shuffle, reader-off, founder-writer-off, and
  renewal-off; positive F64 persistence; erasure removes at least 70% of the
  correct-minus-zero effect and rescue restores at least 70% of it.
- Constructive multiform memory: both reciprocal cross-start target captures
  have lower bound >0.25; carrier crossover point >=0.20 and lower bound >0.10;
  carrier-origin accuracy lower bound >0.75 in both candidates.
- Compressed/noisy success requires the same gates under nominal copying and
  `k<=32`. A pass only at `k=100` is reported as engineered full-register only.
- Joint molecule/carrier relabeling must preserve registered rates within 0.03.

## Runtime and audit

A production-seed-safe benchmark selects the largest predeclared tier projected
under 6.75 hours after a 1.5 safety factor: A=64 futures/cell at F64, B=48 at
F64, C=64 at F32. The cumulative campaign soft-stops at 27,000 seconds and its
detached process group is hard-stopped at 28,800 seconds. Restarts cannot reset
the ledger. An incomplete tier has no verdict. Full completion requires exact
parallel replay, zero discrete discrepancy, zero maximum H discrepancy, and
verified protocol/output checksums.

