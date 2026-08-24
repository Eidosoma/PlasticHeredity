# Preregistered clean-room replication

Date frozen: 2026-08-22 UTC, before any fresh daughter outcome in this package.

## Fixed mechanism

The substrate is the 16 by 16 toroidal Life-like rule 31649,
`B13456/S0578`. States are encoded as big-endian row-major 256-bit strings.
Parent writers observe post-update states. Binary 3 by 3 and 2 by 2 addresses
are row-major with the top-left bit most significant.

The 512-entry writer uses Jeffreys-smoothed parent motif frequencies minus the
label-blind calibration log frequencies, clipped to plus or minus 4. After a
normal CA update, every daughter cell sums the carrier-energy change caused by
flipping across the nine motifs containing it. Improving flips are accepted
with the registered read strength. Decisions use one predicted board and are
applied synchronously. The contextual ceiling uses the signed difference in
Jeffreys-smoothed centre occupancy conditional on the eight-neighbour ring;
eligible flips occur with probability `strength * abs(mark)`. A zero carrier is
exactly inert.

The primary observer is accumulated live 2 by 2 texture over the trailing eight
sweeps. Terminal live 2 by 2 texture is the independent gate observer.
Assignments require cosine similarity at least 0.90 and a margin at least 0.05.
Dead and unresolved futures remain in denominators. Terminal occupancy,
8-connected toroidal component geometry, nearest-lag autocorrelation, and four
low-frequency Fourier powers are retained on intact arms as non-gating
diagnostics.

The existing positional benchmark is imported from the retained round-4
calibration: a 16-sweep occupancy latch with thresholds 0.60/0.40 and reader
strength 0.05; its boundary retention is 0.55. The intentionally
incomplete-reset control carries the first 64 visible parent bits into an
otherwise neutral daughter and disables the motif reader. Both are diagnostics,
not substitutes for the confirmatory motif arm.

## Fresh donor policy

The source snapshot contains 2,048 frozen acquisition donors. Every donor named
by any retained historical outcome, development, or parity pair is excluded.
Remaining A/B donors are paired within launch, never reused, and must differ in
density by at most 0.02. Pair construction and cohort allocation use frozen
SHA-256 ordering and no daughter outcome.

Stage 1 assigns 64 pairs to label-blind calibration, 48 to the full existing
64-configuration screen, and 64 to validation. The four retained historical
nominees receive the existing ten-condition validation panel. Fresh discovery
is descriptive: it cannot replace `motif_energy512-w32-s025-d32` as the sole
confirmatory primary. Inference uses 10,000 deterministic pair-cluster bootstrap
draws and the retained Bonferroni gates.

Stage 2 is unavailable until a complete robust Stage-1 pass is reviewed and an
explicit second registration is written. It quarantines two unused pairs,
audits the writer on 32 more, and uses 96 further pairs for the existing eight
primary environments, three density environments, causal controls, native
noise/corruption, and five registered doses. The reader is imported unchanged.

The default is eight single-threaded workers to coexist with the separate
stress-testing campaign. Per-pair checkpoints are atomic and bound to the input,
design, implementation, and cohort hashes. An incomplete run cannot pass.
There is a mandatory stop after Stage 1. No Stage 3 and no additional experiment
are part of this package.
