# Forensic disposition of the sealed v1 run

The sealed v1 trajectories and reports are preserved byte-for-byte. They are
valid observations of the implementation that produced them, but they are not
a direct replication of the intended Stage-3R mechanism and must not be used
to accept or reject that mechanism.

Three comparability-breaking differences were identified:

1. The v1 reader flipped every favourable cell with a constant probability
   equal to `strength`. Carrier magnitude was ignored. Consequently gains
   0.5, 1, 2, and 4 generated identical decisions and a decaying stale carrier
   kept full steering strength until numerical underflow. The recovered reader
   instead scales probability continuously with energy advantage.
2. The v1 lineage used a pair-specific random reset with exactly 128 live cells.
   The intended reset is the sparse launch-specific initial state, shared by A
   and B and containing 3 to 6 live cells in the four launches.
3. The v1 lineage applied process noise before reading. The intended sweep
   applies the reader before process noise.

A separate reporting defect returned negative infinity when a diagnostic ratio
had a non-positive denominator. The v2 inference layer represents such ratios
as structured undefined values and never serializes NaN or infinity.

The original retained data were audited independently of implementation code:
all 96 pair identities were unique, cohort/design bindings and denominators
were consistent, recorded probabilities had the expected 1/64 granularity,
and the published aggregates recomputed exactly. No arithmetic or data-integrity
defect was found in those retained result files. Those source outcomes remain
context only and supply no evidence for the local v2 verdict.
