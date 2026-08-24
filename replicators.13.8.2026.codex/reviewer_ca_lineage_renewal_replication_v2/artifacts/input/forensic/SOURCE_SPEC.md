# One-time recovered source specification

This document freezes the narrow implementation specification recovered during
the authorized one-time inspection on 2026-08-23. The inspected source is a
hypothesis/protocol aid only. It is not evidence for this replication, is not
imported or executed here, and must not be opened again for this campaign.

Only the functions needed to resolve the reader, state convention, reset,
ordering, writer, visible descriptor, and held-out decoder were inspected.
Unrelated implementation code was not inspected.

## Recovered operations

- A state hex string is converted to an integer. Flattened board position zero
  is the least-significant bit; the resulting 256 bits are reshaped row-major
  to 16 by 16. The inverse representation uses the same convention.
- A 3 by 3 motif is row-major and least-significant-bit first: the top-left
  position is bit 0 and the bottom-right position is bit 8.
- A 2 by 2 texture code is likewise row-major and least-significant-bit first:
  top-left, top-right, bottom-left, bottom-right are bits 0, 1, 2, 3.
- Each generation starts from the sparse `initial_state_hex` shared by the A
  and B donor of that launch. The same reset is used for every history,
  replicate, condition, and generation.
- Each sweep is ordered: Rule-31649 CA update; energy reader when the sweep is
  1 through 32; independent process noise; then writer/observer accumulation.
- For every proposed cell flip, the energy advantage is the sum across the
  nine containing 3 by 3 motifs of
  `carrier[flipped_address] - carrier[current_address]`.
- The flip probability is
  `strength * tanh(max(energy_advantage, 0) / 9)`. The registered strength is
  0.25. A uniform draw strictly below this probability flips the cell.
- The daughter writer accumulates 3 by 3 motif counts at sweeps 49 through 64,
  converts them to a clipped log-ratio carrier against the label-blind pooled
  reference, and applies the universal scalar gain 0.5.
- The primary observer accumulates live 2 by 2 bins 1 through 15 at sweeps 57
  through 64. The terminal observer uses the same bins at sweep 64.
- The 41-value visible descriptor is normalized accumulated 2 by 2 texture
  (15), normalized terminal 2 by 2 texture (15), occupancy (1), five raw
  spatial correlations at shifts `(1,0)`, `(0,1)`, `(1,1)`, `(2,0)`, and
  `(0,2)`, and five Fourier power entries at `(0,1)`, `(1,0)`, `(1,1)`,
  `(0,2)`, and `(2,0)`.
- The held-out decoder uses four deterministic replicate splits within each
  condition. For each split, half of each history's futures train two centroids
  and the other half tests them. Standardization is pooled across the two
  training histories; standard deviations below `1e-8` are replaced by one. A
  test item is correct only when its own-history centroid is strictly closer,
  so ties are counted as incorrect.

This file is part of the v2 implementation manifest and is sealed before any
v2 lineage outcome is generated.
