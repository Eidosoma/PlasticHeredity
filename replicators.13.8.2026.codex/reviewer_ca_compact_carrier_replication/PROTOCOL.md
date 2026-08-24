# Prospective protocol: fresh compact-carrier replication

## Inputs and clean-room boundary

Only the Stage-4 protocol, design metadata, confirmation metadata, and codec
model data are snapshotted from NewIdeas.  No source code or source outcome
files enter the implementation or evidence bundle.  The known positive result
motivates a direct replication but contributes no observations to inference.

## Fresh acquisition

Generate 512 new 64-sweep noisy trajectories from each of four frozen sparse
launch resets (2,048 candidates total; Rule 31649; process noise 0.002).
Accumulate 2×2 texture counts over sweeps 57–64.  Admit a trajectory only when
its cosine similarity to a frozen A or B prototype is at least 0.95 and its
margin over the other prototype is at least 0.05.  Reject dead, unresolved,
duplicate, and historically identical states.  Match A/B donors within launch
at density difference at most 0.02.  Hash-order and freeze 4 engineering, 128
confirmation, and 32 untouched audit-reserve pairs.  If fewer than 164 pairs
exist, stop as underpowered without changing a threshold or extending the bank.

## Carriers

The frozen candidates are identity-r512-f32 (16,384 inherited bits),
pca-r008-q04 (32 inherited bits; 131,328 shared codebook bits), and
walsh-r016-q04 (64 inherited bits; 656 shared codebook bits).  Four-bit codecs
use `q = clip(rint(7*c/scale), -7, 7)` and decode `q*scale/7`; zero is exact.
All bases, scales, shapes, hashes, finite values, and orthonormality are audited.

## Confirmation

Each of 128 fresh matched pairs receives 64 futures per ancestral history for
16 generations, with checkpoints 1, 2, 4, 8, and 16.  The ordinary environment
uses process noise 0.002.  Moderate stress uses process noise 0.004 plus
independent 10% latent erasure and 5% latent sign corruption at every boundary,
including the founder boundary.  CA reader/process randomness is paired across
history, condition, codec, and environment.  Damage masks are paired across
history and condition but keyed by candidate and future.

Boundary order is lineage intervention, environmental damage, decode,
decoded-address intervention, reader, daughter writer, gain 0.5, and encode.
No-rewrite re-encodes 0.5 times the decoded inherited payload.

The twelve conditions are intact, zero every boundary, decoded shuffle, latent
shuffle, read disabled, founder write disabled, no rewrite, ablation entering
generation 3, same-history rescue entering generation 4, opposite-history
rescue entering generation 4, opposite founder, and 1% latent sign corruption.

Founder-pair cluster bootstrap intervals use 10,000 resamples and alpha 0.005
per codec.  The strict gates require durable bidirectional effects, survival,
terminal-observer agreement, four negative controls, active rewriting,
ablation loss, correct rescue, opposite rescue reversal, opposite-founder
reversal, corruption tolerance, and a latent-shuffle advantage.  The targeted
replication requires identity ordinary plus Walsh ordinary and moderate.  PCA
cannot replace Walsh.  Walsh-minus-identity at generations 8 and 16 is a
registered non-gating secondary contrast.
