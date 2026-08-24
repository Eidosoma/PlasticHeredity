# CA substrate-transfer preregistration

## Status

This document fixes the design before calibration, pilot, or confirmation
outcomes are generated.  The unpublished motivating exercise is hypothesis
provenance only.  The development pilot is outcome-visible and non-evidential.
Confirmation is a separately authorized future stage.

## Models

### Spatial hypercycle protocell

- 256 x 256 periodic square lattice; states empty, X, and Y.
- Random-order replication, degradation, and diffusion sweeps.
- `p=1`, `pX=1-pY`, `aX=0.01`, initial 10 x 10 X patch with central Y.
- Mechanics grid:
  `pY={1e-5,1e-4,1e-3,1e-2,1e-1}` and
  `aY={1e-6,1e-5,1e-4,1e-3,1e-2}`.
- A localized individual is a radius-two periodic molecular-proximity cluster
  with at least 20 occupied sites and at least one Y.  Radius two tolerates the
  single-site gaps created by diffusion.  Following the published mechanism,
  division is certified when replicated Y molecules seed two Y-centred
  Voronoi lobes at least eight lattice sites apart, each lobe has at least 20
  molecules, and the separation persists for 64 sweeps.  Bridging X molecules
  do not postpone division indefinitely.

### Evoloop ecology

- Published nine-state, von-Neumann, rotate-four Evoloop transition table.
- 256 x 256 periodic lattice; synchronous updates.
- Initial loop counts `{4,8,16,32}` and Poisson immigration means
  `{0,0.25,0.5,1}` canonical loops per 10,000 ticks.
- Initial and immigrant canonical loops require an empty 9 x 9 placement box;
  periodic placement and orientation are seeded independently.
- A newborn loop is a disconnected component of at least 20 sites with at
  least one enclosed background region, 512-tick persistence, and a construction-arm
  launch within the following 4,096 ticks.
- Passive provenance plurality must be at least 0.80.

For either model, a mechanics cell is retained only when at least 12 of 16
screen seeds reach 20 unambiguous boundaries before 2,000,000 updates, global
occupancy never exceeds 25%, and ambiguity is below 5%.  All passing cells are
retained; none is selected by break-and-renewal outcome.  A single boundary
has a 100,000-update cap in mechanics, main-lineage, and future simulation;
all persistence and construction-arm lookahead updates count against it.

## Similarity and event

An individual is cropped, centroid aligned, and one-hot encoded without
background.  Similarity `S` is the maximum cosine under four rotations and
integer shifts -1, 0, and +1 on each axis.  Each model's threshold is the
NumPy `method="higher"` 95th percentile of matched-stranger similarities in a
disjoint calibration cohort.  Every stranger child is size matched to the
focal child and comes from a different independently seeded world block;
siblings and other members of the focal lineage are ineligible as strangers.  Store the threshold in
hexadecimal floating-point form.

Inheritance is strict `S > tau`; a break is `S <= tau`.  The primary F12 event
requires a break and, strictly later, three consecutive inherited boundaries.
Failure before certification is negative; later failure does not revoke a
certified event.

Every future is simulated for up to 16 boundaries.  The primary endpoint and
its completion flag use the first 12 only; boundaries 13--16 exist solely for
the preregistered F16 sensitivity.

## Cohorts and stopping

- Calibration: 24 blocks/model and at least 1,000 usable matched pairs across
  16 blocks.
- Pilot: 32 new blocks/model, landmarks 20/35/50/65/80, 32 F12 futures/state,
  deterministic halves 0--15 and 16--31.
- Pilot eligibility: 24 complete blocks, 70% complete F12 futures, 100 breaks,
  50 events, and events in eight blocks.
- **Mandatory stop after the sealed pilot report.**
- Later confirmation: 128 new blocks/eligible model, the same landmarks, and
  64 futures/state split 0--31 and 32--63.

No future is retried.  A main world may use at most 100 attempts; its block is
never replaced.  If all 100 attempts fail, every planned future in that block
is retained as an incomplete negative with zero observed boundaries.  Thus a
failed main world cannot disappear from event or sequence denominators.
Calibration, pilot, confirmation, replay, bootstrap, and randomization seeds
are disjoint.

## Primary gates

For a model to pass all of the following must hold:

1. Parent--offspring mean S exceeds matched strangers with a positive
   one-sided 97.5% whole-block interval; inherited boundaries are at least 50%
   and breaks at least 5%.
2. F12 event prevalence is at least 1% in both halves, with at least 100 events
   in at least 16 blocks.
3. In each half, observed event prevalence exceeds the exact order-randomized
   null preserving each future's observed length and inheritance count, with
   positive one-sided 97.5% interval and randomization `p <= 0.025`.
4. A/B statewise event-rate Spearman correlation after block centering has a
   positive one-sided 97.5% interval and `p <= 0.025`.

Inference uses 4,096 whole-block bootstraps of equal-weight block means and
4,096 whole-block sign randomizations.  The model-level alpha is 0.025 because
the headline permits either of two models to pass.  The within-model rule is a
conjunction; no endpoint or model can rescue another.

Raw `S>0.9`, F8/F16, and run lengths two/four are non-rescuing sensitivities.
The models are never pooled.

Raw rasters are retained in compressed per-block sidecars to make the
cross-block stranger comparisons auditable.  Source, protocol, calibration,
mechanics, raw block tables, raster sidecars, and reports are hash committed.
