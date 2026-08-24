# Prospective strict-eight switch-lock protocol

## Questions

1. Does B captured at the third daughter of a registered strict-eight window
   enrich static lineage identity relative to the third daughter of a
   pre-event-geometry-matched F12-only recovery?
2. Does a four-generation target-specific carrier wave combined with reduced
   turnover stabilize B, and does that stabilization persist for 24 generations
   after the support is removed?

Both questions require fresh matrices and future seeds. Earlier outcomes only
fixed the hypotheses.

## Cohort and donor selection

Both reconstructed candidates run on the fixed anchors `(beta,leave)=(1,1)`
and `(2,2)`. For each fresh catalytic matrix, a fixed pool of F32 lineages is
generated. A strict donor uses the first registered eight-daughter window: all
eight boundaries have H>0.90, all daughters have minimum pairwise cosine >0.90,
and every daughter is <=0.85 from the pre-break parent. Its B is the third
daughter of that window, not the eighth. An F12-only control uses the third
daughter of the first post-break run of three ending within F12 and has no
registered strict-eight window anywhere in F32. B therefore has the same local
event age in both groups; global event time is an explicit matching feature.

Each eligible matrix-anchor requires two strict and two unique F12-only donors.
Strict donors are selected by frozen semantic hash. Controls are matched
without using composition identity: break time, run start, mass, first-three
minimum pairwise coherence, and first-three maximum pre-break similarity.

## Fresh forks and arms

For each donor type, arm, and future replicate, two independent forks are run
from each of the two parents. Same-parent similarity averages A1/A2 and B1/B2;
cross-parent similarity averages A1/B1 and A2/B2.

The eight fixed arms are:

- `control`: unmodified anchor dynamics;
- `quench`: absolute leave multiplier 0.5, uniform reservoir;
- `wave`: anchor turnover plus correct target wave;
- `quench_wave`: quench plus correct target wave throughout F32;
- `quench_static`: quench plus a constant correct target field;
- `quench_shuffled`: quench plus a spectrum-matched shuffled target wave;
- `quench_phase_pi`: quench plus the correct wave shifted by half a cycle;
- `pulse_release`: correct quench-wave for F1-F8, then unmodified anchor
  dynamics through F32.

The wave uses the 32 beta-influence-ranked molecule coordinates, coupling 2,
period 4, and nonnegative amplitude
`a(g)=(1+cos(2*pi*(g-1)/4+phi))/2`. The reservoir is the softmax of
`2*a(g)*writer(B)`. It is an engineered target cue, not a latent GARD variable.

## Endpoints and gates

Static cosine at F8/F16/F32 is primary. Phase-aligned similarity is the maximum
mean last-eight similarity over the four registered cyclic lags and is
secondary. Target residence, occupancy, terminal coherent target capture, and
completion are corroborating endpoints.

The primary anchor is `(1,1)`; `(2,2)` is transportability. Whole catalytic
matrix is the inference unit and candidates remain separate.

- Strict enrichment requires, in both candidates at the primary anchor,
  strict control same-minus-cross F8 lower 95% bound >0.10, F32 lower bound
  >0.05, and the strict-minus-F12 difference-in-differences lower bound >0.05.
- Strong static identity additionally requires strict same-parent F8 lower
  bound >0.90. Failure cannot be rescued by a phase score.
- Autonomous switch-lock requires quench-wave gain over control and all
  content/phase/static controls while supported, plus pulse-release F32 target
  capture lower bound >0.20 and gain over control lower bound >0.10 in both
  candidates. Otherwise a supported-only effect is labelled external
  maintenance.
- Phase coding requires phase-aligned same-minus-cross to exceed its static
  counterpart by a lower bound >0.05 in both candidates; it is exploratory.

All intervals are matrix bootstraps. Eligibility and attrition are reported
before any conditional effect.

## Runtime and verification

A production-seed-safe benchmark chooses the largest predeclared A/B/C tier
projected to fit generation, analysis, and a complete exact replay within 6.5
hours. The soft stop is 7.5 cumulative hours and the hard stop is 8 hours.
Donor and future checkpoints are atomic. Exact replay writes a durable receipt
per checkpoint, so a restart never discards completed verification work.
