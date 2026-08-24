# Preregistration: realistic GRN replication of PH F12 prediction

Status: implementation-frozen scientific contract. The canonical numerical
contract is `protocols/grn-f12-v1.json`; this document explains it.

## Claim and hierarchy

The independent unit is a seed-addressed random GRN, with no eligibility filter
based on its outcome. The powered primary test uses continuous noisy expression.
A smaller explicit mRNA/protein tau-leap tier tests portability across biological
realism. A continuous pass is sufficient for the primary claim. Molecular
success upgrades it to cross-realism confirmation but molecular failure does not
erase a continuous result. Interventions receive a separate mechanistic verdict.

## Histories and futures

Each network burns in for 16 uncued generations. Opposite A/B cues act on six
seed-selected genes for four generations. After release, states are retained at
ages 0, 2, 4, 8, and 12. Each retained state starts a frozen equal-half future
panel. There is no exact-state injection in the observational prediction test.

The continuous model has 32 genes, directed edge probability 0.125, balanced
signs, log-normal magnitudes, row normalization, 16 Euler-Maruyama regulatory
substeps per generation, molecularly motivated binomial partition at division,
and concentration noise. The bridge has 16 genes, edge probability 0.25,
integer mRNA/protein counts, volume growth, Poisson births, binomial deaths, and
binomial cytokinesis over 32 tau-leap steps per generation.

## Endpoint and F12

Continuous expression is clipped and logit transformed. Molecular protein
concentration is `log1p` transformed. Genes are centered before similarity.
Similarity is `(1 + Pearson correlation) / 2`; a zero-variance pair scores one
only when identical, otherwise zero. Each tier calibrates its threshold as the
median, across calibration networks, of the within-network fifth percentile of
uncued parent-daughter similarities. Thresholds at 2.5% and 10% are sensitivity
analyses.

`JOINT_BREAK_RUN3/F12` occurs when a parent-daughter similarity at or below the
threshold is followed, within 12 descendant boundaries, by three consecutive
similarities above it. Break incidence, conditional run-three incidence,
coherence, old-anchor separation, run-five, and F24 are retained; only F12 is
primary.

## Frozen cohorts and predictor

Continuous calibration/development/confirmation contain 64/128/320 networks and
256 futures per retained state. Molecular cohorts contain 32/64/160 networks and
128 futures per state. Networks, cues, states, folds, futures, halves, bootstrap
draws, and permutations use disjoint SHA-256 semantic random domains.

History and structural comparators are two-stage ridge hurdle models. The full
model is a permutation-equivariant, three-layer, width-64 signed message-passing
network with mean/std/max pooling, a fixed ten-variable history panel, and heads
for break and recovery conditional on break. Five whole-network development
folds produce an averaged frozen ensemble. Molecular models are trained fresh
under the identical architecture/hyperparameters. Continuous-to-molecular
zero-shot transfer is secondary.

## Confirmatory gates

Inference uses 4,096 whole-network bootstrap resamples and 2,048 whole-network
permutations, with Holm adjustment. All of the following are conjunctive:

- empirical split-half reliability `q >= 0.80`;
- full versus history log-loss gain at least 0.02 nats, with adjusted lower bound
  above zero in each future half;
- full versus history Brier gain with adjusted lower bound above zero in each
  future half;
- overall Spearman at least 0.60, median within-network Spearman at least 0.30,
  and positive correlations in both cue strata;
- adjusted whole-network permutation `p <= 0.01`;
- complete records, semantic-coordinate uniqueness, independent confirmation
  regeneration, checksums, and replay.

Full versus structural prediction is a separate mechanistic-discrimination
result and cannot replace the full-versus-history primary contrast.

## Controls

The frozen full predictor selects high- and low-risk states in the first 80
continuous and first 64 molecular confirmation networks. Common-random-number
arms are self continuation, exact state transplant, basal reset, node shuffle,
and inheritance erased at every division. The self high-minus-low F12 gap must
be positive and at least 0.05; exact transplant must preserve at least 90% of it;
reset, shuffle, and erasure must each preserve at most 30%.

## Operations

The two L4 GPUs run deterministic dynamic shards. An eight-network-per-tier
discarded benchmark must project all scientific work, doubled confirmation, and
controls below 10.5 hours after a 25% margin. New shards stop at 11 hours,
closeout begins by 11.5 hours, and 12 hours is hard. Free-disk guards are 40 GiB
before admission and 30 GiB while running. CPU fallback is an explicit error.
Raw state summaries, endpoints needed for regeneration, frozen models, selected
audits, manifests, and checksums target less than 30 GiB.

