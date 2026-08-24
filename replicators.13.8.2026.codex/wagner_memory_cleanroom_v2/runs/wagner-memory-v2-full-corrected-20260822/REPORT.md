# Corrected Wagner memory replication v2

Overall registered verdict: **WAGNER_MEMORY_STACK_CONFIRMED**

This run kept strict eight-cycle point-form retention separate from the first
three-cycle A/B/other point destination used for prediction, used two
complementary midpoint starts per rulebook, arm-paired random futures, and
whole-rulebook simultaneous bootstrap bounds. Verification
was successful for exact
stage counts, independent state/carrier regeneration, ordered per-cell future
digests, source records, and the registered future-ID replay sample.

## Expression-state channel

- Verdict: STATE_CHANNEL_CONFIRMED
- Direct within-treatment A/B crossover: 0.2628
  (simultaneous lower bound 0.2388)
- Risk gain over reset: 0.0676
- Held-out history log-loss gain: 0.0629
- Split-half crossover reliability: 0.9374
- Self-continuation/transplant pathwise identity: True

## Renewable lineage carrier

- Primary verdict: LINEAGE_CARRIER_CONFIRMED
- Causal verdict: CAUSAL_CARRIER_SUPPORTED
- Distributed bottleneck verdict: DISTRIBUTED_CARRIER_SUPPORTED
- Generation-4 direct crossover: 1.0000
  (simultaneous lower bound 0.9629)
- Generation-4 risk gain over zero carrier: 0.6502
- Held-out history log-loss gain: 0.6854
- Split-half crossover reliability: 1.0000
- Ablation loss fraction: 0.9507
- Rescue fraction: 1.0000
- Targeted k=5 retention fraction: 0.6274

## Other registered stages

- Writer/noise boundary: NOISE_BOUNDARY_REPRODUCED
- Slow passive mark: NO_SLOW_MARK_CONFIRMED

All control outcomes and adjusted bounds are retained in `analysis/*.json`; this
report does not convert a failed gate into partial confirmation.
