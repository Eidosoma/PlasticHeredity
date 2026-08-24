# Independent CA lineage-renewal replication

Verdict: `NO_DURABLE_RENEWAL`.

The fixed `motif_energy512-w32-s025-d32` reader and preregistered strict-49--64
daughter writer with universal gain 0.5 were tested on 96 fully fresh matched
founder pairs, 64 futures per history, and 16 visibly reset generations. The
independent unit was the founder pair; intervals use 10,000 pair-cluster
bootstrap draws at alpha 0.0125.

## Original-form persistence

- Generation 4: 0.0441 [0.0242, 0.0651]
- Generation 8: -0.0347 [-0.0559, -0.0137]
- Generation 16: -0.0358 [-0.0544, -0.0174]
- Terminal observer at generation 8: -0.0400 [-0.0600, -0.0194]

The intact lineage missed the registered original-form gates. Its crossover was
already small at generation 4 and was negative at generations 8 and 16.

## Causal renewal

- No-rewrite generation 8: 0.6995 [0.6503, 0.7489]
- Active-rewrite advantage at generation 8: -0.7342 [-0.7856, -0.6790]
- Opposite rescue at generation 4: -0.1069 [-0.1292, -0.0843]
- Opposite founder at generation 8: -0.0233 [-0.0436, -0.0036]
- One-percent corruption at generation 8: -0.0387 [-0.0592, -0.0177]

The fading, non-rewritten founder carrier retained a strong signal, while active
daughter rewriting destroyed rather than renewed that signal. The registered
no-rewrite loss fraction is undefined because the intact generation-8
denominator was non-positive; it is encoded as JSON `null`. The directly paired
active-rewrite advantage was negative, so the causal-renewal gate clearly
failed independently of that diagnostic ratio.

The opposite-history rescue reversed generation 4 in the predicted direction,
showing short-horizon steering, but same-history rescue could not establish
durable renewal. The complete gate table is in `RESULTS.json`.

## Reporting amendment

The original report command failed only while serializing negative infinity in
the undefined loss-ratio diagnostic. `REPORTING_AMENDMENT.json` records the
single conversion to JSON `null`. No trajectory, checkpoint, estimand, gate, or
verdict was changed.

Claim boundary: synthetic CA lineage memory only; no metabolism, agency, biological-life, or extra-automaton memory claim.
