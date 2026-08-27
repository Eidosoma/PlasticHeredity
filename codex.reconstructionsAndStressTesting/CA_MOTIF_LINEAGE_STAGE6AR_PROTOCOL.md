# Preregistered CA motif-lineage Stage 6A-R: corrected local bridge

## Status and ancestry

Stage 6A completed with the registered decision
`no_bounded_candidate_passed_registered_gate`.  Its two historical
diagnostic anchors retained generation-8 crossover means of 0.949 and 0.959,
but every candidate executed by the new bounded engine lost its lineage
centroid after the first rewrite and reached zero crossover.

Post-run inspection found a normalization error in that engine.  The directed
rectangle reducer returns a spatial mean, while the writer divided that mean
by the rectangle area a second time.  For a rectangle containing `N` sites
and `T` writing sweeps, Stage 6A v1 used

```
(mean_signed_sum + alpha * sign_prior) / (T * N + 512 * alpha)
```

instead of the count-equivalent expression

```
(mean_signed_sum * N + alpha * sign_prior) / (T * N + 512 * alpha).
```

The v1 error attenuates the empirical moment by approximately `1/N`; at full
span this is a factor of 256.  Stage 6A also independently relocated the
one-site seed between generations, so a bounded writer could sample a region
outside the parent's causal cone.  Consequently Stage 6A is retained as an
immutable negative result for its v1 mechanism, but it is not treated as a
clean rejection of bounded local Plastic Heredity.

Stage 6A-R is a developmental repair study.  It never overwrites Stage 6A,
never opens the 62-pair final reserve, and cannot automatically launch Stage
6B or the final audit.  Claims remain limited to engineered synthetic CA
heredity.

## Corrected mechanics

The inherited object remains the confirmed 64-bit `moment16-ridge` payload
plus one occupancy bit.  Every visible daughter is reset bit-for-bit before
development.  Payload propagation remains synchronous occupied
Moore-neighbour consensus copying.  The reader and writer have no lineage
label, parent state, target phenotype, or prototype access at runtime.

The corrected writer converts the reducer's spatial mean back to its
count-equivalent signed sum before applying the same Jeffreys prior and frozen
Stage-5R writer.  The legacy formula is retained only as a non-promotable
diagnostic.  Full-span corrected moments must match direct global motif-count
moments to `1e-12`, and their quantized payloads must be exactly equal on a
frozen trace before any biological-style claim is evaluated.

Three seed-origin policies are registered:

- `co-located`: every daughter seed remains at its founder coordinate;
- `adjacent`: each daughter buds by exactly one Moore-neighbour edge;
- `independent`: every generation is independently relocated and is a
  non-promotable causal-geometry control.

A translated-seed control translates both reading and writing coordinates;
it may not silently separate the seed from its writer endpoint.  Random
fields are semantic and paired across candidate configurations, worker
counts, and resume boundaries.

## Five separately invoked phases

Each phase is a separate detached invocation with at most four workers and a
four-hour wall limit.  No phase launches its successor.

### 1. Audit

The audit freezes and hashes the completed Stage 6A evidence, proves the
normalization identity for every registered span, proves full-span equality
with the frozen Stage-5R compact writer, reproduces the legacy lineage-centroid
collapse, audits decoder ties, and verifies that the final reserve is still
unloaded.  Failure stops all later phases.

### 2. Bridge

The bridge uses the disjoint exposed `scale` cohort: 32 pairs in the reference
profile, four futures per history, and generations 1, 2, and 4.  It crosses
germination hops 2, 4, 5, and 8; writer spans 0, 2, 4, 7, and 15; and all three
origin policies.  A small component panel compares the field-local,
predecoded-local, and nonlocal-global readers.  Historical Stage-5R compact
and exact objects are diagnostics.

The corrected full-communication bridge (`h8`, span 15, co-located) must have
a positive generation-4 bootstrap lower bound and retain at least 70 percent
of the frozen compact anchor.  Otherwise candidate search stops.  Independent
relocation and the legacy writer can diagnose causes but can never be
promoted.

### 3. Screen

At most six bounded co-located or adjacent bridge candidates advance to the
64 exposed Stage-6A screen pairs, eight futures per history, and eight
generations.  Intact, founder-clamped, no-rewrite, write-disabled,
transport-disabled, communication-cut, and translated conditions are run
with paired streams.

A candidate is screen-positive only if survival is at least 0.90, both
lineage directions are positive, at least half of pair effects are positive,
the generation-8 mean is at least 0.15 with a bootstrap lower bound above
zero, active rewriting has a positive advantage, translation preserves at
least 70 percent, and intact recovery retains at least 70 percent of the
corrected full bridge.

### 4. Qualify

At most two screen-positive candidates are tested on the already exposed 96
qualification pairs, 16 futures per history, and 16 generations.  This phase
requires explicit confirmation authorization.  It runs the complete causal
ladder: zero, shuffle, read-off, founder-write-off, write-off, no rewrite,
ablation, same and opposite rescue, opposite founder, corruption, transport
off, regeneration off, consolidation off, translation, communication cut,
and founder clamping.

Qualification requires the unchanged Stage-5R strict renewal gate, finite
light-cone compliance, all targeted local controls, positive generation-16
recovery, and at least 70 percent corrected-anchor retention.  Because these
pairs were exposed by Stage 6A, this remains developmental repair rather than
independent confirmation.

### 5. Endurance

Only a qualified winner is followed on the 32 exposed endurance pairs, eight
futures per history, through generation 64.  Intact, no-rewrite, corruption,
communication-cut, and independent-relocation conditions are compared.
Generation-32 and generation-64 effects must have positive bootstrap lower
bounds, and active rewriting must remain causally necessary.

## Reporting and stopping

Every phase writes atomic status, queue, checkpoint, manifest, result,
decision, technical-report, and lay-summary artifacts.  Carrier entry and
exit centroid distance, parent-child alignment, raw endpoint statistics,
corrected moment scale, quantized payloads, spatial coverage, origin
displacement, causal overlap, phenotype crossover, decoder accuracy, and
decoder tie fraction are retained at registered checkpoints.

Any failed gate is a result.  Thresholds cannot be relaxed inside Stage 6A-R.
The final reserve remains hash-only and untouched regardless of outcome.
