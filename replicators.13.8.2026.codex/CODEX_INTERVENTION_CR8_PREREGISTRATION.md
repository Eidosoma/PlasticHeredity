# Codex intervention CR8 preregistration

## Scientific question

CR7 established that repeated state-dependent substitutions can maintain high
compositional inheritance while feedback remains active. CR8 tests the stronger,
prospectively distinct claim: did sixty fissions of active stabilization install
an organization that persists without intervention and restores itself after a
subsequent molecular challenge?

This phase targets neither the strict-eight event nor Phi/PhiID. It does not
refit, recalibrate, or otherwise modify the frozen `JOINT_BREAK_RUN3` predictor.

## Frozen upstream material

CR8 is conditional on the sealed CR7 sixty-fission gate. Its preparation states
are the exact generation-120 endpoints of the six CR7 replicate lineages for:

- `MODEL_DOWN`;
- `RULE_DOWN`; and
- matched `NOOP`.

The CR7 result, registration, launch matrices, endpoint compositions, histories,
and checkpoint digests are verified before sealing CR8. They are copied into a
compact immutable launch-state archive. No CR7 lineage is selected, discarded,
or regenerated according to its CR8 outcome. The continued-active-control CR7
extension is not used.

Before release, the sealed CR7 summaries must show, separately for each candidate
and each written-state controller, higher inherited-boundary fraction than
`NOOP`, lower frozen risk than `NOOP`, and a change in at least one of entropy,
top-one abundance share, or catalytic throughput. These are preparation audits,
not new hypothesis tests.

## Cohort and release

The fixed cohort contains 48 matrices, two candidates, six CR7 replicate
lineages, and three preparation origins. All 1,728 states are released for sixty
additional fissions with no callback and exactly zero interventions. Extinction
or simulation failure is retained, never retried, and never replaced.

For the same candidate, matrix, and CR7 replicate, all three origins receive a
common release random stream. Origin identity is excluded from the release-future
seed. Diverged states can consume the common stream differently, so these are
common random streams rather than identical realized futures.

At every observed release boundary CR8 records:

- cosine similarity to the generation-120 preparation anchor;
- similarity of each written trajectory to its matched `NOOP` trajectory;
- frozen `JOINT_BREAK_RUN3` risk;
- strict inheritance (`H > 0.9`);
- entropy, occupied types, top-one abundance share, and `x.T @ beta @ x`;
- growth updates and survival.

The final-release inheritance statistic is the strict inherited fraction in the
last six registered release boundaries. An incomplete release has missing future
boundaries and is retained as such; it is not imputed as successful inheritance.

## Challenge construction

Each release-end composition is its own challenge anchor. The three origins are
challenged with six frozen arms:

1. `NONE`: no instantaneous edit;
2. `RANDOM_K2`;
3. `RANDOM_K4`;
4. `RANDOM_K8`;
5. `RANDOM_K16`; and
6. `ADVERSARIAL`.

A random `K` arm removes exactly K individual molecules sampled without
replacement. Its K additions are sampled uniformly with replacement from type
labels that were not removal-source labels. This guarantees nonnegative integer
counts, fixed mass, and exact compositional transport distance
`sum(abs(edited-original))/2 == K`; it also prevents cancellation from silently
reducing the registered dose. Edit selection uses a domain-separated stream.

`ADVERSARIAL` is the one legal molecular substitution with the greatest frozen
predicted increase in `JOINT_BREAK_RUN3` risk. Every legal substitution is scored.
Exact ties use the first lexicographic edit. Realized challenge outcomes never
enter selection.

Each arm launches 32 independent untreated F24 futures. For a fixed candidate,
matrix, CR7 replicate, and branch, the future seed excludes both preparation
origin and challenge arm. Thus all compared origins and arms receive common
random streams. Challenge-edit randomness never consumes a future stream.
Intervention futures are not retried. Every release and challenge future is
regenerated in a complete deterministic replay.

## Frozen classifier

Similarity thresholds use unrounded float64 cosine values.

- Departure: similarity to the challenge anchor is `< 0.7`, including the
  instantaneous post-challenge launch composition.
- Return: strictly after the first departure, similarity is `> 0.9` for three
  consecutive observed post-fission states.
- Mode recovery: the trajectory completed F24, did not qualify as returned, had
  at least five strict inherited boundaries among the final six, and its final
  top-one abundance share was `>= 0.45`.

The mutually exclusive category order is:

1. `held`: completed F24 and never departed;
2. `returned`: completed F24, departed, and certified return;
3. `mode_recovered`: completed F24, departed, did not return compositionally,
   and met the mode-recovery definition;
4. `lost`: every other outcome, including incomplete futures.

## Inference

The catalytic matrix is the sole inference unit. Replicates, origins, arms,
branches, and repeated time points from a matrix remain together in all draws.
Candidates are never pooled. The preregistered draw counts are 4,096 whole-matrix
bootstrap draws and 4,096 paired whole-matrix sign randomizations.

The external written-but-passive hypothesis is classified separately from any
evidence for an autonomous basin. It requires all of the following:

1. For both written origins in both candidates, the matrix-and-lineage mean
   similarity-to-preparation-anchor curve crosses below 0.7 during F1--F60.
2. For both written origins in both candidates, the 90% whole-matrix bootstrap
   interval for written minus matched-`NOOP` last-six inheritance lies wholly
   within `[-0.03, +0.03]` (TOST-equivalence criterion).
3. At each registered random dose `K in {0,2,4,8,16}`, where `K=0` is `NONE`,
   and for both written origins in both candidates, the 90% interval for written
   minus natural `held + returned` probability lies wholly within
   `[-0.05, +0.05]`.
4. No written origin has a significantly positive dose trend: the 95% lower
   bootstrap bound of the within-matrix OLS slope of written-minus-natural
   `held + returned` probability against K is not above zero.
5. The shared registered basin radius is zero. At dose K, a positive shared
   basin requires a positive 95% lower bound for written-minus-natural
   `held + returned` probability for both written origins and both candidates.
   The radius is the largest K satisfying this; none gives radius zero.
6. Complete release and challenge replay passes, release applies zero
   interventions, and all persisted artifacts pass exact readback checks.

The adversarial arm is a registered stress test and is reported with the same
matrix-block intervals, but it is excluded from the random-dose slope and basin
radius.

If the written-but-passive classification passes, the permitted term is
`controller-maintained compotype-like state`. If a nonzero shared basin is found,
it is reported as a cross-clean-room disagreement, without changing thresholds.

## Claim boundaries and stop rule

CR8 can distinguish active maintenance from passive compositional persistence
and restoration in these simulations. It cannot establish biological memory,
autonomous agency, life, error correction, real prebiotic chemistry, a universal
origin-of-life mechanism, or control of strict-eight or Phi/PhiID.

CR8 stops after its sealed result and complete report. CR9 is not launched
automatically.
