# Chapter 5 Phi-r strongest-fair-test program: R0/R1

Status: prospective Codex registration. R0 must be sealed before any new
remeasurement is produced. R1 must remain locked until R0 selects exactly one
estimator by the rule below and a human explicitly authorizes a fresh run.

## Immutable prior result

The completed Chapter 5 pilot, window bridge, feedback-dose experiment, and
PAB24 protocol adjudication remain unchanged. In particular, sustained
stabilization did not raise the registered raw public-PhiRL nine-atom Phi-r in
Codex, and PAB24 classified the cross-clean-room disagreement as residual at
the simulator/trajectory level. This program cannot overwrite that result.

R0 asks a narrower development question: does one theory-motivated member of
the broader Phi-r family respond coherently after correcting three prospective
measurement concerns--Gaussian-copula preprocessing, an arm-independent
physical partition, and normalization against matched-information nulls?

## R0 sealed substrate

R0 generates no new scientific catalytic matrix or intervention outcome. It
deterministically replays the following already sealed PAB24 lineages:

- all 24 PAB24 catalytic matrices;
- both Codex simulator candidates;
- both replicate lineages;
- the `FRESH` launch only;
- `EXHAUSTIVE` `STABILIZE` and `DESTABILIZE`, plus `NOOP`;
- 60 controlled fissions with edits after fissions 1--59;
- primary measurement over fissions 31--60.

The runner reads the archived selected edits, but it regenerates every future
from the original arm-free future seed. Before any new score is admitted, it
must reproduce the archived record digest, final RNG state, final composition,
inherited fraction, and every legacy score to the existing numerical tolerance.
The complete R0 replay is then repeated independently and must match exactly.

## Preprocessing and partition

The parity control retains the frozen pseudocount-0.5, closure, CLR drop-last,
active-coordinate z-score, adaptive lag-one MI graph, and Fiedler partition.

Every new candidate instead uses:

1. pseudocount 0.5, closure, and all 100 CLR coordinates;
2. removal only of coordinates with standard deviation at most `1e-8`;
3. average ranks independently by coordinate;
4. the normal score `Phi^-1((rank - 0.5) / n)` followed by numerical z-score;
5. the frozen covariance ridge already used by the Codex Phi instruments;
6. one physical partition per beta matrix, obtained by deterministic Fiedler
   splitting of `0.5 * (log1p(beta) + log1p(beta.T))` with zero diagonal.

The physical partition is fixed across candidates, replicates, arms, and
representations. It may not be recomputed from an intervention trajectory.

The primary representation concatenates growth observations from fissions
31--60 (`fable_style`). Registered scale sensitivities are the existing
16-point phase-normalized trace and the generation-to-generation trace.

## Frozen estimator menu

Candidates are considered in this order; outcome magnitude cannot reorder them.

1. `NUMIT_MACRO`: public nine-atom revised Phi-r calculated on the two
   beta-partition macro averages and expressed as a NuMIT percentile/probit.
2. `PARTITION_NULL_FULL`: full-block revised Phi-r expressed relative to 128
   balanced random partitions of the same trajectory.
3. `FULL_BLOCK_RAW`: the full-block revised quantity

       I(X;X') - I(A;A') - I(B;B')
       + min{I(A;A'), I(A;B'), I(B;A'), I(B;B')}.

4. `COPULA_PUBLIC_RAW`: the public nine-atom revised Phi-r on the two physical
   macro averages without null normalization.

All four raw readings, whole-system MI, all 16 public atoms, downward causation,
emergence, synergy persistence, partitions, and validity diagnostics are
retained regardless of which candidate is selected.

### NuMIT reference family

For each observed transition count rounded to the nearest positive multiple of
16, generate 4,096 stable bivariate Gaussian VAR(1) systems. Draw a raw 2x2
normal transition matrix and rescale it to a spectral radius drawn uniformly
from `[0, 0.995]`. Draw a lower-triangular innovation factor with log-normal
diagonal and normal off-diagonal entries, form its positive-definite covariance,
and normalize the covariance trace to two. Burn 512 observations and retain the
registered bucket length. Score the same public nine-atom estimator.

For an observed window, select the 256 reference systems nearest in
whole-system lag-one MI. The empirical mid-rank percentile and its clipped
normal-probit transform are retained. An observation outside the sealed
reference MI range is invalid; libraries may not be regenerated after outcomes
are inspected. At least 95% of paired-arm lineages must be valid.

### Matched-partition null

For each window, draw 128 random label partitions preserving the sizes of the
physical blocks. Use the same partition draws for paired arms of one matrix,
candidate, replicate, and representation. Because the trajectory is unchanged,
whole-system MI is held exactly fixed. Report the physical score, null mean,
null standard deviation, z-score, and empirical percentile. This is a
matched-partition null, not NuMIT.

## R0 selection and inference

The catalytic matrix is the inference unit. Candidates and replicates are not
pooled. Replicates remain four separate candidate-by-replicate cells. Report
4,096 whole-matrix bootstrap draws, 4,096 paired whole-matrix sign
randomizations, Holm-adjusted p-values, and every matrix effect. These R0
inferential quantities are descriptive development evidence only.

The first estimator in the frozen menu is selected only if:

1. its `STABILIZE - DESTABILIZE` point estimate is positive in all four cells
   under the primary representation;
2. at least 22 of 24 leave-one-matrix-out estimates are positive in every cell;
3. either phase-normalized or generational measurement is positive in at least
   three of four cells;
4. at least 95% of paired lineages are finite and, where relevant, inside null
   support;
5. permutation, numerical, replay, and readback gates pass.

If no candidate qualifies, R1 remains locked. R0 cannot confirm a Phi-r claim.

## Locked R1 contract

If R0 selects one estimator and a human separately authorizes R1, freeze that
estimator and run 24 completely fresh matrices, both candidates, two fresh
replicates, and `STABILIZE_EXHAUSTIVE`, `DESTABILIZE_EXHAUSTIVE`, and `NOOP`.
Use the same 60-fission control and final-30 measurement contracts, common
future streams, 4,096 matrix bootstraps, 4,096 matrix sign randomizations, Holm
correction across four cells, and complete replay.

R1 passes only if both inherited fraction and the frozen information score have
positive stabilization-minus-destabilization effects, positive 95% matrix
bootstrap lower bounds, and Holm-adjusted one-sided `p < 0.05` in all four
cells; every leave-one-matrix-out information effect must remain positive.

No 48-matrix run is authorized by this registration. Functional/flux,
shared-break, and explicitly interventional Phi-r phases require new additive
registrations after R1 review.

## Integrity, execution, and claim boundary

Validation covers exact public-PhiRL parity, rank-copula behavior, simultaneous
composition/beta permutation, fixed partitions, analytical Gaussian fixtures,
NuMIT calibration fixtures, matched-partition invariants, arm-free futures,
domain-separated null streams, matrix-block inference, serialization, status,
and complete replay. A smoke run uses artificial non-scientific fixtures and
does not report arm effects.

R0 runs detached with at most 12 workers, durable checkpoints, a 1.5 GB free
disk floor, and a registered 30 CPU-hour pause boundary. Large tables, null
libraries, arrays, checkpoints, and logs remain excluded from git. Compact
reports, manifests, hashes, and ledger entries remain durable.

A later R1 pass could support a formulation-specific association between
causal hereditary stabilization and relative or full-block information
integration in Codex. It cannot erase the legacy negative result or establish
consciousness, agency, biological memory, life, Platonic ontology, or a
universal origin-of-life mechanism.
