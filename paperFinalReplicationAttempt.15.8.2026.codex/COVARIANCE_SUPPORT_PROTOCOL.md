# Arrivals Phi-family covariance-support diagnostic

Status: prospective, label-blind numerical-development protocol. Scientific
support results may be generated only after this document, the implementation,
and its tests are source-hashed into a registration.

Execution amendment 001 is documented in
`COVARIANCE_SUPPORT_AMENDMENT_001.md`. It corrects only ambiguous pandas column
access after the first registered run failed before writing or evaluating any
score; all scientific contracts below are unchanged.

## Separation from completed work

The original negative replication, the registered 12-seed formulation bridge,
and its null report are immutable inputs. This diagnostic neither rescues nor
reinterprets them. It responds to evidence from the separate Plastic Heredity
project that a 100-coordinate full-block Gaussian-MI statistic can reverse
direction when its effective transition support changes from 128 to 256 or
512 pairs.

The self-replicator label gate remains unresolved as documented in
`LABEL_CONTRACT_STATUS.md`. This phase must not import the detector module,
read prior label arrays, calculate replicator labels, or inspect association or
prediction outcomes. Its only selection target is numerical behavior under
known synthetic truth and changing transition support.

No intervention, association pilot, forecasting pilot, threshold sweep, or
paper-scale run is authorized.

## Fixed development substrate

Generate exactly six untreated label-blind development trajectories with the
existing GARD parameters except for 160 generations, using seeds:

`26083101, 26083102, 26083103, 26083104, 26083105, 26083106`.

The longer duration provides a final pool of 512 adjacent transition pairs; it
is a numerical-development substrate and not an outcome cohort. A trajectory
with fewer than 512 pairs is retained as ineligible and is not replaced.

For each eligible trajectory, the primary audit fixes the final 513 molecular
observations, hence a common terminal state and 512-pair pool. All transforms,
partitions, and the PCA basis are fitted once on that fixed pool. For each of
12 deterministic repeats, randomly permute the first 511 pair indices and
append the terminal pair. The supports 64, 96, 128, 192, 256, 384, and 512 are
nested prefixes plus the same terminal pair. Thus every support comparison
uses the same trajectory, endpoint, transform, partition, and representation;
only covariance support changes.

A registered operational sensitivity uses the last `n+1` observations at each
support and refits that instrument's permitted preprocessing, partition, and
PCA basis. It has the same endpoint but intentionally includes normal
window-refitting effects. It is reported separately and cannot replace the
primary covariance-only audit.

## Frozen instruments

All instruments use lag-one explicit transition pairs. No cross-subsample
transition is invented.

1. `typeset_full_wms`: the public typeset numerator reconstruction. Apply
   pseudocount-0.5 CLR with the final coordinate removed, active-coordinate
   standardization, the lagged-MI Fiedler split, and the full-coordinate
   quantity `I(X;X') - I(A;X') - I(B;X')`. This is a reconstruction of the
   public equation, not claimed parity with the authors' private estimator.
2. `macro_wms`: the existing reconstruction's same CLR/Fiedler partition,
   after averaging each half to one macro variable, scored as
   `I(AB;A'B') - I(A;A'B') - I(B;A'B')`.
3. `public_nine_atom`: all 100 CLR coordinates, average-rank Gaussian copula
   transform, beta-derived physical Fiedler partition, two block averages, and
   the frozen public-PhiRL nine-atom revised-Phi-r sum.
4. `pca8_full_revised`: the sole prospective stabilized candidate. Use the
   same all-coordinate copula transform and beta partition as instrument 3.
   Fit eight PCA components separately within each beta module on past states
   only and apply those fixed bases to past and future. Score the 16-dimensional
   state using

       I(X;X') - I(A;A') - I(B;B')
       + min{I(A;A'), I(A;B'), I(B;A'), I(B;B')}.

   Eight components per module are fixed before support results. Its joint
   past-future covariance has 32 dimensions, so even the smallest registered
   support has at least two observations per joint dimension.

The prior `raw100_full_revised` is retained as a diagnostic control on the
same copula data and beta partition. It is never eligible for selection or a
later outcome pilot.

Ordinary Gaussian MI uses maximum-likelihood covariance (`ddof=0`) plus the
already ported relative ridge
`max(1e-8, 1e-6 * trace(covariance) / dimension)`. PCA is the stabilization;
no ridge, shrinkage, component count, or eigenvalue threshold may be changed
after support results.

## Required diagnostics

Every estimator, trajectory, repeat, support, and mode retains:

- active state dimensions and dimensions in each partition;
- whole joint past-future dimension;
- effective transition-pair count;
- unregularized covariance rank for every information channel;
- ridge value for every covariance and the complete ridge rule;
- all component MI values, redundancy-channel identity, and ordinary score;
- samples per joint dimension; and
- transform, partition, PCA-basis, pair-index, and trace digests.

Reports show score level, joint-rank fraction, and ordering stability as
support changes. Normalization or standardization may be added only as a
clearly named display column; it may not replace ordinary scores.

## Synthetic gates

Before development trajectories exist, tests must cover:

- independent-block autoregressive null: stabilized revised score near zero;
- cross-coupled VAR: positive stabilized revised score and greater than null;
- redundant-copy dynamics: no substantial positive integration score;
- a Gaussian suppressor/synergistic fixture: joint prediction exceeds both
  part channels and stabilized revised score is positive;
- simultaneous molecule/beta label permutation for the all-coordinate public,
  raw-full, and PCA8 instruments. The legacy drop-last typeset/macro readings
  are measured but are not required to be invariant because changing which
  molecule is "last" changes their declared representation;
- exact pair duplication: ordinary stabilized score changes by at most 2%;
- increasing samples from a fixed coupled process: the stabilized 128-, 256-,
  and 512-pair scores keep sign and remain within 25% of the 512-pair level;
- explicit-pair handling, PCA past-only fitting, covariance-rank reporting,
  and source-label isolation; and
- exact replay of the existing macro WMS on a contiguous window.

Failure of a synthetic direction or invariance gate locks development
generation until the protocol is amended and newly registered. Amendments are
visible failures, never silent tuning.

## Frozen numerical stability gate

Only `pca8_full_revised` can pass. Let each trajectory's 512-pair ordinary
score be its reference. For each smaller support:

1. at least 80% of all trajectory-pair orderings across the 12 repeated nested
   subsamples must agree with the corresponding 512-pair ordering;
2. the Spearman correlation between median repeated scores and the six
   512-pair scores must be at least 0.70;
3. the median normalized drift
   `abs(score_n-score_512)/(abs(score_512)+median(abs(score_512)))` across
   trajectories and repeats must be at most 0.25; and
4. at least 80% of the end-anchored trajectory-pair orderings must agree with
   the 512-pair end-anchored ordering.

All four conditions must hold at every registered support and all synthetic
gates must pass. Missing/ineligible trajectories fail the gate if fewer than
five of six remain.

For every instrument, report the sign-flip rate of each pairwise trajectory
contrast relative to 512 support. An instrument is explicitly called unstable
when any support has more than 20% contrast sign flips. These descriptive
classifications do not select an alternative candidate.

## Stop rule

If the PCA8 candidate fails, retain the numerical null and stop. A different
stabilization rule requires a new prospective protocol; it cannot be chosen by
looking for whichever candidate behaved best.

If PCA8 passes, freeze its exact source hash and prepare—but do not
automatically launch—a new 12-seed outcome protocol. The label gate in
`LABEL_CONTRACT_STATUS.md` still controls whether that study can be described
as replication, provisional reconstruction-label evidence, or remains locked.
