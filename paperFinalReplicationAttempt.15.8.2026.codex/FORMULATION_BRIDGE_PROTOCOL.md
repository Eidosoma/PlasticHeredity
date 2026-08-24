# Arrivals formulation bridge: frozen 12-seed pilot

Status: prospective protocol. The original white-room replication and its
negative conclusions remain unchanged. A scientific pilot may run only after
this document, the bridge implementation, and its tests have been hashed into
a machine-readable registration.

## Question and boundary

The original reconstruction in this repository used a two-macro-variable
local Gaussian whole-minus-sum (WMS) estimator and did not reproduce the
paper's main association, prediction, or intervention claims. Later work in
the separate plastic-heredity project produced a different information
instrument that retains the molecular coordinates and subtracts only each
block's information about its own future. This bridge asks whether estimator
formulation can account for part of the failed observational Phi-r result.

This is a new formulation study, not a rescue, replacement, or continuation of
the original confirmatory replication. `REPLICATION_REPORT.md` is immutable
historical evidence. In particular, this study does not resolve the detector
discrepancy: the reconstruction's 0.95 cosine detector yielded 16.7% control
replication where the paper reported 88%. All outcomes below retain that
existing detector and must be described as reconstruction-label outcomes.

No Phi-guided intervention is authorized. A passing pilot is a reason to
design a larger untouched observational validation, not automatic permission
to intervene. The authors' promised code, if released, starts a separate
implementation-parity audit.

## Identical fresh trajectories

The pilot consists of exactly 12 untreated GARD trajectories with the existing
paper-scale `GardConfig` and these fixed seeds:

`26082101, 26082102, 26082103, 26082104, 26082105, 26082106, 26082107,
26082108, 26082109, 26082110, 26082111, 26082112`.

Each seed is simulated once and checkpointed once. Every estimator reads the
same count, beta, phase, generation, join, leave, and fission arrays. There is
no tuning, seed replacement, trajectory exclusion, detector sweep, or
paper-scale rerun. A failed or constant-label trajectory remains in the
manifest and is reported as non-evaluable where required.

## Frozen four-cell estimator bridge

This is a four-cell factorial-style formulation bridge, not an orthogonal
2-by-2 causal factorial: preprocessing, partition choice, dimensionality, and
algebra change together in some contrasts. Results therefore compare frozen
instruments; they do not identify a unique causal contribution of any one
design choice.

1. `macro_wms` is the original implementation unchanged: pseudocount 0.5,
   CLR with the final coordinate removed, lag-one data-derived spectral split,
   averaging within the two halves, and local
   `I(AB;A'B') - I(A;A'B') - I(B;A'B')`.
2. `macro_mmi` uses the identical original transform, partition, fitted
   Gaussian model, and transitions. It adds the local contribution from the
   one part-to-whole-future channel whose trajectory mean is smaller, exactly
   matching the repository's existing `mmi_synergy` sensitivity.
3. `public_nine_atom` applies the frozen public-PhiRL reconstruction used by
   the plastic-heredity project: all CLR coordinates, inactive-coordinate
   filtering at standard deviation `1e-8`, average-rank Gaussian copula
   transform, and one arm-independent physical Fiedler split of
   `0.5*(log1p(beta)+log1p(beta.T))`. The two blocks are macro-averaged and all
   16 local PhiID atoms are calculated; the registered score is the published
   nine-atom revised-Phi-r sum.
4. `full_revised` uses the same all-coordinate copula transform and the same
   fixed physical partition as cell 3, without macro-averaging. It calculates
   local whole, A-to-A, A-to-B, B-to-A, and B-to-B Gaussian information. The
   score is

       I(X;X') - I(A;A') - I(B;B')
       + min{I(A;A'), I(A;B'), I(B;A'), I(B;B')}.

   The minimum is chosen once from the four trajectory-mean channels and that
   channel's local series is added at every transition. Local Gaussian
   surprisals are additively centered so their sample means equal the
   registered ridge-regularized plug-in Gaussian entropies. Consequently the
   local score mean must equal the global full-block formula to numerical
   tolerance. Pointwise minimization is forbidden.

The physical partition is computed from beta alone and is fixed across the
public and full-dimensional cells. Both use every active molecular coordinate.
The original cells retain their original data-derived partition so that they
remain exact replay controls.

## Synthetic gates before scientific generation

The scientific command is locked until the test suite verifies:

- exact replay parity for original macro WMS and macro MMI;
- all four full-block channel identities and equality of local-score mean to
  the global full-block formula;
- exact nine-atom selection from the complete 16-atom lattice;
- invariance of both PX scores to a simultaneous molecule-label permutation;
- partition-swap invariance of the full-block scalar;
- near-zero full-block score for independent autoregressive blocks and a
  positive score for a cross-coupled fixture; and
- prefix isolation: changing observations after the 25% boundary cannot alter
  any early estimator input.

Registration hashes the protocol, relevant original implementation files,
bridge source, and bridge tests. The pilot refuses source-hash drift.

## Association estimand

For retrospective association only, each estimator is fitted once to the
complete trajectory and yields one local score per adjacent molecular
transition. Scores are aligned to the existing detector label at the future
observation. For every seed and estimator retain:

- Spearman rho and its descriptive p-value;
- mean score under reconstructed replicator and drift labels;
- the one-sided Mann-Whitney p-value for higher scores under replication; and
- score/component identities and finite-value diagnostics.

Across seeds report the evaluable and positive counts, median and mean rho,
positive/significant count, and count with a higher replication-state mean.
No pooled-transition p-value is allowed because the trajectory is the sampling
unit.

## Leakage-free early prediction

The early boundary is `floor(0.25 * molecular_observations)`. Each estimator is
refitted using only observations strictly before that boundary; future
observations may not affect its transform, partition (except beta-only physical
partitions), covariance, redundancy choice, or scores. Its early score series
is linearly resampled to 32 values.

The target is the existing full-trajectory detector label from the boundary
onward, nearest-neighbor resampled to 96 binary values. Prediction uses
12-fold leave-one-seed-out evaluation. In each fold and estimator, the existing
standardized one-hidden-layer MLP contract is frozen at 64 ReLU units,
`alpha=1e-3`, Adam, early stopping, at most 500 iterations, and a fold-specific
but estimator-shared random seed. The comparator is the existing training-only
global majority dummy. Accuracy is retained per held-out seed; no hyperparameter
or threshold may change after results are visible.

## Descriptive pilot screen

Because 12 trajectories are a pilot, the screen is descriptive rather than a
confirmatory significance gate. An instrument is called `pilot_viable` only if:

1. at least 10 of 12 within-run correlations are evaluable;
2. at least `ceil(0.73 * evaluable_runs)` correlations are positive and the
   median evaluable rho is positive; and
3. its mean leave-one-seed-out accuracy exceeds the paired majority dummy and
   it wins on at least 8 of 12 held-out seeds.

All four instruments and all failed criteria remain visible. The 0.73 screen
mirrors the paper's reported 73/100 positive-correlation count; it is not a
fitted target or a claim of replication. A viable result permits only a frozen,
larger observational validation proposal. A null pilot preserves the original
negative result and stops this branch pending author code or a newly justified
instrument.

## Provenance

The PX port is based on these local, non-author reference snapshots:

- `plastic_heredity/phir_extension_common.py`, SHA-256
  `550f871092f2b05079293db3e75c5a1337f1a097665fbf845acf7f1073572c7a`;
- `plastic_heredity/phir_rescue_instruments.py`, SHA-256
  `55b1b8cb328a25ca497330ef8770f758c2807364cf22408200859663e252c62c`;
- `plastic_heredity/phir_instruments.py`, SHA-256
  `69132410f668a2d1c4767a75bf9f4e9c25a9182d12be887c15e75bd6e4f29205`.

These snapshots are from the separate
`replicators.13.8.2026.codex` project. They are not the paper authors' code.
Exact runtime package versions, source hashes, trace hashes, configuration,
registration identifier, and all result tables are persisted.
