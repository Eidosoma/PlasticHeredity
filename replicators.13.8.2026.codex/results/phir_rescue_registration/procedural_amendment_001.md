# Chapter 5 Phi-r rescue procedural amendment 001

Date: 2026-08-19

## Scope

This amendment repairs only two bookkeeping defects in the R0
replay-and-remeasurement runner. It changes no scientific question, archived
trajectory, catalytic matrix, composition, intervention, simulator contract,
seed, preprocessing, partition rule, Phi-r estimator, null generator, null
draw, window, selection rule, inference rule, CPU boundary, R1 barrier, or
claim boundary.

The original R0 registration was
`d4e26c19d5a39b80c64a048e195ede50d21281092e863cd7e4dddfaa704762b8`.
The detached attempt completed all deterministic NuMIT reference libraries and
all 24 generation matrices, then stopped at its mandatory archived-score gate.
No R0 estimator contrast, arm ordering, candidate selection, or scientific
effect was analyzed or reported before this amendment was written.

## Failure record

- Failed status SHA-256:
  `feaf2bf2dc998c64092de3311600b3eee2d94ca98188b2616d86a05378ba3070`.
- Failed log SHA-256:
  `5378cfd66781f4413aee1379bda0e8fe44d909c7bd48806ffd5b610b5fcdad18`.
- Original registration JSON SHA-256:
  `276410e13273522ee3ec28318e2f0bc35b5e0178ba27bd743360d65ba4ed0185`.
- Failed archive-audit SHA-256:
  `621bda1784509ea9d8021e9492c16221673a5020df04b510c3695d9d5cab6182`.
- NuMIT libraries: 108 of 108 complete.
- Generated matrices: 24 of 24 complete.
- Replay matrices: 0 of 24 generated.
- Scientific result directory: absent.

The archived audit established that all 288 selected lineages had exact keys,
completed horizons, extinction flags, fission-record digests, final RNG-state
digests, final compositions, and inherited fractions. The maximum inherited-
fraction difference was floating-point roundoff (`1.11e-16`). The gate failed
because newly calculated physical-partition copula diagnostics used generic
field names such as `causation`, `synergy`, `atom_*`, and `partition_*`. When
prefixed by representation, those names collided with the legacy replay fields
and replaced them in the temporary row before comparison. The new revised
candidate fields themselves did not collide.

A read-only failure diagnosis also found that checkpoint classes created by a
`python -m` process were recorded under `__main__`, preventing a later imported
process from loading them without a compatibility unpickler. No checkpoint was
reused or analyzed through that workaround.

## Repair

1. Every new physical-partition macro diagnostic is placed under an explicit
   `copula_*` namespace, including its atoms and partition metadata. The
   `full_block_*`, `partition_null_*`, and `numit_*` candidate names remain
   unchanged because they were already disjoint. A regression test requires
   the complete new-score key set to be disjoint from the legacy replay key
   set.
2. When executed with `python -m`, the runner registers its dataclasses under
   the canonical `plastic_heredity.phir_rescue` module name. Checkpoints must
   load in a new interpreter and retain their canonical scientific digest.

The failed validation, registration, smoke, work tree, and log are retained in
paths suffixed `_pre_amendment_001` or `_failed_pre_amendment_001`. The 108
NuMIT libraries may be hard-linked into the new work tree only after their
complete SHA-256 manifest is written and their embedded transition counts and
system counts pass the amended runner's normal validation. They are pure,
deterministic functions of the already sealed null seeds and unchanged
estimator, so recomputing them would produce the same bytes and add no
scientific protection. The 24 collided lineage checkpoints are not reused.

Validation, registration, and the non-scientific smoke are regenerated under
the amended source seal before the same R0 replay is restarted detached. R1
and every 48-matrix run remain locked.
