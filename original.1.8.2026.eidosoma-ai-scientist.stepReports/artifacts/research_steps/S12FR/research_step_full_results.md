# S12FR full results — Exact replay comparator repair

## Top summary

- **Research step ID:** `E01-S12FR-EXACT-REPLAY-COMPARATOR-REPAIR-v1.0.0`.
- **Completion status:** `COMPLETED_AT_ORIGINAL_S12F_HUMAN_REVIEW_BOUNDARY`; stopped at the original S12F human-review boundary.
- **Artifacts written:** complete comparator contract, pair/field/RNG/trace diagnostics, confirmation/firewall evidence, conditional ABC and candidate outputs, failure/runtime/scope/replay/provenance/hash manifests, status JSON, and this canonical report under `/artifacts/research_steps/S12FR/`.
- **Validation result:** `PASS_REPAIR_AND_UNTOUCHED_CONFIRMATION`.
- **Outcome classification:** `EXACT_REPLAY_COMPARATOR_REPAIR_CONFIRMED`; conditional time-base classification `NONIDENTIFIABLE_TIMEBASE_ENSEMBLE`.
- **Caveats or blockers:** S12F and its SIMULATOR_IDENTIFICATION_FAILED classification remain immutable. The comparator recognizes only paired zero-update exposure-extrema NaNs; it supplies no finite tolerance. A repaired comparator does not erase prior negative scientific evidence or establish author-code identity. Labels, emergence, local Phi-r, prediction, interventions, S12G, and S13 were not executed.
- **Lay summary:** The repair asked whether S12F's repeatability stop was only a representation problem. The old equality rule treated paired undefined exposure values as unequal. The new rule was allowed to recognize only matching NaNs caused by the same zero-update generation, while retaining exact equality everywhere else. The conditional time-base result is reported separately and does not rewrite S12F.
- **Recommended next action:** Mandatory human review. Keep S12G and S13 blocked; do not start labels, emergence, prediction, interventions, or any downstream step automatically.

## Frozen question and scope

S12FR is one additive operational repair. S01–S12F, including S12F's `SIMULATOR_IDENTIFICATION_FAILED` result and suppressed outputs, were hash-baselined and remained unchanged. The simulator, RNG derivation, roots, particles, exposure families, clocks, targets, summary vector, distance, acceptance gates, adaptive trigger, candidate limit, and 32-matrix confirmation rule were not modified. Previously suppressed S12F distances were never opened.

## Methods

The preregistered comparator uses exact type, shape, sequence, and value equality; finite floats require identical IEEE-754 binary64 bits. Its sole tagged normalization is a paired NaN at `GenerationSummary.maximum_exposure` or `minimum_exposure` when both matching summaries have `update_count == 0`. One-sided NaNs, other NaNs, infinities, finite tolerances, coercion, sequence reordering, or changed RNG consumption fail.

Each audited pair ran the unchanged simulator twice through a recording RNG delegate and once uninstrumented. The delegate recorded exact seed identities, initial/final bit-generator-state hashes, ordered method calls, argument/result hashes, finite/nonfinite counts, and complete integer state, Poisson join/loss, trim, fission, daughter, and stopping sequences. Compact complete canonical traces are retained under `/cache/e01_s12fr/replay_traces`; collectible Parquet manifests record every path, size, and SHA-256. When a pair passed every exact gate one canonical trace represented both identical sides; divergent sides would have been retained separately.

The original campaign re-executed the exact 256 fixed-family round-1 particles on eight original matrices each. Conditional confirmation used all 16 frozen benchmark configurations and 256 untouched particles on eight matrices under disjoint preregistered roots. Fresh ABC distances were permitted only after those gates passed unanimously. Any scientific continuation reused the original S12F implementation and rules.

## Inputs and provenance

- S12F source/method commit: `cf5b27b370a2d8d12e6867034d6ec8f4f96b3fc7`.
- S12FR preregistration commit: `a9dbb36b2485b50b3f91c6f3646fcef93ecf5404`.
- Comparator: `S12FR_SCHEMA_CAUSAL_EXACT_COMPARATOR_v1.0.0`; RNG audit: `E01-S12FR-RNG-SEQUENCE-AUDIT-v1.0.0`.
- Original paper, historical GARD, IIGR, PhiRL, safe lattice, S12F artifacts/caches, plans, and manifests are pinned in `source_input_snapshot_manifest.json`, `immutable_prior_baseline.json`, `s12f_cache_baseline.json`, and `s12f_suppressed_input_manifest.json`.
- Prior immutability: 473 files checked, pass `True`. S12F cache: 1 files checked, pass `True`.

## Comparator diagnosis and confirmation results

- Original pairs: 2048/2,048.
- Old comparator failures recovered: 323.
- Permitted paired zero-update NaN field instances: 22040.
- Original repaired pair gates: 2048/2,048.
- Original discrete/finite/forbidden-nonfinite/RNG divergences: 0/0/0/0.
- Benchmark repaired replay: 16/16.
- Untouched repaired replay: 2048/2,048.
- Repair seed/matrix firewall: `True`.

Every old-comparator failure was required to contain at least one permitted tagged NaN and no other left-versus-replay difference. Instrumented/uninstrumented parity and exact trace digests were independent gates.

## Conditional fresh ABC results

| family                |   round |   particlesEvaluated |   particlesRetained |   trajectoryPairs |   epsilonMedian |   minimumDistance |   envelopePassCount |   wallSeconds |
|:----------------------|--------:|---------------------:|--------------------:|------------------:|----------------:|------------------:|--------------------:|--------------:|
| FIXED_COMMON_EXPOSURE |       1 |                  256 |                 128 |              2048 |       10.4819   |             0.078 |                  15 |      337.886  |
| FIXED_COMMON_EXPOSURE |       2 |                  128 |                  64 |              1024 |        3.01407  |             0.087 |                  11 |       92.0981 |
| FIXED_COMMON_EXPOSURE |       3 |                   64 |                  64 |               512 |        0.994875 |             0.1   |                  12 |       45.9384 |

No old suppressed distance was reused. The table contains only fresh post-confirmation calculations. The conditional adaptive family appears only if the unchanged fixed-family final envelope contained no accepted particle.

## Posterior-predictive time-base confirmation

| candidateId       | exposureFamily        |        h |   c |   hMax | daughterRule    | overshootRule             | clockId                       |   posteriorMass |   completedLineages |   q05TPhi |   medianTPhi |   q95TPhi |   maximumTPhi |   sampleEndpointsInsideQ05Q95 | aggregateCompatible   |   medianPostFissionMass |   q95Overshoot |   fractionMaxsteps |   confirmationDistance | confirmationGatePassed   | gateReason   |
|:------------------|:----------------------|---------:|----:|-------:|:----------------|:--------------------------|:------------------------------|----------------:|--------------------:|----------:|-------------:|----------:|--------------:|------------------------------:|:----------------------|------------------------:|---------------:|-------------------:|-----------------------:|:-------------------------|:-------------|
| S12F-CANDIDATE-01 | FIXED_COMMON_EXPOSURE | 0.508116 | nan |    nan | FIRST_DAUGHTER  | RETAIN_OVERSHOOT          | C0_BATCH_UPDATES_ONLY         |       0.0452914 |                  32 |    431.65 |        812   |   1182.95 |          1281 |                             3 | True                  |                      42 |        29.25   |                  0 |                0.36475 | True                     | PASS         |
| S12F-CANDIDATE-02 | FIXED_COMMON_EXPOSURE | 0.603153 | nan |    nan | FIRST_DAUGHTER  | TRIM_NEW_ENTRANTS_TO_NMAX | C1_SELECTED_DAUGHTER_RETAINED |       0.0263847 |                  32 |    552.55 |        889.5 |   1137.95 |          1291 |                             3 | True                  |                      40 |        32.9225 |                  0 |                0.18975 | True                     | PASS         |
| S12F-CANDIDATE-03 | FIXED_COMMON_EXPOSURE | 0.561332 | nan |    nan | RANDOM_NONEMPTY | TRIM_NEW_ENTRANTS_TO_NMAX | C1_SELECTED_DAUGHTER_RETAINED |       0.0260782 |                  32 |    543.85 |        872   |   1176.6  |          1341 |                             3 | True                  |                      40 |        29.4275 |                  0 |                0.383   | True                     | PASS         |

Confirmed candidate count: **3**. The resulting lock contains only update kernel, exposure, clock, overshoot, daughter, state/indexing, seeds, upstream fingerprints, and trajectory hashes. It contains no label or information-theory output.

## Validation

- Frozen contract committed and pushed before any rerun: **PASS**.
- Original pair identity/cardinality and complete diagnostics: `True`.
- Narrow comparator cause rule; zero finite/discrete/RNG divergence: recorded in pair summaries and field tables.
- 16/16 benchmark and untouched 2,048/2,048 confirmation: `True`.
- Development/confirmation seed and matrix firewall: `True`.
- Exact replay, trace hashes, instrumentation parity, prior/cache immutability, scope, runtime/storage, schemas, manifests, and required outputs: recorded in `regeneration_validation.json`.
- Runtime: 2.999 worker CPU-hours; 0.521 summed phase wall-hours; CPU float64 authoritative; GPU unused.

## Commands

```bash
PYTHONPATH=src python -m pytest -q tests/e01/test_s12fr_replay_repair.py
ruff check src/e01_replay_repair scripts/e01/freeze_s12fr_preregistration.py scripts/e01/run_s12fr_replay_repair.py tests/e01/test_s12fr_replay_repair.py
python scripts/e01/freeze_s12fr_preregistration.py
git commit -m "Preregister S12FR exact replay comparator repair"
git push origin eidosoma/groups/42
python scripts/e01/freeze_s12fr_preregistration.py --record-commit
python scripts/e01/run_s12fr_replay_repair.py --stage diagnose-original
# comparator lock committed and pushed only after the original campaign passed
python scripts/e01/run_s12fr_replay_repair.py --stage confirm-repair
python scripts/e01/run_s12fr_replay_repair.py --stage resume-abc
# candidate lock committed and pushed only when candidates existed
python scripts/e01/run_s12fr_replay_repair.py --stage confirm-timebase
python scripts/e01/run_s12fr_replay_repair.py --stage finalize
```

All long simulator commands used six process workers and one BLAS/OpenMP thread per worker.

## Caveats and interpretation

Comparator success is operational evidence, not scientific validation of a time base. The only normalization is schema-causal and explicitly tagged; nevertheless, this repair was authorized after observing a global failure, so the untouched confirmation firewall is essential. Any candidate is a paper-as-data simulator-identification result, not the unavailable author implementation. S12F remains failed exactly as originally executed, and extensive prior negative or underdetermined Phi-r evidence is unchanged.

## Provenance and artifact completeness

Pair diagnostics retain every pair identity and every field difference. Trace manifests point to complete compact sequence payloads with SHA-256 identities. Seed manifests, firewalls, code/contract locks, runtime records, scope ledger, failure ledger, and artifact manifest provide the complete audit chain. No required downstream value was accessed.

## Recommended next action

Return for mandatory human review with S12G and S13 blocked. Do not automatically begin label reconstruction, causal-emergence analysis, intervention work, or another repair.
