# S13 Full Results: Confirmed Time-base Baseline Held-out Scale-up

## Top summary

- **Research step ID:** `E01-S13-CONFIRMED-TIMEBASE-BASELINE-SCALEUP-v1.0.0` (S13).
- **Completion status:** `STOPPED_FAIL_CLOSED_AT_SOURCE_SCHEMA_AGGREGATION`; no candidate association statistic or two-candidate adjudication was computed.
- **Artifacts written:** Complete pre-outcome locks and compute audit; 200 trajectory identities and replay rows; 100 pairing rows; 200 source-task validations; complete cache/schema/seed/runtime/storage/provenance/immutability/failure evidence; schema-bearing suppressed downstream outputs; six stop-state figures; artifact hashes; and this canonical report.
- **Validation result:** `FAIL_CLOSED_AT_SOURCE_LABEL_PARQUET_SCHEMA_GATE_AFTER_200_OF_200_TRAJECTORY_AND_SOURCE_TASK_COMPLETIONS`. Simulation and task-level source replay/suffix gates passed, but the frozen global schema gate failed.
- **Outcome classification:** `S13_VALIDATION_FAILED_CLOSED` (constraining/contradictory operational evidence); the held-out association question is `NOT_EVALUATED`.
- **Caveats or blockers:** The mismatch concerns physical Parquet types for optional all-null label fields. An adapter or schema cast would be a repair after source outcomes existed and is forbidden. The source-informed pipeline still cannot identify the unavailable author implementation, and repeated E01 overrides remain a procedural-credibility caveat.
- **Lay summary:** The new simulations and all 200 expensive source calculations finished and replayed, but their per-trajectory label tables did not share one physical file schema. The preregistered rule required an immediate stop on any schema failure. Therefore none of the generated emergence values was turned into a correlation or scientific verdict.
- **Recommended next action:** Mandatory human review. Do not repair or rerun S13, and keep S14–S18, prediction, MLP work, interventions, estimator repair, report-bundle progression, E02, and every further scale-up blocked.

## Frozen question and scope

S13 asked whether S12J's near-zero/non-support result persists on 100 genuinely new catalytic matrices shared across the two S12FR-confirmed time-base candidates. Candidate 2 used fixed exposure 0.6031526490073492, first-daughter continuation, trimmed new entrants, and C1. Candidate 3 used fixed exposure 0.5613315384859516, random-nonempty continuation, trimmed new entrants, and C1. Candidate 1 was not run and remains `HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED`.

A positive held-out result required both confirmed candidates to pass the unchanged retrospective and prospective rules after scaling only the preregistered count gates from 32 to 100 matrices. No positive/negative scientific classification is possible because aggregation stopped before any candidate statistic.

## Inputs and provenance

- Pushed outcome-blind design commit: `0c41fd81d66a5e5b152b35e58e44089cd3e11ff9`.
- Reporting-only failure finalizer commit: `24818158d0aa68cbfe50bcbb01b8ba4d4dcd7550`; it did not cast, adapt, concatenate, or analyze source values.
- New S13 seed-root ID: `E01-S13-HELDOUT-ROOT-v1.0.0`; 111,528 executed stream identities were reconstructed and audited.
- IIGR commit `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`; PhiRL commit `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; historical GARD commit `86dff6320d5ae91b4e831471079ff46749b14df9`.
- S12C source equivalence remained 14/14 and S12D source-emergence identity remained 40/40 before execution.
- All 660 frozen S01–S12J artifact files retained their preregistered hashes.

## Detailed methods and commands

The simulator generated 100 shared catalytic matrices and 100 shared 40-distinct-singleton initial states from the new domain. Each matrix was run under both candidates for exactly 100 fissions. A second same-seed execution validated every trajectory using the S12FR exact comparator. The retained C1 state sequence recorded the initial state, every Poisson batch update, and every selected post-fission daughter.

Each retained trajectory then ran the frozen S12J label and source contract: historical H>0.9 primary label, past-only cosine secondary label, additive-0.5 closure, full CLR, removal of original component 100, IIGR source-defined emergence primary, PhiRL emergence robustness, and corrected local Phi-r comparator. Full fits were retrospective. Prefix fits were independently refit at post-fission endpoints after 256 C1 transitions, replayed exactly, and checked against suffix deletion, shuffle, and replacement. These per-task calculations were cached, not statistically aggregated.

```bash
PYTHONPATH=src python -m pytest -q tests/e01/test_s13_confirmed_timebase_scaleup.py tests/e01/test_s12g_frozen_timebase_ensemble.py tests/e01/test_s12j_aggregation_interface_repair_confirmation.py
python -m ruff check src/e01_confirmed_timebase_scaleup scripts/e01/freeze_s13_preregistration.py scripts/e01/run_s13_confirmed_timebase_scaleup.py tests/e01/test_s13_confirmed_timebase_scaleup.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s13_preregistration.py --record-commit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13_confirmed_timebase_scaleup.py --workers 6
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/finalize_s13_fail_closed.py
```

## Results

- Retained primary trajectories: 200/200; complete to 100 fissions: 200/200.
- Exact simulator replay: 200/200; shared catalytic-matrix and initial-state identities: 100/100.
- Completed source tasks: 200/200; tasks with full replay, eligible-prefix replay, suffix flags, and zero failure rows: 200/200.
- Candidate statistics, bootstrap, circular shift, block-aware inference, spike aggregation, metric-identity comparison, future-dependence summary, and final two-candidate gate: all `NOT_REACHED_SOURCE_SCHEMA_GATE`.
- No source value was used to select, eliminate, reweight, or reclassify a candidate.

## Terminal schema failure

The source worker writes one Parquet file per trajectory. Optional label columns become Arrow `null` when every value in a task is absent, but become `string` or `double` in tasks containing defined values. The completed campaign contained 2 label-schema variants. The first global concatenation used one task schema and rejected another before a combined label table or any statistic existed. `source_schema_diagnostics.csv` preserves all 200 pair identities and table-schema fingerprints; `source_schema_failure_diagnostics.json` preserves representative types and counts.

No cast, alias, adapter, row deletion, schema normalization, cache reuse, source rerun, or partial-candidate analysis was attempted. The partial `label_values.parquet` written before Arrow raised is retained and explicitly marked `PARTIAL_UNPROMOTED_SCHEMA_FAILURE`; it is not scientific evidence. Every other downstream table is schema-bearing and empty.

## Validation

- Pre-simulation compute gate: PASS, projected cumulative 144.000/250 CPU-hours and 2.000/80 GPU-hours.
- Benchmark gate: PASS; projected cumulative 149.844 CPU-hours, 11.641 wall-hours, and 0.078 GiB.
- Prior immutability: PASS, 660/660 files unchanged.
- Seed firewall: PASS, 111,528 unique executed streams, zero prior stream/material overlap, no statistics seed executed.
- Task-level replay/suffix evidence: PASS, 200/200 tasks and zero task failure rows.
- Global source schema: FAIL, 2 label schema variants; terminal stop applied.
- Artifact table schemas: present, but scientific eligibility is false and the label table is partial/unpromoted.
- Runtime/storage: PASS against hard ceilings; approximately 9.042 wall-hours, 51.817 source-worker CPU-hours, zero GPU-hours, and 0.074 GiB cache.

## Caveats, blockers, and limitations

- This is an operational validation failure, not held-out positive or negative association evidence.
- A schema cast would likely be mechanically narrow, but it is still a prohibited repair after source values existed; S13 does not test or endorse one.
- The run extended a long E01 chain with repeated negative results and human overrides. Even a future positive result would require adjudication against that history.
- Full-trajectory values are retrospective and future-dependent. Fixed-window, pre-256-transition, early-warning, prediction, and causal-control claims remain unresolved and were not tested.
- Public IIGR/PhiRL behavior is source-informed and is not the unpublished author implementation or exact paper replication.

## Provenance and artifact contract

The design/method lock, prior hash baseline and postcheck, compute ledger, benchmark, source snapshot, 100 pairing identities, 200 trajectory hashes/replay rows, 200 source-task completion records, 2,000 cache-file hashes, all 200 source schema fingerprints, executed seed manifest/firewall, failure ledger, scope record, runtime/storage reports, suppression schemas, and collectible artifact hashes preserve the complete stop state. Large raw/source caches remain under `/cache/e01_s13/` and are represented by hashes.

## Recommended next action

Return for mandatory human review. The frozen no-repair rule has fired. Do not begin S14–S18, prediction, MLP work, interventions, estimator repair, report-bundle progression, E02, or another scale-up automatically.
