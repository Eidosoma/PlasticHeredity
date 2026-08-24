# S12I Full Results: Aggregate-Support Waiver Sensitivity Analysis

## Top summary

- **Research step ID:** `E01-S12I-AGGREGATE-SUPPORT-WAIVER-SENSITIVITY-v1.0.0` (S12I)
- **Completion status:** `STOPPED_FAIL_CLOSED_AFTER_ALL_96_SOURCE_TASKS`; no valid candidate association, drift statistic, or all-three sensitivity adjudication was produced, and no downstream step began.
- **Artifacts written:** 53 status-bearing files under `/artifacts/research_steps/S12I/`, including the pushed preregistration/method lock, single-gate waiver contract, source/input/prior/cache audits, 96-task collated label/full/prefix/partition/diagnostic/replay outputs, suppressed statistical tables, six stop-state figures, execution/runtime/schema/regeneration/failure/status evidence, artifact hashes, and this canonical report.
- **Validation result:** `FAIL_CLOSED`. All 96 fresh source tasks, source/replay/finite-coverage/suffix/input/seed/immutability/runtime/storage gates passed, but the frozen post-collation candidate-statistics stage raised `KeyError: 'rawObservationIndex'` before returning any scientific statistic.
- **Outcome classification:** `S12I_VALIDATION_FAILED_CLOSED` (constraining/contradictory operational evidence); scientific associations are `NOT_EVALUATED`.
- **Caveats or blockers:** S12H's `aggregateSupportCompatible` gate remains failed; candidate 1 remains a human-waived near-envelope, non-confirmed derivative. Raw source outputs were materialized but are not promoted to scientific evidence. No post-access code repair or rerun is permitted within S12I.
- **Recommended next action:** Return for mandatory human review. Keep S13, prediction, MLP, interventions, estimator repair, and scale-up blocked; authorize no automatic S12I repair.

## Lay summary

The human authorized one transparent sensitivity test after candidate 1 narrowly missed the paper-axis support rule. The source calculations themselves completed for every one of the 96 existing trajectories, and all replay, future-suffix, and finite-value checks passed. The next frozen analysis function nevertheless expected a column name that the frozen prefix-output schema does not contain. Because this happened after label and information-theory values had been computed, changing the adapter would be an undocumented post-outcome repair. S12I therefore stopped and reports no positive, negative, or candidate-sensitive scientific conclusion.

## Frozen question and decision boundary

S12I asked whether the exact S12G label/source-emergence conclusion was consistent across two S12FR-confirmed candidates and one explicitly non-confirmed, near-envelope C1 sensitivity case. The only human waiver was S12H's failed `aggregateSupportCompatible` gate. That gate was never relabeled as passed, and candidate 1 was never called upstream confirmed.

The frozen rules required fresh computation from raw S12FR trajectories, exact source and suffix replay, at least 80% finite coverage, unchanged candidate-specific inference, and the same all-three unanimity rule as S12G. They prohibited any favorable candidate selection, S12G payload reuse, new simulation, and undocumented post-outcome method change. Consequently the aggregation exception is terminal for this version.

## Inputs and fixed candidate identities

Exactly 96 S12FR confirmation trajectories were used, 32 per candidate. All 32 catalytic-matrix and initial-state identities were shared across candidates and therefore eligible for paired diagnostics had the statistical stage completed.

| Candidate | Exposure | Daughter rule | Overshoot | Clock | Evidence status |
| --- | ---: | --- | --- | --- | --- |
| `S12F-CANDIDATE-01` | 0.5081160391061118 | First daughter | Retain | C1 selected daughter retained | `HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED` |
| `S12F-CANDIDATE-02` | 0.6031526490073492 | First daughter | Trim new entrants | C1 selected daughter retained | `S12FR_UPSTREAM_CONFIRMED` |
| `S12F-CANDIDATE-03` | 0.5613315384859516 | Random nonempty daughter | Trim new entrants | C1 selected daughter retained | `S12FR_UPSTREAM_CONFIRMED` |

No catalytic matrix, initial state, GARD trajectory, exposure, clock, candidate, or ranking weight was generated, searched, removed, or updated. Every raw-cache hash and every S12FR repaired-replay flag passed.

Pinned source commits were:

- IIGR `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`.
- PhiRL `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`.
- Historical GARD `86dff6320d5ae91b4e831471079ff46749b14df9` as provenance context.

Scientific execution used only the audited safe lattice JSON with SHA-256 `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`. S12C source equivalence remained 14/14 and S12D source-emergence identity remained 40/40.

## Detailed methods

The primary label was `HISTORICAL_H090_REPLICATOR`; `PAST_ONLY_COSINE_REPLICATOR` was the secondary causal companion. Integer counts received the frozen additive-0.5 closure, full 100-component centered log-ratio transform, and removal of original component 100.

The primary metric was IIGR source-defined emergence, exactly synergy plus the two downward-causation atoms. PhiRL emergence was the regularization robustness companion. Corrected `local_phi_r` was retained only as a frozen comparator.

Complete-trajectory fits were labeled `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL`. At each post-fission endpoint having at least 256 prior C1 transitions, the same source implementation was independently refit from the beginning of that prefix and only the endpoint value was retained. Every full and eligible-prefix result was replayed. Every prefix received a byte-exact structural future-suffix check; first, middle, and last eligible endpoints per trajectory and implementation were also recomputed for suffix deletion, deterministic shuffle, and domain-separated replacement.

Had operational aggregation completed, candidate-specific current/next-generation associations, replicator-minus-drift differences, temporal dependence, three-sigma and robust spikes, metric identity, full-versus-prefix future dependence, partition stability, and paired cross-candidate comparisons would have used the unchanged S12G 4,096-replicate bootstrap, circular-shift, and block-aware rules. No such summary is reported because the first candidate-statistics call failed.

## Commands and dependencies

The method was tested, committed, and pushed before scientific access:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/e01/test_s12i_aggregate_support_waiver_sensitivity.py \
  tests/e01/test_s12g_frozen_timebase_ensemble.py
python -m ruff check \
  src/e01_aggregate_support_waiver_sensitivity \
  scripts/e01/freeze_s12i_preregistration.py \
  scripts/e01/run_s12i_aggregate_support_waiver_sensitivity.py \
  tests/e01/test_s12i_aggregate_support_waiver_sensitivity.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src \
  python scripts/e01/freeze_s12i_preregistration.py \
  --design-commit 2af64e849c7c1d8e486191cfbfe0e24a4d22d2d1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
ARTIFACTS_DIR=/artifacts PYTHONPATH=src \
  python scripts/e01/run_s12i_aggregate_support_waiver_sensitivity.py --workers 6
```

Focused tests passed 11/11 and Ruff passed. No dependency was installed. CPU float64 was authoritative, six source workers were used, each numerical-library thread count was one, and GPU use was zero.

## Execution results

The three-trajectory benchmark passed. It projected 19.093 CPU-hours and 3.183 wall-hours, within the 250 CPU-hour and 72 wall-hour ceilings. All 96 fresh tasks then completed:

| Output/evidence | Count | Validation |
| --- | ---: | --- |
| Fresh source tasks | 96/96 | Passed; zero task failure rows |
| Label rows | 19,200 | Complete: 96 × 100 generations × 2 labels |
| Full local rows | 168,880 | Status-bearing; all replayed |
| Prefix endpoint rows | 19,200 | Complete: 96 × 100 endpoints × 2 implementations |
| Eligible prefix rows | 13,340 | All replayed; 100% finite emergence |
| Partition fits | 13,532 | Status-bearing |
| Source diagnostic rows | 13,532 | Status-bearing |
| Suffix validation rows | 40,020 | All structural checks passed |
| Executed suffix sentinels | 1,728/1,728 | All deletion/shuffle/replacement checks passed |
| Seed rows / unique stream IDs | 27,064/27,064 | No duplicate stream identity |

Full and eligible-prefix emergence finite coverage was 1.0 for each of the six candidate/implementation branches. Every full replay and every eligible-prefix replay passed. Thus the source operational gate itself passed.

The run then failed before candidate statistics were returned. Static inspection localizes the mismatch: `trajectory_association_summary` and `replicator_drift_summary` sort their input by `rawObservationIndex`, but the frozen prefix table exposes the endpoint field as `endpointRawObservationIndex`. Passing that table unchanged therefore raises `KeyError: 'rawObservationIndex'`. Altering the adapter, aliasing the column, or rerunning would be a post-access method modification, so none was attempted.

The association, drift, temporal, spike, metric-identity, future-dependence, cross-candidate, and adjudication tables are schema-valid and empty. The six figures are explicit stop-state placeholders. No cached local value was promoted into a scientific conclusion.

## Validation

- Exact waiver scope passed: the only S12H failure was and remains `aggregateSupportCompatible`; all 13 nonwaived gates remain true.
- Source snapshot, safe-lattice, S12C-equivalence, and S12D-identity gates passed.
- All 96 raw trajectory hashes and replay records passed.
- All 96 task-level full/prefix replay and suffix indicators passed.
- Minimum full and eligible-prefix finite coverage was 1.0, above the frozen 0.80 threshold.
- All 20 required table schemas passed; statistical tables are intentionally empty and status-bearing.
- End-of-step immutability passed for 568 prior artifact files, 96 locked raw caches, and 950 forbidden S12G cache files, with zero changed files.
- Artifact completeness passed with no required file missing. Retained artifacts were about 13.5 MB and fresh disposable cache about 21 MB, far below 30 GiB.
- Approximate wall time was 4.236 hours; summed worker CPU time was 24.203 hours. Runtime ceilings passed.
- No new GARD trajectory, S12G payload read/reuse, prediction, MLP, intervention, estimator repair, scale-up, or S13 access occurred.

The overall validation status is nevertheless false because scientific aggregation failed and the preregistered no-repair rule suppressed adjudication.

## Provenance and hashes

The pre-scientific design commit and pushed remote identity were `2af64e849c7c1d8e486191cfbfe0e24a4d22d2d1`. `preregistration_record.json`, `method_lock.json`, and `implementation_lock.json` bind the exact configuration, runner, reused S12G backend, schemas, tests, source wrappers, and branch identity. `immutable_prior_baseline.json` and `immutable_prior_validation.json` preserve before/after hashes. `artifact_manifest.json` records final sizes and SHA-256 identities for every retained S12I artifact. Large disposable per-task payloads remain under `/cache/e01_s12i/source_results` and are represented by their compact completion, collated, runtime, and validation evidence rather than copied into the artifact directory.

## Caveats, blockers, and limitations

- This is a post-result human waiver. It cannot make candidate 1 or the three-candidate set upstream confirmed.
- S12H's failed aggregate-support gate remains false and its classification is unchanged.
- The materialized local values are source-informed outputs, not the unavailable author implementation or paper-primary evidence.
- Full fits are retrospective. Prefix fits passed their technical prospective checks, but no association statistic is valid because aggregation failed.
- The raw trajectories had already served S12FR time-base confirmation; they are not new GARD holdouts.
- Exact replay is bounded to the pinned wrappers, runtime, CPU-float64 policy, and platform.
- S12F, S12FR, S12G, S12H, and every earlier negative, failed, or suppressed result remain unchanged.

## Recommended next action

Return for mandatory human review. Treat the S12I scientific question as `NOT_EVALUATED`, preserve the 96 fresh task outputs as failed operational evidence, and authorize no automatic repair. S13 remains `BLOCKED_PENDING_S12I_HUMAN_REVIEW`.
