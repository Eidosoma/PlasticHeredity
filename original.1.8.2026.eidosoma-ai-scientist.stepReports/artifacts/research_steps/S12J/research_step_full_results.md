# S12J Full Results: Aggregation Interface Repair Confirmation

## Top summary

- **Research step ID:** `E01-S12J-AGGREGATION-INTERFACE-REPAIR-CONFIRMATION-v1.0.0` (S12J)
- **Completion status:** `COMPLETED_AT_MANDATORY_S12J_HUMAN_REVIEW_BOUNDARY`; no downstream step began.
- **Artifacts written:** 39 required status-bearing report, table, validation, manifest, and figure paths under `/artifacts/research_steps/S12J/`, including the adapter audit view, frozen candidate statistics, replay evidence, and this canonical report.
- **Validation result:** `PASS`. The alias adapter passed every row/field/endpoint gate; immutable S12I source outputs passed; both complete executions of the frozen statistics were exact; schemas, hashes, provenance, runtime/storage, and S01–S12I immutability passed.
- **Outcome classification:** `SENSITIVITY_SET_WIDE_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE` (constraining/contradictory evidence within the bounded human-waived sensitivity scope).
- **Caveats or blockers:** S12I remains `S12I_VALIDATION_FAILED_CLOSED`; candidate 1 remains `HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED`; S12H's aggregate-support failure remains visible; this post-result adapter override weakens confirmatory credibility; full fits remain retrospective and source-informed.
- **Recommended next action:** Mandatory human review. Keep S13, prediction, MLP, interventions, estimator repair, report-bundle progression, and scale-up blocked. No further S12J repair is permitted.

## Lay summary

None of the three candidates met either the frozen retrospective coherent gate or the frozen prospective gate. This is non-support across the bounded source-informed sensitivity set, not proof about the unavailable author code.

The operational repair was as narrow as authorized: a copied statistical view received one column whose 19,200 values exactly duplicate the already validated endpoint index. No source model was fitted again, no trajectory was generated, no prior artifact changed, and no scientific setting or seed changed.

## Frozen question and scope

S12J asked whether S12I's already validated source outputs could pass through the unchanged statistics interface after adding only `rawObservationIndex = endpointRawObservationIndex`, and then what the unchanged candidate-specific and unanimous all-three rules conclude. This is an explicit human override of S12I's no-repair rule, limited to one separately versioned interface correction. S12I's failed version is immutable and independently interpretable.

S12J used only `/artifacts/research_steps/S12I/label_values.parquet`, `full_source_values.parquet`, `prefix_endpoint_values.parquet`, `partition_history.parquet`, `source_diagnostic_outputs.parquet`, `replay_suffix_validation.parquet`, `seed_manifest.parquet`, and `preprocessing_diagnostics.parquet`. It opened no S12G task cache and generated no GARD trajectory or source fit.

## Adapter methods and validation

The adapter created one in-memory copied field, `rawObservationIndex`, by exact assignment from `endpointRawObservationIndex`. The input table remained unchanged on disk and in memory. The derived audit view contains only frozen identities, both index fields, and a row ordinal.

- Rows checked: 19,200; endpoint matches: 19,200.
- Monotone trajectory/implementation groups: 192.
- Original fields checked independently: 26; every before/after canonical Arrow hash was identical.
- Original row-order hash: `36561c0ca9c481a62970b73620c8730a57b9870e5938533ba0bfe88b6073f4b4`; post-adapter original-field hash: `36561c0ca9c481a62970b73620c8730a57b9870e5938533ba0bfe88b6073f4b4`.
- Adapter gate result: `True`; source Parquet hash was unchanged.

## Frozen statistical methods

The code called the exact locked S12G/S12I procedures in their original order: candidate association and replicator-versus-drift summaries; temporal dependence and spike summaries; emergence-versus-corrected-local-Phi-r identity; full-versus-prefix future dependence; paired cross-candidate comparisons; and all-three adjudication. It retained 4,096 trajectory-bootstrap, circular-shift, and block-aware resamples with the original S12G seed root and derivation.

The primary branch remained IIGR source-defined emergence (synergy plus two downward-causation atoms) with `HISTORICAL_H090_REPLICATOR`. PhiRL remained a regularization robustness companion; corrected `local_phi_r` remained a comparator. Full values remain exactly `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL`; eligible prefix endpoints begin only after 256 locked-clock transitions.

## Candidate-specific results

| Candidate | Evidence status | IIGR full median rho | Full association gate | Full median rep-drift mean difference | Drift gate | Full coherent | IIGR prefix median rho | Prefix gate | Candidate classification |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- |
| S12F-CANDIDATE-01 | HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED | -0.01675 | FAIL | -0.00052828 | FAIL | FAIL | 0.021815 | FAIL | CANDIDATE_NOT_SUPPORTED |
| S12F-CANDIDATE-02 | S12FR_UPSTREAM_CONFIRMED | 0.015458 | FAIL | -2.1372e-05 | FAIL | FAIL | 0.020946 | FAIL | CANDIDATE_NOT_SUPPORTED |
| S12F-CANDIDATE-03 | S12FR_UPSTREAM_CONFIRMED | -0.0026378 | FAIL | 0.00014186 | FAIL | FAIL | 0.030518 | FAIL | CANDIDATE_NOT_SUPPORTED |

The all-three classification is `SENSITIVITY_SET_WIDE_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE`. Candidate-specific results remain primary. The S12FR ranking weights were neither interpreted as author-identity probabilities nor used in analysis.

## Temporal and spike results

| Candidate | Aggregate trend p | Runs with positive 3-sigma spike | Raw Ljung-Box p<=0.05 | Differenced Ljung-Box p<=0.05 |
| --- | ---: | ---: | ---: | ---: |
| S12F-CANDIDATE-01 | 5.3132e-11 | 30 | 16 | 32 |
| S12F-CANDIDATE-02 | 0.074965 | 32 | 12 | 32 |
| S12F-CANDIDATE-03 | 0.0074152 | 30 | 10 | 32 |

The punctuated gate remains a separate descriptive rule and does not override association or drift gates.

## Metric identity and future dependence

| Candidate | Median full-prefix Spearman | Median normalized absolute difference | Median fraction rank shifts >10 points | Median partition ARI | Median full emergence/local-Phi-r Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| S12F-CANDIDATE-01 | 0.24987 | 0.83844 | 0.69231 | 0.052375 | 0.59394 |
| S12F-CANDIDATE-02 | 0.34378 | 0.65174 | 0.65275 | 0.1076 | 0.58514 |
| S12F-CANDIDATE-03 | 0.2632 | 0.76149 | 0.69178 | 0.040207 | 0.58367 |

Completed-trajectory source values may depend on future observations. The full-versus-prefix outputs therefore remain a future-dependence audit, not prospective early-warning or causal-control evidence.

## Cross-candidate analysis

All 32 matrix and initial-state identities were shared, so the frozen pairwise label, association, drift, and partition contrasts were legitimately paired. The output contains 864 rows and no manufactured pairing or weight update.

## Commands and dependencies

```bash
PYTHONPATH=src python -m pytest -q \
  tests/e01/test_s12j_aggregation_interface_repair_confirmation.py \
  tests/e01/test_s12i_aggregate_support_waiver_sensitivity.py \
  tests/e01/test_s12g_frozen_timebase_ensemble.py
python -m ruff check \
  src/e01_aggregation_interface_repair \
  scripts/e01/freeze_s12j_preregistration.py \
  scripts/e01/run_s12j_aggregation_interface_repair_confirmation.py \
  tests/e01/test_s12j_aggregation_interface_repair_confirmation.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src \
  python scripts/e01/freeze_s12j_preregistration.py --design-commit <pushed-commit>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
ARTIFACTS_DIR=/artifacts PYTHONPATH=src \
  python scripts/e01/run_s12j_aggregation_interface_repair_confirmation.py
```

No dependency was installed. CPU float64 was authoritative; the statistics ran serially with every BLAS/OpenMP thread count fixed to one and no GPU use.

## Validation results

- Immutable input/source gate: True; all eight S12I input files retained exact hashes and row counts.
- Adapter gate: True; all 15 preregistered gates passed.
- Exact deterministic statistics replay: True; 10/10 result frames and the classification matched exactly.
- Result cardinality/gate validation: True.
- Prior immutability: True; 621 S01–S12I artifact files checked, zero changed.
- Runtime/storage: wall 0.1269 hours, process CPU 0.1268 hours, retained bytes recorded in the artifact manifest, GPU hours 0.
- S12I source evidence reused without refitting: 96/96 fresh tasks, zero task failures, full and eligible-prefix replay passed, 40,020 structural suffix checks and 1,728 executed sentinels passed, and 27,064 seed identities remained unique.

## Provenance

The pushed pre-statistics design commit, every locked code/config hash, input SHA-256 identity, adapter contract, and output hash are recorded in `preregistration_record.json`, `method_lock.json`, `input_manifest.json`, `adapter_validation.json`, `statistics_replay_validation.json`, `provenance_manifest.json`, and `artifact_manifest.json`. The original paper remains an interpretive target only; these public-source values are not the unavailable author implementation.

## Caveats, blockers, and limitations

- This is an explicitly post-result, one-repair human override. It does not retroactively make S12I pass.
- Candidate 1 is a near-envelope human-waived sensitivity case, not an upstream-confirmed paper-time-base candidate.
- A positive all-three result, if present, is exploratory consistency only; it cannot establish an upstream-confirmed ensemble or support S13.
- Full fits are retrospective and can use completed-trajectory partitions and Gaussian parameters.
- The historical-H090 label and source-defined emergence implementation are frozen reconstructions, not author-primary or paper-primary identities.
- The 96 trajectories were already used for upstream time-base confirmation and are not new GARD holdouts.
- Exact replay is bounded to the pinned Python/runtime/platform and frozen float64 implementation.
- S01–S12I, including every negative, failed, waived, future-dependent, and suppressed result, remain unchanged.

## Recommended next action

Return for mandatory human review. Do not begin S13 or any prediction, MLP, intervention, estimator repair, report-bundle progression, scale-up, or additional adapter repair. Treat `SENSITIVITY_SET_WIDE_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE` only within this bounded source-informed sensitivity scope.
