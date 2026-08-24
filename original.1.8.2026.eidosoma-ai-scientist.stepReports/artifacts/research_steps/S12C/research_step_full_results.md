# S12C full results: bounded source-equivalence confirmation

## Top summary

- **Research step ID:** S12C (`E01-S12C-SOURCE-EQUIVALENCE-CONFIRMATION-v1.0.0`)
- **Completion status:** COMPLETED_CONDITIONAL_TWELVE_TRAJECTORY_AUDIT
- **Artifacts written:** `S12C_FULL_RESULTS.md`, `artifact_manifest.json`, `benchmark.json`, `classification.json`, `confirmation_access_ledger.json`, `confirmation_fixture_results.csv`, `confirmation_summary.json`, `development_attempt_history.json`, `development_fixture_results.csv`, `development_summary.json`, `failure_ledger.csv`, `figures/association_distributions.png`, `figures/final_decision_matrix.png`, `figures/full_trajectory_matched_sources.png`, `figures/full_versus_prefix_representative.png`, `figures/partition_stability.png`, `figures/spike_overlap.png`, `full_trajectory_local_values.parquet`, `future_dependence_results.csv`, `immutable_input_audit.json`, `implementation_lock.json`, `partition_history.parquet`, `prefix_endpoint_values.parquet`, `preregistration.yaml`, `preregistration_record.json`, `prospective_associations.csv`, `repair_delta.md`, `replay_validation.json`, `research_step_full_results.md`, `retrospective_associations.csv`, `runtime_manifest.json`, `safe_lattice_reference.json`, `scope_compliance.json`, `seed_firewall.json`, `source_diagnostic_outputs.parquet`, `source_equivalence_results.csv`, `source_snapshot_manifest.json`, `spike_analysis.csv`, `status.json`
- **Validation result:** PASS — 14/14 untouched confirmation rows, replay, suffix invariance, immutability, scope, runtime, storage, and artifact gates passed
- **Outcome classification:** constraining/contradictory; decision `SOURCE_FAMILY_NOT_SUPPORTED`
- **Caveats or blockers:** S12B remains a failed immutable step and S12 remains unchanged.; The public source family is source-informed, not the unpublished GARD implementation.; Full-trajectory values are retrospective and can depend on future observations.; S13 and interventions remain blocked pending human review.
- **Lay summary:** S12C tested one predeclared explanation for S12B's lone equivalence failure: a vectorized IIGR correlation calculation had changed a numerically degenerate partition. Development and an untouched confirmation suite were separated by a committed implementation lock. Only after every confirmation row passed was the frozen twelve-run source audit permitted.
- **Recommended next action:** Do not authorize S13. Close the E01 Phi-r reconstruction or move replacement-variable work to a separately preregistered E02 after human review. S13 remains blocked pending a new human decision; do not start it automatically. S13 remains blocked regardless of this result.

## Frozen question and evidence boundary

The question was whether exactly one wrapper-only correction—the pinned IIGR nested pairwise `scipy.stats.pearsonr` MI loop and assignment order—could restore full source-wrapper equivalence on disjoint development and untouched-confirmation fixtures, and conditionally permit the unchanged S12B completed-fit versus prefix audit. The public source family is classified only as `SOURCE_INFORMED_RECONSTRUCTION`; it is not the unpublished GARD implementation and has no `AUTHOR_PRIMARY`, `PAPER_PRIMARY`, or `EXACT_GARD_IMPLEMENTATION` identity.

S10, S11, S11R, S12, and failed S12B remained byte-exact. S12B's failure was neither deleted nor relabeled. No new GARD trajectory, intervention, MLP, reinforcement-learning, BioModels, gene-regulatory-network, estimator-development, or S13 work was performed.

## Inputs and provenance

- Pinned IIGR commit: `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`.
- Pinned PhiRL commit: `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; regularization ancestor `9030b598f436cd23c39a3c3fc312ff79c79fb2ad`.
- Safe lattice: immutable S12B JSON SHA-256 `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`; the raw pickle was not loaded by S12C scientific code.
- Conditional GARD inputs: only the twelve frozen S12 baseline trajectories, additive-0.5/drop-component-100 99-dimensional CLR substrate, historical labels, and S12 provenance named in `preregistration.yaml`.
- Development fixtures: 14/14 rows passed.
- Confirmation fixtures: 14/14 untouched rows passed.
- Implementation lock: `/artifacts/research_steps/S12C/implementation_lock.json`.

## Detailed methods

Seven fixture families—ordinary coupled Gaussian, coupled autoregressive, constant, exact singular duplicate, near-singular duplicate, low-rank, and partial-constant replay—were generated separately for development and confirmation at 384 observations by 10 variables. Domain-separated roots, stream identities, seed values, and payload hashes were required to have zero cross-phase intersection. Both original pinned implementations ran twice in isolated `python -I -B` processes; the wrapper ran twice in-process. The raw pickle was confined to the already audited disposable adapter behavior; scientific execution used only safe JSON.

Every one of 14 rows per phase had to match source status, retained-variable availability and identity, processed-array availability (and `1e-12` numerical check), MI availability and maximum difference at most `1e-10`, partition availability and identity up to side exchange, partition-average availability and difference at most `1e-10`, local Phi-r and diagnostic availability and differences at most `1e-9`, and exact replay. The singular fixture and all-branches-must-pass rule remained mandatory. An exception was never treated as equivalent to an eligible result.

If confirmation passed, the runner reused the frozen S12B scientific contract with only the confirmed wrapper identity and S12C output identifiers changed. Full fits used completed trajectories and were labeled retrospective; prefixes were independently refit at eligible post-fission endpoints after 256 preceding molecular transitions. The primary prospective estimand remained current-generation Spearman association, with 4,096 trajectory bootstraps and 4,096 circular shifts. Future-suffix invariance, replay, nonfinite, ineligibility, benchmark, runtime, and storage gates remained unchanged.

## Commands

```text
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/e01/freeze_s12c_preregistration.py
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/e01/run_s12c_source_equivalence_confirmation.py --phase development --workers 6
# implementation lock committed and pushed
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/e01/run_s12c_source_equivalence_confirmation.py --phase confirmation --workers 6
PYTHONPATH=src pytest -q tests/e01/test_s12c_source_equivalence_confirmation.py
```

## Results

Development: 14/14 rows passed. Confirmation: 14/14 untouched rows passed.

Benchmark and ceiling evaluation:

```json
{"cpuGatePassed": true, "fullReplayPassed": true, "futureSuffixPassed": true, "matrixIndex": 0, "observedCpuSeconds": 1444.742446296, "observedResultBytes": 650501, "observedWallSeconds": 1445.7882532959338, "passed": true, "prefixReplayPassed": true, "projection": {"artifactBytes": 9757515, "cpuHours": 6.0197601929, "formula": "observed_complete_matrix0_times_12_plus_25_percent_reserve", "wallHours": 6.024117722066391}, "storageGatePassed": true, "trajectoryId": "E01-S12-B00", "wallGatePassed": true}
```

Conditional scientific summary:

```json
{
  "classification": "SOURCE_FAMILY_NOT_SUPPORTED",
  "coverage": {
    "IIGR_CORRECTED_SOURCE": {
      "expectedFullOutputs": 125666,
      "expectedPrefixEndpointsAfterBoundary": 1147,
      "fullNonfiniteOrSuppressedFraction": 0.0003103464739865994,
      "prefixIneligibleFraction": 0.007846556233653008
    },
    "PHIRL_REGULARIZED_SOURCE": {
      "expectedFullOutputs": 125678,
      "expectedPrefixEndpointsAfterBoundary": 1147,
      "fullNonfiniteOrSuppressedFraction": 5.569789461958338e-05,
      "prefixIneligibleFraction": 0.0052310374891020054
    }
  },
  "futureSuffixInvariancePassed": true,
  "prospective": {
    "IIGR_CORRECTED_SOURCE": {
      "coverage": 0.992153443766347,
      "primaryBootstrap95": [
        -0.07438280324402756,
        0.0037002031097231928
      ],
      "primaryCircularShiftPositiveP": 0.6497437149133513,
      "primaryMedianRho": -0.019348598232742785,
      "primaryPositiveTrajectories": 3
    },
    "PHIRL_REGULARIZED_SOURCE": {
      "coverage": 0.994768962510898,
      "primaryBootstrap95": [
        -0.05926527484927596,
        0.10820395072117693
      ],
      "primaryCircularShiftPositiveP": 0.686599951183793,
      "primaryMedianRho": -0.02301292166628075,
      "primaryPositiveTrajectories": 5
    }
  },
  "retrospectiveCoherence": {
    "IIGR_CORRECTED_SOURCE": false,
    "PHIRL_REGULARIZED_SOURCE": false
  },
  "trajectoryCount": 12
}
```

The machine-readable equivalence rows, full and prefix local values, partition history, diagnostics, associations, spikes, future-dependence metrics, classification, suppression/failure ledger, replay audit, and figures preserve the complete status-bearing result.

### Source-equivalence confirmation

The untouched suite passed 14/14 rows. IIGR MI matrices were operation-exact (`max |delta| = 0`); PhiRL's largest MI difference was `3.12e-17`. Partition averages were exact, and the largest local Phi-r difference across either branch was `1.33e-15`, versus the frozen `1e-9` limit. The mandatory singular-duplicate case was `ELIGIBLE` in both the pinned IIGR source and wrapper. Constant-input PhiRL was identically `INELIGIBLE_TOO_FEW_ACTIVE_DIMENSIONS`; it was not relabeled as eligible. All original and wrapper replays were exact. Development/confirmation stream, numeric-seed, and fixture-payload intersections were all empty.

### Retrospective completed-fit source reconstruction

| Metric | IIGR corrected source | PhiRL regularized source |
|---|---:|---:|
| Finite coverage | 0.999690 | 0.999944 |
| Aggregate molecular-time slope (p) | 0.000443 (0.391970) | -0.102056 (0.006546) |
| Positive / negative 3-sigma excursions | 218 / 129 | 372 / 306 |
| Robust-MAD positive excursions | 27,271 | 700 |
| Defined within-run correlations | 11 | 12 |
| Positive within-run correlations | 8 | 0 |
| Median within-run Spearman rho | 0.021688 | -0.070582 |
| Trajectory-bootstrap 95% interval | [-0.124878, 0.065019] | [-0.108240, -0.042924] |
| Replicator-minus-drift mean Phi-r | -0.014762 | -1.358687 |
| Replicator-minus-drift median Phi-r | 0.000274 | 0.201389 |
| Runs with higher mean Phi-r during replication | 3 / 12 | 2 / 12 |
| Ljung-Box-significant runs | 6 / 12 | 12 / 12 |
| Frozen paper-directed coherence | FAIL | FAIL |

Completed-fit traces were punctuated and strongly temporally dependent, but punctuations alone were not the frozen success criterion. IIGR's small positive median correlation had an interval spanning zero, only three runs had a higher replication-state mean, and its circular-shift positive-direction p-value was 0.339. PhiRL pointed oppositely: all 12 within-run correlations were nonpositive, its median was negative, and the negative-direction circular-shift p-value was 0.0251. Thus neither full-trajectory implementation supplied a coherent positive paper-directed association on this audit set.

### Past-only prefix source analysis

| Branch and estimand | Eligible coverage | Defined / positive runs | Median rho | Bootstrap 95% interval | Positive-direction circular-shift p |
|---|---:|---:|---:|---:|---:|
| IIGR, current generation (primary) | 0.992153 | 11 / 3 | -0.019349 | [-0.074383, 0.003700] | 0.649744 |
| IIGR, next generation | 0.992153 | 11 / 5 | -0.014590 | [-0.045028, 0.034868] | 0.631438 |
| PhiRL, current generation (primary) | 0.994769 | 11 / 5 | -0.023013 | [-0.059265, 0.108204] | 0.686600 |
| PhiRL, next generation | 0.994769 | 11 / 5 | -0.009589 | [-0.032324, 0.101187] | 0.573590 |

The median first eligible generation was 6 for both implementations. One trajectory correlation was undefined where its historical label lacked within-run variation; it was retained as undefined rather than imputed. Neither implementation approached the `9/12` positive-run, positive median, positive bootstrap-bound, or `p <= 0.05` prospective gates. Exact future-suffix invariance passed for every executed sentinel.

### Future-dependence audit

| Shared-endpoint metric | IIGR | PhiRL |
|---|---:|---:|
| Shared eligible points | 1,134 | 1,139 |
| Median absolute full-prefix difference | 0.016420 | 4.858088 |
| Difference / full-fit IQR | 0.779867 | 0.761040 |
| Full-prefix Spearman rho | 0.197664 | 0.256761 |
| Sign agreement | 0.579365 | 0.667252 |
| Aggregate positive-3-sigma spike Jaccard | 0.061538 | 0.022831 |
| Median partition adjusted Rand index | 0.015734 | 0.046418 |
| Rank changed by more than 10 percentile points | 0.638448 | 0.664618 |

Completed-fit and prefix values therefore disagreed materially even though the prefix computation was internally reproducible. In the direct first-quarter refit audit, median normalized absolute differences were 0.900816 (IIGR) and 0.761888 (PhiRL); full-versus-quarter-refit correlations were 0.021760 and 0.415932, and more than 10-percentile rank changes affected 0.735005 and 0.671266 of values. These are descriptive future-dependence findings, not prospective early-warning evidence.

### Eligibility, suppression, and decision

Across the full traces, 39 of 125,666 IIGR and 7 of 125,678 PhiRL expected values were nonfinite. After the 256-transition prefix boundary, 9 of 1,147 IIGR and 6 of 1,147 PhiRL endpoints were nonfinite; all retained explicit statuses. Fractions were far below the frozen 20% full and 50% prefix stop thresholds. The numerical warnings arose where local Gaussian entropy terms became nonfinite and were preserved in `failure_ledger.csv` and row-level statuses.

Because neither completed-fit branch was coherently positive and neither prefix branch met the prospective gates, the frozen decision is `SOURCE_FAMILY_NOT_SUPPORTED`. This is a constraining result on these twelve frozen baselines, not proof about every possible GARD implementation. It neither rescues nor overturns S12 and cannot establish the unpublished author method.

## Validation

PASS — 14/14 untouched confirmation rows, replay, suffix invariance, immutability, scope, runtime, storage, and artifact gates passed. The preregistration, source identities, immutable prior artifact directory identities, implementation lock, phase firewall, exact replay, future-suffix invariance, numerical tolerances, schemas, artifact completeness, hashes, storage, runtime, and report-copy requirement were checked. CPU float64 was authoritative; GPU was not used.

## Runtime and storage

- Wall seconds: 6357.7889842898585.
- Worker CPU hours: 6.009035123501111.
- Benchmark: {"cpuGatePassed": true, "fullReplayPassed": true, "futureSuffixPassed": true, "matrixIndex": 0, "observedCpuSeconds": 1444.742446296, "observedResultBytes": 650501, "observedWallSeconds": 1445.7882532959338, "passed": true, "prefixReplayPassed": true, "projection": {"artifactBytes": 9757515, "cpuHours": 6.0197601929, "formula": "observed_complete_matrix0_times_12_plus_25_percent_reserve", "wallHours": 6.024117722066391}, "storageGatePassed": true, "trajectoryId": "E01-S12-B00", "wallGatePassed": true}.
- New trajectory count: 0; intervention trajectory count: 0.
- BLAS/OpenMP thread counts: one; source-analysis worker ceiling: six.

## Caveats, blockers, and limitations

- S12B remains a failed immutable step and S12 remains unchanged.
- The public source family is source-informed, not the unpublished GARD implementation.
- Full-trajectory values are retrospective and can depend on future observations.
- S13 and interventions remain blocked pending human review.

Even a successful public-source audit cannot rescue S12, establish the unpublished author method, validate early intervention, or authorize S13 automatically. Completed-fit local values use future observations and are descriptive only. Prefix values are an additive forensic causalization, not the paper's fixed-window method or MLP experiment.

## Provenance and artifacts

`source_snapshot_manifest.json`, `immutable_input_audit.json`, `implementation_lock.json`, `seed_firewall.json`, `scope_compliance.json`, `runtime_manifest.json`, and `artifact_manifest.json` provide source, input, code, seed, runtime, and file-level SHA-256 provenance. `S12C_FULL_RESULTS.md` and `research_step_full_results.md` are byte-exact copies.

## Recommended next action

Do not authorize S13. Close the E01 Phi-r reconstruction or move replacement-variable work to a separately preregistered E02 after human review. S13 remains blocked pending a new human decision; do not start it automatically. Stop here for mandatory human review. Do not begin S13, interventions, a further equivalence repair, new simulations, or any excluded campaign.
