# E01/S14 — Reconstruct Descriptive Causal-Emergence Dynamics

## Concise top summary

| Field | Result |
| --- | --- |
| Research step ID | `S14` (`E01-S14-DESCRIPTIVE-CAUSAL-EMERGENCE-DYNAMICS-v1.0.0`) |
| Completion status | **Complete** — only S14 was executed; S15 was not started |
| Artifacts written | 27 required paths under `/artifacts/research_steps/S14`: report, 14 machine-readable result tables, four figures, method/input/provenance/validation/status/failure manifests, and artifact manifest |
| Validation result | **PASS** — 18/18 checks; deterministic two-pass frame hashes and frozen-input before/after hashes matched |
| Outcome classification | **Constraining/contradictory — `PUNCTUATED_EXCURSIONS_WITH_AGGREGATE_TREND_DISCREPANCY`** |
| Caveats or blockers | Completed-fit values are retrospective; molecular trajectories have unequal lengths; the paper omits its Ljung–Box lag and exact spike threshold scope; past-only values are sparse post-fission endpoints; S13Y did not serialize covariance condition numbers |
| Lay summary | The reconstructed series do spike and remain strongly time-dependent, much like the paper, but both simulator candidates show a statistically detectable positive aggregate trend rather than the paper's reported no-trend result. |
| Recommended next action | Hand control back. Keep S15 queued and inactive until separately instructed; retain the aggregate-trend discrepancy and completed-fit/past-only dependence as fixed S15 context. |

## Lay summary

The closest locked reconstruction recovers the paper's qualitative picture of irregular bursts: 90 of 100 runs in each candidate contain at least one positive three-standard-deviation excursion, and all 100 differenced trajectories in each candidate reject the paper-like Ljung–Box independence test. Negative excursions are at least as prevalent, however, and the aggregate median rises significantly in both candidate pipelines. The visual resemblance therefore does not reproduce the paper's central combination of spikes *without* an aggregate trend.

The numerical values also depend materially on when the PhiRL fit is performed. Completed-trajectory values and independently refit past-only endpoint values agree only weakly to moderately at shared endpoints and have low signed-excursion overlap. This is descriptive evidence of retrospective temporal-fitting dependence, not early-warning or causal-control evidence.

## Frozen question

Do the frozen S13Y source-defined emergence values reproduce the paper's Figure 2-like combination of no aggregate linear trend and punctuated run-level excursions, separately for both confirmed simulator candidates?

## Inputs

- Frozen S13Y completed-fit source values: `full_source_values.parquet` (180,435 rows).
- Frozen S13Y past-only endpoint values: `prefix_endpoint_values.parquet` (20,000 status-bearing rows; eligible rows analyzed).
- Frozen S13Y partition history, source diagnostics, preprocessing diagnostics, trajectory manifest, and simulation summary.
- Original arXiv v1 paper PDF, SHA-256 recorded in `input_manifest.json`.
- Candidate 2: `h=0.6031526490073492`, first-daughter continuation. Candidate 3: `h=0.5613315384859516`, random-nonempty daughter continuation. Both retain trim-new-entrants and selected-daughter-boundary semantics.
- Exact S13Y PhiRL regularized source-emergence branch; no scalar, threshold, partition, preprocessing, label, alignment, or simulator was changed.

## Detailed methods

### Aggregate alignment and trend

The paper-like primary view groups completed-fit values by selected molecular-state index, takes the available-case median, and calculates the sample standard deviation across contributing trajectories. Ordinary unweighted linear regression of that median on molecular index is the primary trend. To expose unequal-length dependence without choosing a favorable result, the same locked calculation is also reported on full-cohort support, majority support, and a 101-point normalized-lifetime interpolation. Theil–Sen slopes and intervals are robustness diagnostics only.

### Excursions and morphology

The inherited S13Y rule is within-trajectory mean ± three population standard deviations (`ddof=0`). Robust excursions use median ± three times `1.4826 × MAD`. Positive and negative excursions remain separate. Consecutive flagged observations form an episode; the most signed-extreme observation (first tie) is its peak. The catalog records episode width, raw-index span, half-prominence width, prominence, inter-peak spacing, molecular/generation timing, normalized timing, and proximity to fission.

### Temporal dependence

Each eligible raw and first-differenced trajectory receives one Ljung–Box test at `max(1, min(10, floor(n/5)))`, exactly inheriting S13Y. Tests are unadjusted paper-like descriptive diagnostics. The paper does not specify its lag, so numerical agreement remains lag-underdetermined.

### Completed-fit versus past-only and dependency diagnostics

Eligible past-only endpoint rows are joined one-to-one to the exact completed-fit row by candidate, trajectory, matrix, and selected-sequence index. Unordered bipartitions are compared across consecutive eligible prefix fits, so swapping partition sides does not count as change. Fission enrichment uses descriptive Fisher exact tests on point-level 2×2 tables. Numerical diagnostics use only fields serialized by S13Y (finite flags, component-identity error, retained-variable count, and closure error); no missing covariance condition number was reconstructed.

## Commands

```bash
PYTHONPATH=src pytest -q tests/e01/test_s14_descriptive_causal_emergence.py
PYTHONPATH=src ruff check src/e01_descriptive_causal_emergence scripts/e01/run_s14_descriptive_dynamics.py tests/e01/test_s14_descriptive_causal_emergence.py
PYTHONPATH=src python -m compileall -q src/e01_descriptive_causal_emergence scripts/e01/run_s14_descriptive_dynamics.py
PYTHONPATH=src python scripts/e01/run_s14_descriptive_dynamics.py --output-root /artifacts/research_steps/S14
```

No simulator, PhiRL fitter, GPU process, package installer, or network call was invoked. CPU float64 was authoritative; execution was serial because the frozen-input analysis is small and deterministic.

## Results

### Aggregate trend and paper-facing excursions

| Candidate | Available-case slope | two-sided p | +3σ runs | −3σ runs | Robust +MAD runs | Robust −MAD runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S12F-CANDIDATE-02 | 8.19089e-05 | 3.1857e-13 | 90/100 | 99/100 | 100/100 | 100/100 |
| S12F-CANDIDATE-03 | 3.2348e-05 | 0.000428345 | 90/100 | 100/100 | 100/100 | 100/100 |

Both candidates reproduce punctuated positive excursions in a majority of runs. Both also have more prevalent negative excursions, and neither reproduces the paper's nonsignificant aggregate trend under the inherited available-case primary alignment. The alternate alignment rows are retained in `aggregate_trend_results.csv`; none is selected post hoc to replace the primary result.

### Ljung–Box reconstruction

| Candidate | Raw rejects | Raw median p | Differenced rejects | Differenced median p |
| --- | ---: | ---: | ---: | ---: |
| S12F-CANDIDATE-02 | 82/100 | 2.65971e-08 | 100/100 | 9.23561e-40 |
| S12F-CANDIDATE-03 | 79/100 | 7.00925e-06 | 100/100 | 4.1829e-43 |

The differenced 100/100 rejection count closely reconstructs the reported count for both candidates. Raw rejection is directionally similar but below the paper's 86/100 target. Because the original lag is unavailable, all such comparisons retain an explicit lag-underdetermined qualifier.

### Completed-fit versus past-only values

| Candidate | Shared endpoints | Event Spearman | Median runwise Spearman | Median absolute difference | +3σ Jaccard | −3σ Jaccard | Partition-change fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S12F-CANDIDATE-02 | 6767 | 0.3234 | 0.2983 | 0.6442 | 0.01695 | 0.06222 | 0.7454 |
| S12F-CANDIDATE-03 | 6938 | 0.3273 | 0.2966 | 0.6918 | 0.01695 | 0.06838 | 0.7288 |

Completed-fit and past-only endpoint values are therefore not interchangeable. Completed-fit partitions are fixed once per whole trajectory, so within-run completed-fit partition changes are structurally not applicable. Prefix partitions can change between endpoint refits; their direct and cross-mode spike associations are reported without causal interpretation.

### Numerical-condition boundary

All serialized S13Y fits used here were eligible, had finite MI/partition summaries, and retained exact emergence-component closure within the recorded tolerance. Preprocessing closure errors remained at floating-point scale. S13Y did not serialize covariance condition numbers, so spike dependence on that specific condition measure is **not evaluable without an unauthorized upstream recomputation**.

### Paper-target classification

The machine-readable `paper_target_comparison.csv` assigns candidate-specific statuses. In brief: E01-C014 (positive spikes) is directionally supported; E01-C024 (differenced Ljung–Box count) is closely reconstructed but lag-underdetermined; E01-C022/C023 are directionally supported but not numerically exact and lag-underdetermined; E01-C013 (no aggregate trend) is not supported within the tested scope.

## Figures and tables

- `figures/figure2_candidate_specific.png`: candidate-specific aggregate, fixed representative run, excursion prevalence, and Ljung–Box panels.
- `figures/figure2_pooled_secondary.png`: pooled secondary aggregate and spike timing.
- `figures/completed_fit_vs_past_only.png`: exact endpoint value comparison.
- `figures/spike_dependency_diagnostics.png`: fission and prefix-partition diagnostics.
- Machine-readable tables preserve every aggregate alignment, episode, run summary, trend, Ljung–Box result, partition comparison, and paper-target classification.

## Validation

Validation passed 18/18 checks. Two independent executions of every derived frame had identical content hashes. All 54 S13Y manifest members and all 200 frozen raw trajectory cache files matched before and after S14. Row keys, source-component identity, candidate contracts, trajectory cardinality, exact shared-endpoint joins, excursion catalog reconciliation, closed-form trend slopes, artifact schemas, figures, and final hashes passed. No S14 trajectory or estimator cache was created.

Repository checks passed: five focused S14 tests, Ruff, and bytecode compilation. The exact commit and remote branch are recorded in `provenance_manifest.json`.

## Caveats, blockers, failed assumptions, and limitations

- The primary completed-fit values use the completed trajectory to fit partitions and Gaussian parameters; they are retrospective.
- Molecular trajectories have unequal lengths. Available-case tail positions contain fewer trajectories; full-cohort, majority, and normalized-time alternatives are reported but cannot identify the unpublished paper alignment.
- The paper omits the exact 3σ threshold scope, Ljung–Box lag, and spike morphology definition. S14 inherits S13Y's locked within-run and lag rules rather than tuning them.
- Past-only values begin only after 256 prior locked-clock transitions and occur at post-fission endpoints, so they are a sparse comparator rather than a full molecular-time reconstruction.
- Fission and partition analyses are descriptive, point-dependent, and not causal tests. Completed-fit partition change is not a meaningful within-run variable.
- Covariance condition-number dependence is unavailable from the frozen serialized diagnostics. Recomputing it would change the authorized upstream analysis surface and was not done.
- Candidate pooling is secondary only. The constraining outcome follows the separate candidate-specific primary results.
- This step does not evaluate level/change association, prediction, or intervention claims and cannot support early warning or causal control.

## Provenance

- Repository: `https://github.com/Eidosoma/arrival-of-self-replicators.git` on `eidosoma/groups/42` at `1db3bf6672ca81a4b633c461354aeccde2ffd760`.
- Python: `3.13.14`; NumPy `2.4.6`, pandas `2.3.3`, SciPy `1.18.0`, statsmodels `0.14.6`, scikit-learn `1.9.0`, PyArrow `24.0.0`.
- Numeric policy: CPU float64, serial execution; no GPU acceleration.
- Input and output SHA-256 hashes: `input_manifest.json` and `artifact_manifest.json`.
- Reproducible source: repository config, `src/e01_descriptive_causal_emergence/`, runner, and focused tests; no repository source was copied into artifacts.

## Recommended next action

Hand control back to the Chief Scientist workflow. S15 is next in the directed queue but remains inactive until a separate instruction. When started, S15 should preserve this S14 aggregate-trend discrepancy, the strong signed/robust excursion evidence, the unspecified-lag boundary, and the completed-fit/past-only divergence. Do not begin S15 in this execution.
