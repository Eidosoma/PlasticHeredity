# Clean-room plastic-heredity replication results

This run covers only the proposed plastic-heredity discovery. It does not run PhiID, first-replicator prediction, or intervention analyses.

## Outcome

The central qualitative discovery replicated in both explicit candidates, but the reported numerical signature did not: the F12 event probability was state-dependent, and the frozen full state/graph/history student improved on direct history both within matrices and by branch log loss, while several reported prevalences and effect magnitudes fell outside their supplied ranges.

## Untouched confirmation

| Candidate | Split-half rho | Centered split-half rho | Full rho | Full centered rho | History centered rho | Log-loss gain A/B |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 0.9304 | 0.6644 | 0.8449 | 0.5609 | 0.3600 | 0.0292 / 0.0289 |
| 03 | 0.9190 | 0.6960 | 0.8391 | 0.6057 | 0.3370 | 0.0309 / 0.0333 |

## Clean-room evidence gates

| Candidate | Transition-region states | Reliability lower | Centered reliability lower | Minimum log-loss-gain lower | Minimum q-Brier-gain lower | Max permutation p |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 793/1000 | 0.9139 | 0.6080 | 0.0230 | 0.0082 | 0.0019 |
| 03 | 813/1000 | 0.8998 | 0.6508 | 0.0227 | 0.0096 | 0.0019 |

## Plastic-heredity process

Reported below are confirmation estimates. Episode quantities are conditional on a break; old-anchor quantities use the pre-break parent.

| Candidate | Break | Resume-2 | Episode-3 | Persist-5 | Old return | Mean old-anchor gain |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 0.5029 | 0.8483 | 0.7380 | 0.4950 | 0.0441 | -0.1804 |
| 03 | 0.5557 | 0.8620 | 0.7499 | 0.5019 | 0.0472 | -0.1773 |

## Numerical comparison

8/32 candidate-metric comparisons fall inside the ranges stated in the supplied manuscript. Range matching is descriptive and was not used by the simulator or model.

Exact confirmation regeneration: **True**.

See `reported_comparison.csv` for every comparison and `metrics.json` for directional values, confidence intervals, and permutation tests.

## Interpretation boundary

This is an independent implementation, not a rerun of the unavailable L53/L54 code. The published materials do not specify the candidate contracts, the 195 feature coordinates, development cohort size, or all conditional event details. Those choices are frozen and disclosed in `manifest.json` and the repository `REPLICATION.md`. Agreement supports robustness to this explicit implementation; disagreement does not by itself refute the private implementation.
