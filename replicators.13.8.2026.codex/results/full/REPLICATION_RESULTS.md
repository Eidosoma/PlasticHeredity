# Clean-room plastic-heredity replication results

This run covers only the proposed plastic-heredity discovery. It does not run PhiID, first-replicator prediction, or intervention analyses.

## Outcome

The central qualitative discovery replicated in both explicit candidates, but the reported numerical signature did not: the F12 event probability was state-dependent, and the frozen full state/graph/history student improved on direct history both within matrices and by branch log loss, while several reported prevalences and effect magnitudes fell outside their supplied ranges.

## Untouched confirmation

| Candidate | Split-half rho | Centered split-half rho | Full rho | Full centered rho | History centered rho | Log-loss gain A/B |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 0.9341 | 0.6850 | 0.8263 | 0.5538 | 0.3158 | 0.0389 / 0.0352 |
| 03 | 0.9270 | 0.7219 | 0.8394 | 0.6107 | 0.2708 | 0.0350 / 0.0349 |

## Clean-room evidence gates

| Candidate | Transition-region states | Reliability lower | Centered reliability lower | Minimum log-loss-gain lower | Minimum q-Brier-gain lower | Max permutation p |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 155/200 | 0.8874 | 0.5523 | 0.0234 | 0.0094 | 0.0019 |
| 03 | 156/200 | 0.8824 | 0.6079 | 0.0182 | 0.0086 | 0.0019 |

## Plastic-heredity process

Reported below are confirmation estimates. Episode quantities are conditional on a break; old-anchor quantities use the pre-break parent.

| Candidate | Break | Resume-2 | Episode-3 | Persist-5 | Old return | Mean old-anchor gain |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 0.4468 | 0.8328 | 0.7379 | 0.5141 | 0.0575 | -0.1681 |
| 03 | 0.5034 | 0.8523 | 0.7539 | 0.5306 | 0.0602 | -0.1663 |

## Numerical comparison

11/32 candidate-metric comparisons fall inside the ranges stated in the supplied manuscript. Range matching is descriptive and was not used by the simulator or model.

Exact confirmation regeneration: **True**.

See `reported_comparison.csv` for every comparison and `metrics.json` for directional values, confidence intervals, and permutation tests.

## Interpretation boundary

This is an independent implementation, not a rerun of the unavailable L53/L54 code. The published materials do not specify the candidate contracts, the 195 feature coordinates, development cohort size, or all conditional event details. Those choices are frozen and disclosed in `manifest.json` and the repository `REPLICATION.md`. Agreement supports robustness to this explicit implementation; disagreement does not by itself refute the private implementation.
