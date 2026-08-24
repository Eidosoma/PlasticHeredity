# Clean-room plastic-heredity replication results

This run covers only the proposed plastic-heredity discovery. It does not run PhiID, first-replicator prediction, or intervention analyses.

## Outcome

This reduced profile is an implementation smoke test only; exact regeneration and the full matrix-level confirmation design were not run, so it carries no scientific verdict.

## Untouched confirmation

| Candidate | Split-half rho | Centered split-half rho | Full rho | Full centered rho | History centered rho | Log-loss gain A/B |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 0.8490 | 0.7772 | 0.7514 | 0.8136 | 0.6598 | 0.0574 / 0.0483 |
| 03 | 0.8496 | 0.3561 | 0.8041 | 0.6109 | 0.3880 | 0.0896 / 0.0502 |

## Clean-room evidence gates

| Candidate | Transition-region states | Reliability lower | Centered reliability lower | Minimum log-loss-gain lower | Minimum q-Brier-gain lower | Max permutation p |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 7/18 | 0.5746 | 0.4607 | 0.0093 | -0.0055 | 0.0308 |
| 03 | 4/18 | 0.6899 | 0.2336 | 0.0003 | -0.0047 | 0.0308 |

## Plastic-heredity process

Reported below are confirmation estimates. Episode quantities are conditional on a break; old-anchor quantities use the pre-break parent.

| Candidate | Break | Resume-2 | Episode-3 | Persist-5 | Old return | Mean old-anchor gain |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 0.2315 | 0.8200 | 0.6600 | 0.4800 | 0.1200 | -0.2547 |
| 03 | 0.2407 | 0.9231 | 0.8846 | 0.5192 | 0.0577 | -0.1863 |

## Numerical comparison

5/32 candidate-metric comparisons fall inside the ranges stated in the supplied manuscript. Range matching is descriptive and was not used by the simulator or model.

Exact confirmation regeneration: **not run**.

See `reported_comparison.csv` for every comparison and `metrics.json` for directional values, confidence intervals, and permutation tests.

## Interpretation boundary

This is an independent implementation, not a rerun of the unavailable L53/L54 code. The published materials do not specify the candidate contracts, the 195 feature coordinates, development cohort size, or all conditional event details. Those choices are frozen and disclosed in `manifest.json` and the repository `REPLICATION.md`. Agreement supports robustness to this explicit implementation; disagreement does not by itself refute the private implementation.
