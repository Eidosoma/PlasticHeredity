# S13Y Clean Directional Confirmation: Full Results

## Concise top summary

- **Research step ID:** `E01-S13Y-CLEAN-DIRECTIONAL-CONFIRMATION-v1.0.0` (actual step `S13Y`).
- **Completion status:** Complete; stopped at the mandatory S13Y human-review boundary.
- **Artifacts written:** 55 retained paths under `$ARTIFACTS_DIR/research_steps/S13Y/` (54 hash-listed artifacts plus the artifact manifest), including 200 trajectory identities, all status-bearing source/prefix rows, circularity controls, machine-readable inference, figures, manifests, and this report.
- **Validation result:** PASS_ALL_FROZEN_COMPUTE_SIMULATION_PAIRING_SOURCE_REPLAY_SUFFIX_LABEL_CIRCULARITY_STATISTICS_SCHEMA_PROVENANCE_IMMUTABILITY_RUNTIME_STORAGE_AND_HASH_GATES.
- **Outcome classification:** `LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE` (supportive_with_structural_circularity_constraint).
- **Caveats or blockers:** S13X selected this branch adaptively; full fits are future-fitted and retrospective; the primary binary label is exactly determined by H; prior strict, prefix, intervention, sensitivity, and held-out non-support remain unchanged; no author-code identity is established.
- **Recommended next action:** Mandatory human review. Do not begin evidence synthesis, E02, report-bundle generation, S14–S18, prediction, or intervention work automatically.

## Lay summary

We generated a completely new set of 100 catalytic matrices and ran each through both previously confirmed simulator time-base candidates. We then tested only the one pattern found in S13X: whether the public PhiRL source-emergence number is higher when consecutive molecular compositions are very similar (`H>0.9`). The completed-trajectory calculation is descriptive because its partition and Gaussian model use the finished run. We also repeated the calculation using only past prefixes. Finally, we checked the central circularity directly: the binary target is literally constructed by thresholding H, so exact H predicts the label perfectly and no other quantity can add unrestricted information about that label conditional on exact H.

## Frozen question and interpretation boundary

The frozen hypothesis was the exact S13X lead `S13X-P-684e66c4cffe914c`: PhiRL regularized source-defined emergence, level transform, molecular adjacent-incoming `H>0.9`, and same-state alignment. Candidate-specific evidence is primary and both candidates had to pass the same association and replicator-minus-drift gates. `H>0.97` was descriptive only. No search, intervention, prediction, MLP, estimator change, or extra simulator was allowed.

Every completed-fit value is labeled `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL`. It cannot support early warning, intervention, prediction, or causal control. S13Y is a clean test of a post-selection hypothesis, not a test of the unavailable author implementation.

## Inputs and provenance

- Original paper and its reported directional fingerprints were refreshed before design freeze.
- Simulator candidates were unchanged S12FR candidates 2 and 3, with the exact exposures, daughter rules, trimmed-new-entrant semantics, and C1 clock recorded in the preregistration.
- PhiRL was pinned to commit `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; the safe lattice hash was `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`.
- The S13Y seed root was new and domain separated. Catalytic matrices and initial states were shared across candidates; dynamics streams were candidate-specific.

## Detailed methods

Exactly 100 new beta matrices were sampled as `exp(-4 + 4Z)` and 100 matched mass-40 distinct-singleton initial states were generated. Each pair was simulated under both candidate contracts for 100 fissions. Counts received additive-0.5 closure, full CLR, and original component 100 removal. PhiRL source emergence was synergy plus the two downward-causation atoms. Full fits used the complete trajectory. Prefix fits independently reran the same source pipeline from the start through each post-fission endpoint after 256 C1 transitions and retained only the final local value.

For every trajectory we calculated Spearman association and the mean emergence difference between label-positive and label-negative molecular states. Candidate inference used 4,096 trajectory bootstraps and 4,096 nonzero circular rotations, which preserve each cyclic binary label sequence and its episode durations. The clean gate required at least 80 defined trajectories, at least 65% positive correlations, positive median rho with a positive bootstrap lower bound and shift p<=0.05; the drift gate required at least 50% positive differences plus the analogous median/bootstrap/null conditions.

Circularity controls recorded continuous incoming H, ordinary Euclidean L2 composition change, exact `Y=I(H>0.9)` identity, and a fixed smooth-H/L2 within-trajectory emergence regression. The exact deterministic identity is authoritative: `H(Y|H)=0`, hence unrestricted `I(E;Y|H)=0`. The smooth regression is only a model-dependent threshold-discontinuity diagnostic.

## Commands

```bash
PYTHONPATH=src python scripts/e01/freeze_s13y_preregistration.py
PYTHONPATH=src python scripts/e01/run_s13y_clean_directional_confirmation.py --stage benchmark
PYTHONPATH=src python scripts/e01/run_s13y_clean_directional_confirmation.py --stage full --workers 6
PYTHONPATH=src pytest -q tests/e01/test_s13y_clean_directional_confirmation.py
ruff check src/e01_clean_directional_confirmation scripts/e01/freeze_s13y_preregistration.py scripts/e01/run_s13y_clean_directional_confirmation.py tests/e01/test_s13y_clean_directional_confirmation.py
```

## Primary retrospective results

| Candidate | Defined | Positive | Median rho | 95% bootstrap | shift p | Higher during replication | Median difference | Candidate gate |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| S12F-CANDIDATE-02 | 99 | 78 (0.788) | 0.058007 | [0.039025, 0.075124] | 0.000244 | 75 (0.758) | 0.395836 | True |
| S12F-CANDIDATE-03 | 99 | 76 (0.768) | 0.055443 | [0.030550, 0.076080] | 0.000244 | 74 (0.747) | 0.516156 | True |

The all-candidate classification is `LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE`. A favorable candidate could not rescue the other candidate.

## Label-circularity controls

- S12F-CANDIDATE-02: 0/87487 label-identity mismatches; baseline H classification accuracy 1.000; median emergence–H rho 0.269943; median emergence–negative-L2-change rho 0.277763.
- S12F-CANDIDATE-03: 0/92948 label-identity mismatches; baseline H classification accuracy 1.000; median emergence–H rho 0.270713; median emergence–negative-L2-change rho 0.275472.

Because the label is exactly a thresholded H value, the binary target has zero conditional entropy given exact H. This is a structural result, not a finite-sample failure: PhiRL cannot add unrestricted information about that same binary target after exact H is known. Any positive primary association is therefore bounded to the preregistered label-coupled retrospective interpretation.

## Past-only prefix falsification

| Candidate | Defined | Positive | Median rho | 95% bootstrap | shift p | Prefix gate |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| S12F-CANDIDATE-02 | 85 | 29 (0.341) | -0.074072 | [-0.109534, -0.052205] | 1.000000 | False |
| S12F-CANDIDATE-03 | 84 | 30 (0.357) | -0.069320 | [-0.107069, -0.029635] | 1.000000 | False |

Prefix results are a secondary falsification and do not change the retrospective-only status of full-fit values. No trajectory was pooled to cross the 256-transition boundary.

## Temporal and spike descriptions

| candidateId       |   trajectoryCount |   aggregateSlope |   aggregateSlopeP |   positive3SigmaRunFraction |   negative3SigmaRunFraction |   robustMadRunFraction |   rawLjungBoxSignificantFraction |   differencedLjungBoxSignificantFraction | punctuatedDescriptiveGatePassed   | weakAggregateTrendDescriptiveGatePassed   |
|:------------------|------------------:|-----------------:|------------------:|----------------------------:|----------------------------:|-----------------------:|---------------------------------:|-----------------------------------------:|:----------------------------------|:------------------------------------------|
| S12F-CANDIDATE-02 |               100 |      8.19089e-05 |       3.1857e-13  |                         0.9 |                        0.99 |                      1 |                             0.82 |                                        1 | True                              | False                                     |
| S12F-CANDIDATE-03 |               100 |      3.2348e-05  |       0.000428345 |                         0.9 |                        1    |                      1 |                             0.79 |                                        1 | True                              | False                                     |

These are descriptive paper-resemblance checks and are not permitted to rescue a failed primary association.

## Validation

- PASS_ALL_FROZEN_COMPUTE_SIMULATION_PAIRING_SOURCE_REPLAY_SUFFIX_LABEL_CIRCULARITY_STATISTICS_SCHEMA_PROVENANCE_IMMUTABILITY_RUNTIME_STORAGE_AND_HASH_GATES.
- 200/200 trajectories completed 100 fissions and replayed exactly; pairing, source replay, prefix replay, suffix invariance, component identity, finite coverage, schema, seed firewall, immutability, statistics replay, runtime, storage, and artifact gates passed: True.
- The focused repository suite passed 5/5 tests under `PYTHONPATH=src`; Ruff passed with no findings.
- Cumulative CPU envelope after S13Y: 140.947/250 hours; GPU envelope: 2.000/80 hours.
- CPU float64 was authoritative; six workers and one numerical-library thread per worker were used; the L4 was not used.

## Caveats, failed assumptions, and limitations

- The hypothesis was generated adaptively in S13X, so this is clean post-selection confirmation rather than independent discovery.
- Exact H-label determinism is a circularity constraint. A thresholded label cannot demonstrate incremental information beyond its own defining coordinate.
- PhiRL is a later public source implementation, not the unavailable GARD-paper code.
- Full fits estimate partitions and Gaussian distributions from completed trajectories and are future-dependent.
- The fixed-window and early-time claims remain unavailable; prefix values begin only after 256 locked-clock transitions.
- S12 strict estimates, S13RRR held-out results, S13X prefix and intervention results, and all historical failures remain unchanged and must coexist with this result.

## Artifact and software provenance

Design commit: `41b235c882effe03accf86256e586c2530eadf66` on `eidosoma/groups/42`. Python 3.13.14, NumPy 2.4.6, pandas 2.3.3, SciPy 1.18.0. Raw trajectory and resumable source caches remain under `/cache/e01_s13y_v1`; compact evidence is under `$ARTIFACTS_DIR/research_steps/S13Y/`.

## Recommended next action

Return for mandatory human review. Do not begin evidence synthesis, E02, report-bundle generation, S14–S18, another scale-up, intervention, prediction, or estimator work automatically.
