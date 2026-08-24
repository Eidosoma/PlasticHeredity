# S19-L04 decision summary

## Concise top summary

- **Research step ID:** S19-L04
- **Completion status:** COMPLETE; mandatory human review reached
- **Artifacts written:** full L04 evidence, validation and hash manifests, plus append-only S19 ledgers
- **Validation result:** PASS_ALL_LOCK_REPLAY_IMMUTABILITY_BOOTSTRAP_LOO_GENERATION_BLOCK_STORAGE_REGENERATION_AND_HASH_CHECKS
- **Outcome classification:** `EXPLORATORY_DIRECTIONAL_MATCH`; 0 promoted lead(s)
- **Caveats or blockers:** Previously studied matrices; completed-run future dependence; unavailable author definition; occupancy alone prohibited
- **Recommended next action:** Human review; authorize no L05 or S20 automatically
- **Lay summary:** One fixed cross-generation recurrence definition was tested against the full paper fingerprint, not tuned to 88% occupancy.

## Decision evidence

| candidateId       | labelId                              |   meanOccupancy |   meanPersistence |   meanConsistency |   meanFirstOnsetRawIndex0 |   meanFirstOnsetNormalized |   meanEpisodeCount |   nonreplicatingAtCutoffFraction |
|:------------------|:-------------------------------------|----------------:|------------------:|------------------:|--------------------------:|---------------------------:|-------------------:|---------------------------------:|
| S12F-CANDIDATE-02 | MOL_ADJACENT_INCOMING_H900           |          0.9809 |          858.4100 |            0.0713 |                    1.1000 |                     0.0016 |            16.8600 |                           0.0300 |
| S12F-CANDIDATE-03 | MOL_ADJACENT_INCOMING_H900           |          0.9827 |          913.7800 |            0.0906 |                    0.9700 |                     0.0014 |            16.3000 |                           0.0200 |
| S12F-CANDIDATE-02 | MOL_CROSS_GENERATION_RECURRENCE_H900 |          0.9197 |          796.2400 |            0.5416 |                    6.2300 |                     0.0075 |            30.5400 |                           0.1100 |
| S12F-CANDIDATE-03 | MOL_CROSS_GENERATION_RECURRENCE_H900 |          0.9216 |          847.4400 |            0.5750 |                    6.4500 |                     0.0072 |            30.1200 |                           0.0400 |

## Negative-control gate

| candidateId       | labelId                              | onsetMode   | controlId                          |   replicates |   observedPaperDistance |   nullLower2_5 |   nullMedian |   nullUpper97_5 |   lowerTailP | negativeControlPassed   | occupancyInvariantByConstruction   | completedRunMembershipInvariantByConstruction   |
|:------------------|:-------------------------------------|:------------|:-----------------------------------|-------------:|------------------------:|---------------:|-------------:|----------------:|-------------:|:------------------------|:-----------------------------------|:------------------------------------------------|
| S12F-CANDIDATE-02 | MOL_CROSS_GENERATION_RECURRENCE_H900 | RAW         | GENERATION_BLOCK_ORDER_PERMUTATION |         4096 |                  1.6181 |         1.4702 |       1.4819 |          1.4989 |       1.0000 | False                   | True                               | True                                            |
| S12F-CANDIDATE-02 | MOL_CROSS_GENERATION_RECURRENCE_H900 | NORMALIZED  | GENERATION_BLOCK_ORDER_PERMUTATION |         4096 |                  1.6566 |         1.4788 |       1.4905 |          1.5074 |       1.0000 | False                   | True                               | True                                            |
| S12F-CANDIDATE-03 | MOL_CROSS_GENERATION_RECURRENCE_H900 | RAW         | GENERATION_BLOCK_ORDER_PERMUTATION |         4096 |                  1.8845 |         1.6958 |       1.7081 |          1.7254 |       1.0000 | False                   | True                               | True                                            |
| S12F-CANDIDATE-03 | MOL_CROSS_GENERATION_RECURRENCE_H900 | NORMALIZED  | GENERATION_BLOCK_ORDER_PERMUTATION |         4096 |                  1.9190 |         1.7033 |       1.7156 |          1.7329 |       1.0000 | False                   | True                               | True                                            |

Promoted leads: **0** (none).

## Human-review boundary

Stop now. L05, S20, E02, author contact, and report generation are inactive.
