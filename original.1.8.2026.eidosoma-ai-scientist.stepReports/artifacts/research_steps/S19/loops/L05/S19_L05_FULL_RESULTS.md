# E01/S19-L05 Full Results — Past-Only Cross-Generation Recurrence Activation

## Concise handoff summary

- **Research step ID:** `S19-L05`
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Artifacts written:** complete L05 preregistration/method/label/bundle/seed locks; exact preanalysis and independent replay evidence; 361,270 frozen-comparator/structural label rows; temporal fingerprints, episode/cutoff/recurrence tables; suffix audits; 4,096-replicate matrix bootstrap and recomputed generation-block permutation controls; leave-one-out, fixed-L04 comparison, validation, runtime/storage, status, ledgers, and hash manifests
- **Validation result:** `PASS_ALL_LOCK_PREANALYSIS_PRIMARY_INDEPENDENT_COMPARATOR_SUFFIX_IMMUTABILITY_BOOTSTRAP_LOO_RECOMPUTED_GENERATION_BLOCK_STORAGE_REGENERATION_AND_HASH_CHECKS`
- **Outcome classification:** `EXPLORATORY_DIRECTIONAL_MATCH`; `CONSTRAINING_OR_NULL_EXPLORATORY`; `0` lead(s) promoted
- **Caveats or blockers:** exploratory reuse of studied matrices; the paper does not uniquely specify this one-sided algorithm; known paper fingerprints informed the locked gate; slow drift can still generate high recurrence; no emergence, prediction, intervention, or causal-control evidence was produced
- **Recommended next action:** mandatory human review. Choose a separately authorized bounded action; L06, S20, E02, author contact, and report generation remain inactive.

## Lay summary

L04 asked whether a composition matched another generation anywhere in the completed run. That improved the paper-like temporal pattern but used future observations. L05 made the rule genuinely one-sided: a state can turn positive only when it matches a nonadjacent state already observed in an earlier generation. It never backfills earlier states and never carries a positive state forward automatically. The one past-only label did not pass every locked promotion gate and was not promoted. The failed gate(s) constrain this exact one-sided rule; they do not identify the unavailable author implementation.

The comparison with 88% occupancy was deliberately joint rather than occupancy-only. Persistence, onset, consistency, episodes, quarter-cutoff eligibility, recurrence counts, candidate agreement, uncertainty, influence, future-suffix invariance, and a block-order negative control were all evaluated under the single frozen rule.

## Frozen question

Does strict past-only recurrence to a nonadjacent state in an earlier positive-numbered generation create a meaningful online label onset and improve the locked four-dimensional paper fingerprint over adjacent molecular `H>0.9` in both simulator candidates, without future-suffix dependence?

This is an exploratory label-definition question only. It does not test causal-emergence association, prediction, intervention, or causal control.

## Inputs

- Frozen S13Y `trajectory_manifest.parquet`, `label_values.parquet`, and 200 trajectory caches: 100 shared catalytic matrices for each of candidate 2 and candidate 3.
- Candidate 2 and candidate 3 were analyzed separately. No favorable-candidate selection and no primary pooling were used.
- Frozen L04 aggregate evidence was read only for the specified retrospective comparison; no L04 value was recomputed or changed.
- Original paper and pinned historical GARD source context identified cross-generation recurrence as plausible but did not uniquely identify this one-sided molecular algorithm.
- No new GARD trajectory, PhiRL/emergence value, prediction fit, intervention, GPU computation, or author contact occurred.

## Detailed methods

### Label contract

For selected-clock state `t` in positive generation `g`, L05 assigns one iff there is at least one state `s <= t-2` with `0 < g_s < g` and strict cosine `H(s,t) > 0.9` on L1-closed 100-component compositions. Generation zero is retained but ineligible. Each distinct earlier generation counts once. The algorithm is sequential, uses no future state, performs no backfill, and adds no carry-forward or persistence rule. Adjacent molecular `H>0.9` is comparator-only.

There is exactly one L05 structural specification: no threshold grid, `H>0.97`, cluster, centroid, boundary projection, alignment, modal-reference choice, emergence selection, prediction selection, or intervention selection.

### Temporal fingerprint and paper-distance

Each catalytic-matrix trajectory is the independent unit. Reported dimensions include occupancy, persistence, zero- and one-based raw onset, normalized onset, consecutive-label Pearson consistency, entries/exits, episode number and duration, longest episode, state and no-onset status through the floor-25% cutoff, recurrent-generation counts, and candidate agreement.

The locked joint distance is the RMS of four standardized deviations from the paper control targets: persistence `716 +/- 198`, occupancy `0.88 +/- 0.03`, consistency `0.38 +/- 0.06`, and either raw onset `37 +/- 27` or normalized onset `0.37 +/- 0.27`. Raw and normalized onset remain separate because the paper's units are ambiguous.

### Robustness and falsification

- Exact two-pass materialization was required for both labels on every trajectory.
- A separately written row-loop implementation replayed primary labels, scores, distinct-generation counts, qualifying-state counts, earliest/latest matches, and full matching-index identities.
- At five length-derived endpoints per trajectory, deletion, row shuffling, and component replacement of the future suffix had to leave all prefix results exactly invariant: 3,000 locked sentinels total.
- The matrix bootstrap used 4,096 paired replicates per candidate with identical resampled matrix identities for structural and comparator labels.
- Every leave-one-matrix-out result was evaluated.
- The negative control independently permuted all 100 complete positive-generation blocks in each matrix for each of 4,096 replicates, preserved within-block order, sequentially renumbered generations, reapplied `s<=t-2`, and recomputed the past-only label. Passing required observed raw and normalized paper-distance below the null 2.5th percentile in both candidates.

## Results

### Candidate-specific temporal fingerprints

| candidateId       | labelId                                        |   meanOccupancy |   meanPersistence |   meanConsistency |   meanFirstOnsetRawIndex0 |   meanFirstOnsetNormalized |   meanEntryCount |   meanExitCount |   meanEpisodeCount |   meanMeanEpisodeDuration |   meanLongestEpisode |   nonreplicatingAtCutoffFraction |   noReplicatorThroughCutoffFraction |   meanRecurrentGenerationCount |
|:------------------|:-----------------------------------------------|----------------:|------------------:|------------------:|--------------------------:|---------------------------:|-----------------:|----------------:|-------------------:|--------------------------:|---------------------:|---------------------------------:|------------------------------------:|-------------------------------:|
| S12F-CANDIDATE-02 | MOL_ADJACENT_INCOMING_H900                     |          0.9809 |          858.4100 |            0.0713 |                    1.1000 |                     0.0016 |          16.8600 |         15.9700 |            16.8600 |                   79.2762 |             251.8900 |                           0.0300 |                              0.0000 |                       100.5300 |
| S12F-CANDIDATE-03 | MOL_ADJACENT_INCOMING_H900                     |          0.9827 |          913.7800 |            0.0906 |                    0.9700 |                     0.0014 |          16.3000 |         15.4000 |            16.3000 |                  102.0275 |             281.1600 |                           0.0200 |                              0.0000 |                       100.5400 |
| S12F-CANDIDATE-02 | MOL_PAST_ONLY_CROSS_GENERATION_RECURRENCE_H900 |          0.8018 |          685.2000 |            0.6534 |                    9.1000 |                     0.0112 |          49.8800 |         49.2700 |            49.8800 |                   32.1080 |             128.6100 |                           0.1900 |                              0.0000 |                        98.1000 |
| S12F-CANDIDATE-03 | MOL_PAST_ONLY_CROSS_GENERATION_RECURRENCE_H900 |          0.8027 |          728.0900 |            0.6541 |                    9.3300 |                     0.0106 |          51.0400 |         50.4900 |            51.0400 |                   33.2289 |             140.5000 |                           0.2000 |                              0.0000 |                        98.1900 |

The full matrix-level distributions, medians, standard deviations, eligibility counts, episode tables, quarter-cutoff results, and recurrence diagnostics are machine-readable. Occupancy is presented as only one part of the fingerprint.

### Joint comparison with adjacent `H>0.9`

| candidateId       | onsetMode   |   paperDistance |   comparatorDistance |   distanceDifferenceCandidateMinusComparator |   distanceImprovementFraction |   closerDimensionCount | structureDimensionImproved   | occupancyCloser   |
|:------------------|:------------|----------------:|---------------------:|---------------------------------------------:|------------------------------:|-----------------------:|:-----------------------------|:------------------|
| S12F-CANDIDATE-02 | RAW         |          2.6764 |               3.1646 |                                      -0.4882 |                        0.1543 |                      4 | True                         | True              |
| S12F-CANDIDATE-02 | NORMALIZED  |          2.7088 |               3.1683 |                                      -0.4595 |                        0.1450 |                      4 | True                         | True              |
| S12F-CANDIDATE-03 | RAW         |          2.6722 |               3.0723 |                                      -0.4000 |                        0.1302 |                      4 | True                         | True              |
| S12F-CANDIDATE-03 | NORMALIZED  |          2.7057 |               3.0756 |                                      -0.3699 |                        0.1203 |                      4 | True                         | True              |

Negative distance differences favor L05. A directional improvement does not establish an exact numerical match, and neither can bypass the prospective robustness gates.

### Fixed comparison with L04 completed-run symmetric membership

| candidateId       | metric                    |   l05Mean |   l04FrozenMean |   l05MinusL04 |
|:------------------|:--------------------------|----------:|----------------:|--------------:|
| S12F-CANDIDATE-02 | persistence               |  685.2000 |        796.2400 |     -111.0400 |
| S12F-CANDIDATE-02 | occupancy                 |    0.8018 |          0.9197 |       -0.1179 |
| S12F-CANDIDATE-02 | consistency               |    0.6534 |          0.5416 |        0.1117 |
| S12F-CANDIDATE-02 | firstOnsetRawScore        |    9.1000 |          6.2300 |        2.8700 |
| S12F-CANDIDATE-02 | firstOnsetNormalizedScore |    0.0112 |          0.0075 |        0.0038 |
| S12F-CANDIDATE-03 | persistence               |  728.0900 |        847.4400 |     -119.3500 |
| S12F-CANDIDATE-03 | occupancy                 |    0.8027 |          0.9216 |       -0.1190 |
| S12F-CANDIDATE-03 | consistency               |    0.6541 |          0.5750 |        0.0791 |
| S12F-CANDIDATE-03 | firstOnsetRawScore        |    9.3300 |          6.4500 |        2.8800 |
| S12F-CANDIDATE-03 | firstOnsetNormalizedScore |    0.0106 |          0.0072 |        0.0035 |

This comparison isolates the cost of removing future membership and backfilling. L04 remains a frozen retrospective result and is not eligible as online evidence.

### Matrix bootstrap

| candidateId       | onsetMode   |   meanDistanceDifference |   lower95 |   upper95 |   probabilityDistanceImprovement |
|:------------------|:------------|-------------------------:|----------:|----------:|---------------------------------:|
| S12F-CANDIDATE-02 | RAW         |                  -0.4851 |   -0.7009 |   -0.2679 |                           1.0000 |
| S12F-CANDIDATE-02 | NORMALIZED  |                  -0.4565 |   -0.6701 |   -0.2420 |                           1.0000 |
| S12F-CANDIDATE-03 | RAW         |                  -0.3959 |   -0.6230 |   -0.1638 |                           0.9998 |
| S12F-CANDIDATE-03 | NORMALIZED  |                  -0.3658 |   -0.5909 |   -0.1350 |                           0.9993 |

Selected metric-difference intervals:

| candidateId       | metric                    |   meanDifference |   lower95 |   upper95 |
|:------------------|:--------------------------|-----------------:|----------:|----------:|
| S12F-CANDIDATE-02 | persistence               |        -173.3718 | -194.9563 | -151.8675 |
| S12F-CANDIDATE-02 | occupancy                 |          -0.1793 |   -0.1971 |   -0.1612 |
| S12F-CANDIDATE-02 | consistency               |           0.5821 |    0.5538 |    0.6096 |
| S12F-CANDIDATE-02 | firstOnsetRawScore        |           7.9999 |    7.1100 |    8.9100 |
| S12F-CANDIDATE-02 | firstOnsetNormalizedScore |           0.0096 |    0.0085 |    0.0109 |
| S12F-CANDIDATE-03 | persistence               |        -185.6887 | -210.6800 | -161.7912 |
| S12F-CANDIDATE-03 | occupancy                 |          -0.1799 |   -0.1988 |   -0.1610 |
| S12F-CANDIDATE-03 | consistency               |           0.5635 |    0.5233 |    0.6008 |
| S12F-CANDIDATE-03 | firstOnsetRawScore        |           8.3620 |    7.5400 |    9.1800 |
| S12F-CANDIDATE-03 | firstOnsetNormalizedScore |           0.0092 |    0.0083 |    0.0101 |

### Future-suffix invariance

| candidateId       | variant   |   sentinels |   passed |   mutationsEffective |
|:------------------|:----------|------------:|---------:|---------------------:|
| S12F-CANDIDATE-02 | DELETE    |         500 |      500 |                  500 |
| S12F-CANDIDATE-02 | REPLACE   |         500 |      500 |                  500 |
| S12F-CANDIDATE-02 | SHUFFLE   |         500 |      500 |                  500 |
| S12F-CANDIDATE-03 | DELETE    |         500 |      500 |                  500 |
| S12F-CANDIDATE-03 | REPLACE   |         500 |      500 |                  500 |
| S12F-CANDIDATE-03 | SHUFFLE   |         500 |      500 |                  500 |

Every suffix check compares labels, scores, distinct and qualifying recurrence counts, and all retained matching-index identities at the locked prefix endpoint.

### Recomputed generation-block negative control

| candidateId       | onsetMode   |   observedPaperDistance |   nullLower2_5 |   nullMedian |   nullUpper97_5 |   lowerTailP | negativeControlPassed   |
|:------------------|:------------|------------------------:|---------------:|-------------:|----------------:|-------------:|:------------------------|
| S12F-CANDIDATE-02 | RAW         |                  2.6764 |         4.4033 |       4.4608 |          4.5190 |       0.0002 | True                    |
| S12F-CANDIDATE-02 | NORMALIZED  |                  2.7088 |         4.4453 |       4.5017 |          4.5586 |       0.0002 | True                    |
| S12F-CANDIDATE-03 | RAW         |                  2.6722 |         4.4717 |       4.5353 |          4.5974 |       0.0002 | True                    |
| S12F-CANDIDATE-03 | NORMALIZED  |                  2.7057 |         4.5110 |       4.5744 |          4.6352 |       0.0002 | True                    |

### Comparator overlap and cross-candidate agreement

| candidateId       | labelId                                        | baselineId                 |   commonEligibleCount |   accuracy |   jaccard |   mismatchFraction |   structuralPositiveAdjacentNegative |   structuralNegativeAdjacentPositive |
|:------------------|:-----------------------------------------------|:---------------------------|----------------------:|-----------:|----------:|-------------------:|-------------------------------------:|-------------------------------------:|
| S12F-CANDIDATE-02 | MOL_PAST_ONLY_CROSS_GENERATION_RECURRENCE_H900 | MOL_ADJACENT_INCOMING_H900 |                 87487 |     0.7973 |    0.7939 |             0.2027 |                                  232 |                                17499 |
| S12F-CANDIDATE-03 | MOL_PAST_ONLY_CROSS_GENERATION_RECURRENCE_H900 | MOL_ADJACENT_INCOMING_H900 |                 92948 |     0.7960 |    0.7929 |             0.2040 |                                  224 |                                18738 |

| labelId                                        | metric                    |   pairedDefinedCount |   candidate2Mean |   candidate3Mean |   absoluteMeanDifference |   matrixLevelPearson |
|:-----------------------------------------------|:--------------------------|---------------------:|-----------------:|-----------------:|-------------------------:|---------------------:|
| MOL_ADJACENT_INCOMING_H900                     | occupancy                 |                  100 |           0.9809 |           0.9827 |                   0.0018 |               0.6907 |
| MOL_ADJACENT_INCOMING_H900                     | consistency               |                   98 |           0.0713 |           0.0906 |                   0.0192 |               0.6741 |
| MOL_ADJACENT_INCOMING_H900                     | firstOnsetNormalizedScore |                  100 |           0.0016 |           0.0014 |                   0.0001 |               0.7084 |
| MOL_ADJACENT_INCOMING_H900                     | episodeCount              |                  100 |          16.8600 |          16.3000 |                   0.5600 |               0.7441 |
| MOL_PAST_ONLY_CROSS_GENERATION_RECURRENCE_H900 | occupancy                 |                  100 |           0.8018 |           0.8027 |                   0.0009 |               0.7623 |
| MOL_PAST_ONLY_CROSS_GENERATION_RECURRENCE_H900 | consistency               |                  100 |           0.6534 |           0.6541 |                   0.0007 |               0.4864 |
| MOL_PAST_ONLY_CROSS_GENERATION_RECURRENCE_H900 | firstOnsetNormalizedScore |                  100 |           0.0112 |           0.0106 |                   0.0006 |               0.5972 |
| MOL_PAST_ONLY_CROSS_GENERATION_RECURRENCE_H900 | episodeCount              |                  100 |          49.8800 |          51.0400 |                   1.1600 |               0.8375 |

### Promotion gates

| gate                                                         | passed   |
|:-------------------------------------------------------------|:---------|
| structuralNotComparator                                      | True     |
| exactTwoPassReplayAll400LabelTrajectories                    | True     |
| independentStructuralReplayAll200Trajectories                | True     |
| exactFrozenAdjacentComparatorReplay                          | True     |
| exactFutureSuffixInvarianceAll3000Sentinels                  | True     |
| allSuffixMutationsEffective                                  | True     |
| preciseHumanLockedPaperRelationship                          | True     |
| noOutcomeTunedChoice                                         | True     |
| untouchedS20DesignComplete                                   | True     |
| occupancyCloser_S12F-CANDIDATE-02                            | True     |
| jointDistanceBetterBothModes_S12F-CANDIDATE-02               | True     |
| threeDimensionsIncludingOnsetOrConsistency_S12F-CANDIDATE-02 | True     |
| bootstrapUpperBelowZeroBothModes_S12F-CANDIDATE-02           | True     |
| allLeaveOneOutImproved_S12F-CANDIDATE-02                     | True     |
| generationBlockPermutationControlBothModes_S12F-CANDIDATE-02 | True     |
| coverage_S12F-CANDIDATE-02                                   | True     |
| quarterEligibility_S12F-CANDIDATE-02                         | False    |
| occupancyCloser_S12F-CANDIDATE-03                            | True     |
| jointDistanceBetterBothModes_S12F-CANDIDATE-03               | True     |
| threeDimensionsIncludingOnsetOrConsistency_S12F-CANDIDATE-03 | True     |
| bootstrapUpperBelowZeroBothModes_S12F-CANDIDATE-03           | True     |
| allLeaveOneOutImproved_S12F-CANDIDATE-03                     | True     |
| generationBlockPermutationControlBothModes_S12F-CANDIDATE-03 | True     |
| coverage_S12F-CANDIDATE-03                                   | True     |
| quarterEligibility_S12F-CANDIDATE-03                         | False    |
| crossCandidateAgreement                                      | True     |

At most one lead could be promoted. The result above follows the conjunction of every gate; no favorable candidate or onset mode could rescue a failure.

## Validation

`PASS_ALL_LOCK_PREANALYSIS_PRIMARY_INDEPENDENT_COMPARATOR_SUFFIX_IMMUTABILITY_BOOTSTRAP_LOO_RECOMPUTED_GENERATION_BLOCK_STORAGE_REGENERATION_AND_HASH_CHECKS`

- `1608` immutable prior files from S01-S18/V1/V2/S19-L01-L04, including the S17 waiver, matched their frozen size and SHA-256 records after execution.
- The pushed repository lock and artifact copies of the preregistration and method lock matched exactly, and the execution worktree was clean.
- Preanalysis replay covered all 200 trajectory/cache identities and all frozen molecular clocks, adjacent-H arrays, and strict `H>0.9` labels.
- Primary materialization, independent row-loop replay, comparator replay, all 3,000 suffix sentinels, 4,096 paired bootstraps, 400 leave-one-out mode/candidate checks, and 16,384 permutation result rows were validated.
- Candidate/matrix cardinality, single-specification scope, threshold absence, storage ceilings, deterministic report regeneration, artifact completeness, and SHA-256 manifests passed.

## Commands, runtime, and dependencies

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l05.py
ruff check src/e01_s19_past_only_recurrence scripts/e01/prepare_s19_l05_lock.py scripts/e01/run_s19_l05.py tests/e01/test_s19_l05.py
python scripts/e01/prepare_s19_l05_lock.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s19_l05.py --workers 8
```

- Repository commit: `de812d023fa80cc642f5f64316f5f972353f39d2` on `eidosoma/groups/42`, pushed before outcome access.
- Authoritative computation: CPU float64; 8 workers; one numerical-library thread per worker; GPU unused.
- Scientific CPU hours: `0.147700`; wall hours: `0.023200`.
- Python `3.13.14`, NumPy `2.4.6`, pandas `2.3.3`, SciPy `1.18.0`, scikit-learn `1.9.0`, PyArrow `24.0.0`.

## Provenance and regeneration

The exact contract is in `preregistration.yaml` and `method_lock.json`; all RNG streams are in `seed_manifest.parquet`; source identities and retained hashes are in `source_snapshot_manifest.json`; trajectory inputs and hashes are in `input_manifest.json`; matrix-level values are in `results.parquet`; controls and robustness are in their named Parquet tables; and the full artifact manifest records every retained file's size and SHA-256. Disposable label and permutation caches remain under `/cache/e01_s19_l05`; compact final evidence is under `/artifacts/research_steps/S19/loops/L05`.

## Caveats, blockers, and interpretation boundary

1. The 100 matrices were already examined in earlier E01 work, and the paper fingerprint was known. L05 is exploratory even though it was outcome-blind after lock.
2. The paper does not uniquely specify a one-sided molecular recurrence algorithm. This implementation may still label gradual drift rather than recurrence to a genuine compositional attractor.
3. Matching 88% occupancy alone was prohibited; it cannot rescue wrong onset, consistency, episode, cutoff, or permutation behavior.
4. A past-only label construction is not itself an early-warning predictor. No emergence feature or held-out outcome was tested.
5. No association, intervention, or causal-control conclusion changes here. S18 and all prior classifications remain immutable.

## Outcome and recommended next action

**Classification:** `EXPLORATORY_DIRECTIONAL_MATCH` with `0` promoted lead(s). The one past-only label did not pass every locked promotion gate and was not promoted.

Stop at the mandatory human-review boundary. The human may choose another explicitly bounded S19 loop, activate an allowed S20 mode, or pause; nothing downstream is active automatically.
