# S19-L04 — Cross-generation recurrence membership

## Concise top summary

- **Research step ID:** S19-L04
- **Completion status:** COMPLETE; mandatory human-review boundary reached; L05, S20, E02, author contact, and report-bundle generation remain inactive
- **Artifacts written:** complete compact L04 preregistration, method/label/bundle/specification locks, exact replay, label values, recurrence evidence, temporal fingerprints, bootstrap, leave-one-out, generation-block permutation, validation, provenance, status, manifest, and append-only S19 ledger evidence
- **Validation result:** PASS_ALL_LOCK_REPLAY_IMMUTABILITY_BOOTSTRAP_LOO_GENERATION_BLOCK_STORAGE_REGENERATION_AND_HASH_CHECKS
- **Outcome classification:** `EXPLORATORY_DIRECTIONAL_MATCH`; promoted leads: 0 (none)
- **Caveats or blockers:** The rule is a completed-run retrospective reconstruction on previously studied matrices, not recovered author code. Occupancy alone could not promote it, and no prospective, predictive, intervention, or causal claim is eligible.
- **Recommended next action:** Human review must choose the next bounded program action; do not activate L05 or S20 automatically.
- **Lay summary:** This loop tested a middle ground between “the current state resembles the immediately previous state” and “the state belongs to one dominant compotype.” A state counted as replicating only if it resembled a nonadjacent state from another fission generation at strict `H>0.9`. The analysis judged the entire temporal pattern, not the requested 88% occupancy alone.

## Frozen question and nonduplication

L02 directly classified molecular rows by adjacent similarity, one centroid, one K-means cluster, or historical local non-drift. L03 projected one modal boundary compotype. Neither allowed multiple recurring regions while requiring evidence from another generation. L04 locked exactly that singleton hypothesis before opening its outcomes. No second interpretation, threshold, centroid, cluster, modal reference, projection, or alignment was available for selection.

## Inputs

- Frozen S13Y: 100 shared catalytic matrices, 200 complete candidate-specific trajectories, and their frozen adjacent-H arrays and labels.
- Candidate 2 and candidate 3 remained separate; pooling was not used for any gate.
- Original arXiv v1 paper, frozen S08/S13X label context, and frozen L01–L03 comparison evidence.
- New GARD trajectories: **0**. New PhiRL/emergence values: **0**. Prediction/intervention outcomes used: **0**. GPU use: **0**.

## Detailed methods

For every selected-clock molecular state in positive-numbered generation `g`, the analysis L1-closed the 100-component count vector and computed historical cosine H to every completed-run state. A state was positive only if at least one reference state met all three conditions: strict `H>0.9`, a different positive-numbered generation, and absolute selected-sequence separation greater than one. Thus same-generation similarity and an immediate cross-generation neighbor alone could not establish recurrence. Each matching generation counted once regardless of how many states matched. The initial generation-zero state was retained but ineligible.

The relation was evaluated symmetrically over the completed run, so it is explicitly future-dependent and retrospective. Adjacent molecular `H>0.9` was the only comparator. Catalytic matrix was the inferential unit. The frozen joint paper-distance was the root mean square of four deviations scaled by the paper's declared control dispersions: persistence 716±198, occupancy 0.88±0.03, consistency 0.38±0.06, and raw onset 37±27 or normalized onset 0.37±0.27 as two separate analyses.

Robustness comprised a paired 4,096-replicate matrix bootstrap, every leave-one-matrix-out omission, cross-candidate agreement, exact independent label replay, and 4,096 generation-block permutations. The permutation preserved each block's internal order, completed-run membership, and occupancy while shuffling the 100 whole growth-fission blocks within every trajectory; it therefore tested whether the observed temporal arrangement was more paper-like than arbitrary generation order. Its preregistered pass rule required observed paper-distance below the null 2.5th percentile in both modes and candidates.

## Results

### Candidate-specific temporal fingerprints

| Candidate   | Label            |   Persistence |   Occupancy |   Consistency |   Onset idx0 |   Onset step1 |   Onset norm |   Episodes |   Longest |   Nonrep@25% |   No onset@25% |
|:------------|:-----------------|--------------:|------------:|--------------:|-------------:|--------------:|-------------:|-----------:|----------:|-------------:|---------------:|
| C02         | Adjacent H>.9    |      858.4100 |      0.9809 |        0.0713 |       1.1000 |        2.1000 |       0.0016 |    16.8600 |  251.8900 |       0.0300 |         0.0000 |
| C03         | Adjacent H>.9    |      913.7800 |      0.9827 |        0.0906 |       0.9700 |        1.9700 |       0.0014 |    16.3000 |  281.1600 |       0.0200 |         0.0000 |
| C02         | Cross-generation |      796.2400 |      0.9197 |        0.5416 |       6.2300 |        7.2300 |       0.0075 |    30.5400 |  228.3000 |       0.1100 |         0.0000 |
| C03         | Cross-generation |      847.4400 |      0.9216 |        0.5750 |       6.4500 |        7.4500 |       0.0072 |    30.1200 |  242.3800 |       0.0400 |         0.0000 |

### Cross-generation recurrence descriptors

| Candidate   |   Mean other generations |   Median other generations |   Max other generations |   Fraction >=2 |   Fraction >=5 |   Generation-pair density |   Immediate-only rows |
|:------------|-------------------------:|---------------------------:|------------------------:|---------------:|---------------:|--------------------------:|----------------------:|
| C02         |                  17.3844 |                    15.7250 |                 38.1000 |         0.6568 |         0.4444 |                    0.2273 |               13.2100 |
| C03         |                  17.0125 |                    15.3250 |                 36.9700 |         0.6519 |         0.4335 |                    0.2230 |               12.3600 |

### Joint paper-fingerprint comparison

| candidateId       | onsetMode   |   paperDistance |   comparatorDistance |   distanceDifferenceCandidateMinusComparator |   distanceImprovementFraction |   closerDimensionCount | occupancyCloser   |
|:------------------|:------------|----------------:|---------------------:|---------------------------------------------:|------------------------------:|-----------------------:|:------------------|
| S12F-CANDIDATE-02 | RAW         |          1.6181 |               3.1646 |                                      -1.5465 |                        0.4887 |                      4 | True              |
| S12F-CANDIDATE-02 | NORMALIZED  |          1.6566 |               3.1683 |                                      -1.5117 |                        0.4771 |                      4 | True              |
| S12F-CANDIDATE-03 | RAW         |          1.8845 |               3.0723 |                                      -1.1878 |                        0.3866 |                      4 | True              |
| S12F-CANDIDATE-03 | NORMALIZED  |          1.9190 |               3.0756 |                                      -1.1566 |                        0.3761 |                      4 | True              |

### Paired matrix-bootstrap differences, cross-generation minus adjacent

| candidateId       | labelId                              | metric                    |   replicates |   meanDifference |   lower95 |   upper95 |
|:------------------|:-------------------------------------|:--------------------------|-------------:|-----------------:|----------:|----------:|
| S12F-CANDIDATE-02 | MOL_CROSS_GENERATION_RECURRENCE_H900 | occupancy                 |         4096 |          -0.0612 |   -0.0710 |   -0.0518 |
| S12F-CANDIDATE-02 | MOL_CROSS_GENERATION_RECURRENCE_H900 | consistency               |         4096 |           0.4704 |    0.4316 |    0.5064 |
| S12F-CANDIDATE-02 | MOL_CROSS_GENERATION_RECURRENCE_H900 | firstOnsetRawScore        |         4096 |           5.1444 |    4.2000 |    6.1200 |
| S12F-CANDIDATE-02 | MOL_CROSS_GENERATION_RECURRENCE_H900 | firstOnsetNormalizedScore |         4096 |           0.0059 |    0.0048 |    0.0072 |
| S12F-CANDIDATE-03 | MOL_CROSS_GENERATION_RECURRENCE_H900 | occupancy                 |         4096 |          -0.0611 |   -0.0703 |   -0.0519 |
| S12F-CANDIDATE-03 | MOL_CROSS_GENERATION_RECURRENCE_H900 | consistency               |         4096 |           0.4846 |    0.4367 |    0.5269 |
| S12F-CANDIDATE-03 | MOL_CROSS_GENERATION_RECURRENCE_H900 | firstOnsetRawScore        |         4096 |           5.4743 |    4.6200 |    6.3100 |
| S12F-CANDIDATE-03 | MOL_CROSS_GENERATION_RECURRENCE_H900 | firstOnsetNormalizedScore |         4096 |           0.0057 |    0.0048 |    0.0066 |

### Generation-block permutation negative control

| candidateId       | labelId                              | onsetMode   | controlId                          |   replicates |   observedPaperDistance |   nullLower2_5 |   nullMedian |   nullUpper97_5 |   lowerTailP | negativeControlPassed   | occupancyInvariantByConstruction   | completedRunMembershipInvariantByConstruction   |
|:------------------|:-------------------------------------|:------------|:-----------------------------------|-------------:|------------------------:|---------------:|-------------:|----------------:|-------------:|:------------------------|:-----------------------------------|:------------------------------------------------|
| S12F-CANDIDATE-02 | MOL_CROSS_GENERATION_RECURRENCE_H900 | RAW         | GENERATION_BLOCK_ORDER_PERMUTATION |         4096 |                  1.6181 |         1.4702 |       1.4819 |          1.4989 |       1.0000 | False                   | True                               | True                                            |
| S12F-CANDIDATE-02 | MOL_CROSS_GENERATION_RECURRENCE_H900 | NORMALIZED  | GENERATION_BLOCK_ORDER_PERMUTATION |         4096 |                  1.6566 |         1.4788 |       1.4905 |          1.5074 |       1.0000 | False                   | True                               | True                                            |
| S12F-CANDIDATE-03 | MOL_CROSS_GENERATION_RECURRENCE_H900 | RAW         | GENERATION_BLOCK_ORDER_PERMUTATION |         4096 |                  1.8845 |         1.6958 |       1.7081 |          1.7254 |       1.0000 | False                   | True                               | True                                            |
| S12F-CANDIDATE-03 | MOL_CROSS_GENERATION_RECURRENCE_H900 | NORMALIZED  | GENERATION_BLOCK_ORDER_PERMUTATION |         4096 |                  1.9190 |         1.7033 |       1.7156 |          1.7329 |       1.0000 | False                   | True                               | True                                            |

### Label overlap with adjacent H

| candidateId       | labelId                              | baselineId                 |   commonEligibleCount |   accuracy |   jaccard |   mismatchFraction |   structuralPositiveAdjacentNegative |   structuralNegativeAdjacentPositive |
|:------------------|:-------------------------------------|:---------------------------|----------------------:|-----------:|----------:|-------------------:|-------------------------------------:|-------------------------------------:|
| S12F-CANDIDATE-02 | MOL_CROSS_GENERATION_RECURRENCE_H900 | MOL_ADJACENT_INCOMING_H900 |                 87487 |     0.8966 |    0.8963 |             0.1034 |                                 1441 |                                 7604 |
| S12F-CANDIDATE-03 | MOL_CROSS_GENERATION_RECURRENCE_H900 | MOL_ADJACENT_INCOMING_H900 |                 92948 |     0.8985 |    0.8983 |             0.1015 |                                 1426 |                                 8005 |

The singleton cross-generation rule did not pass the complete joint-fingerprint promotion gate; it is not eligible for S20 confirmation under this lock. All directional comparisons were assessed in both candidates and both onset interpretations. A favorable occupancy result could not override persistence, onset, consistency, episode, cutoff, recurrence, negative-control, uncertainty, or simulator disagreement.

## Validation

- Immutable S01–S18/V1/V2/L01–L03 baseline: PASS across 1,556 files.
- Pre-outcome clean pushed repository lock and exact S13Y identity/cache/clock/H/label replay: PASS.
- Independent two-pass replay of both labels on all 200 trajectories: PASS.
- Frozen adjacent-H comparator reproduction: PASS.
- Matrix bootstrap, leave-one-out, generation-block control, cross-candidate, schema/cardinality, storage, deterministic-report, and artifact-hash checks: PASS.
- No trajectory, PhiRL value, emergence value, prediction model, or intervention outcome was generated.

## Commands, dependencies, and runtime

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l04.py
python -m ruff check src/e01_s19_cross_generation_recurrence scripts/e01/prepare_s19_l04_lock.py scripts/e01/run_s19_l04.py tests/e01/test_s19_l04.py
git commit <pre-outcome L04 lock> && git push origin eidosoma/groups/42
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src python scripts/e01/prepare_s19_l04_lock.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src python scripts/e01/run_s19_l04.py --workers 8
```

CPU float64 was authoritative. The loop used 0.049649 scientific CPU-hours and 0.032090 wall-hours, with eight workers and one numerical-library thread per worker; GPU use was zero.

## Caveats, blockers, and limitations

1. The paper supports across-generation recurrence in general but does not state this exact all-molecular-state, nonadjacent existential rule.
2. Completed-run symmetric membership uses future observations and can support only retrospective paper-facing reconstruction.
3. The same 100 matrices have been studied in earlier loops, so known paper fingerprints create adaptive-overfitting risk; only untouched S20 data could confirm a promoted lead.
4. Strict `H>0.9` remains a similarity proxy and may classify slow cross-generation drift as recurrence even after excluding immediate neighbors.
5. Raw and normalized onset remain separate because the paper's Table 1 unit is internally inconsistent.
6. No downstream emergence association, prediction, or intervention result was recalculated under this exploratory label.

## Provenance

- Pushed pre-outcome repository commit: `9d8f43dca4e5d0420e86e6ed931d9e211a16ab7c` on `eidosoma/groups/42`.
- Original paper: arXiv `2607.28250v1`; hashes are recorded in `input_manifest.json` and `source_snapshot_manifest.json`.
- Frozen S13Y identities and exact replay: `preanalysis_replay_evidence.parquet` and `frozen_comparator_replay.parquet`.
- Exact formula and seeds: repository source, `method_lock.json`, `label_registry.yaml`, `specification_ledger.parquet`, and `seed_manifest.parquet`.

## Recommended next action and mandatory boundary

Return control for human review now. L04 issued no confirmatory verdict. L05, S20, E02, author contact, and report-bundle generation remain inactive unless a later explicit human decision authorizes one bounded action.
