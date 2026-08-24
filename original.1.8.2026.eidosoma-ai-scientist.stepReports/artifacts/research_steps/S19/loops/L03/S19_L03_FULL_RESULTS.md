# S19-L03 — Boundary compotype projection and recurrence activation

## Concise top summary

- **Research step ID:** S19-L03
- **Completion status:** COMPLETE; mandatory human-review boundary reached; S19-L04, S20, and all downstream work remain inactive
- **Artifacts written:** 48 compact L03 preregistration, label, boundary, fingerprint, robustness, replay, validation, provenance, status, and handoff files plus append-only S19 root-ledger updates
- **Validation result:** PASS_ALL_LOCK_REPLAY_IMMUTABILITY_STORAGE_AND_REGENERATION_CHECKS_PLUS_VALUE_PRESERVING_REPORTING_AMENDMENT_001
- **Outcome classification:** EXPLORATORY_NON_SUPPORT; NOT_PROMOTABLE
- **Caveats or blockers:** The modal reference is a completed-run retrospective reconstruction, not recovered author code. Matching occupancy alone was prohibited, and every prior result remains unchanged.
- **Recommended next action:** Human review must select any next bounded S19 theme or another program action; do not begin it automatically.
- **Lay summary:** This loop asked whether the paper's replicator state may be assigned only at fission or generation boundaries and then spread across the intervening molecular steps. It fixed one `H>0.9` recurring-boundary rule and compared backfilling, activation after recurrence, incoming/outgoing projection, and pre-/post-fission boundary choices. Every boundary label was far too sparse (about 23%–29%, not 88%), began much later, and was much more autocorrelated than the paper fingerprint. None was promotable. The loop did not tune the threshold or use causal emergence to choose a label.

## Frozen question and nonduplication decision

L02 did not fully test this ambiguity. Its centroid and K-means labels classified every molecular observation directly, while its historical branch projected only a local adjacent non-drift state. L03 instead identified one modal compotype among 100 boundary compositions and projected boundary membership onto molecular time. The underlying strict-`H>0.9` maximum-neighbor medoid, tie rule, minimum recurrence of two, four structural candidates, and all statistical gates were pushed before outcomes.

## Inputs

- Frozen S13Y: 100 shared matrices, candidate 2 and candidate 3 kept separate, 200 complete trajectories.
- Original arXiv v1 paper and frozen S08/S13X/S18/L01/L02 context.
- Pinned historical GARD `tgs_agard_v10.m`, `tgs_nondrift.m`, and `getcomposometime_v10.m` source identities.
- New GARD trajectories: **0**. New PhiRL/emergence values: **0**. Prediction/intervention outcomes used: **0**. GPU use: **0**.

## Detailed methods

For each trajectory, boundary compositions were L1 closed. A boundary state became the modal reference when it had the most other boundary states at strict historical cosine `H>0.9`, including itself; ties went to the earliest generation and boundary index. Fewer than two members meant no recurrent compotype. The five locked labels were the frozen adjacent molecular comparator plus:

1. post-fission modal membership backfilled to incoming boundary-ending intervals;
2. the same label activated only from its second occurrence, incoming aligned;
3. the activated label projected to outgoing boundary-starting intervals;
4. the activated label built from the historical pre-fission generation-end substrate, incoming aligned.

The incoming/outgoing pair changes only interval alignment. The backfill/activation pair changes only recurrence activation. The post-fission/generation-end pair changes only boundary substrate. Full-run reference selection makes every structural label retrospective even when activation itself is forward ordered.

Each trajectory retained persistence, occupancy, raw and normalized onset, Pearson consecutive-label consistency, entries/exits, episode structure, recurrence frequency/span, and 25% cutoff status. Catalytic matrix was the bootstrap and leave-one-out unit; molecular rows were never treated as independent replicates. Paper-distance used persistence 716, occupancy 0.88, consistency 0.38, and raw onset 37 or normalized onset 0.37 as two separate analyses.

## Results

### Candidate-specific temporal fingerprints

| Cand.   | Label                 |   Persist. |   Occup. |   Consist. |   Onset idx0 |   Onset step1 |   Onset norm. |   Episodes |   Longest |   Nonrep.@25% |   No onset by 25% |
|:--------|:----------------------|-----------:|---------:|-----------:|-------------:|--------------:|--------------:|-----------:|----------:|--------------:|------------------:|
| C02     | Adjacent H>.9         |   858.4100 |   0.9809 |     0.0713 |       1.1000 |        2.1000 |        0.0016 |    16.8600 |  251.8900 |        0.0300 |            0.0000 |
| C03     | Adjacent H>.9         |   913.7800 |   0.9827 |     0.0906 |       0.9700 |        1.9700 |        0.0014 |    16.3000 |  281.1600 |        0.0200 |            0.0000 |
| C02     | PF backfill/incoming  |   170.6100 |   0.2425 |     0.9383 |     161.1800 |      162.1800 |        0.1710 |     5.9400 |   73.5300 |        0.7600 |            0.2600 |
| C03     | PF backfill/incoming  |   170.6100 |   0.2349 |     0.9357 |     158.9900 |      159.9900 |        0.1616 |     5.4900 |   78.4600 |        0.7800 |            0.2100 |
| C02     | PF activated/incoming |   163.7800 |   0.2346 |     0.9395 |     195.2100 |      196.2100 |        0.2059 |     5.6500 |   71.7000 |        0.7600 |            0.3000 |
| C03     | PF activated/incoming |   163.4800 |   0.2270 |     0.9371 |     194.1100 |      195.1100 |        0.1936 |     5.2800 |   76.0000 |        0.8100 |            0.2700 |
| C02     | PF activated/outgoing |   178.3000 |   0.2529 |     0.9440 |     201.2400 |      202.2400 |        0.2126 |     5.6500 |   74.5500 |        0.7300 |            0.3100 |
| C03     | PF activated/outgoing |   179.4600 |   0.2460 |     0.9425 |     200.5400 |      201.5400 |        0.2005 |     5.2800 |   78.7500 |        0.8000 |            0.2700 |
| C02     | GE activated/incoming |   212.8500 |   0.2892 |     0.9483 |     169.9700 |      170.9700 |        0.1851 |     6.0000 |   91.0900 |        0.6800 |            0.2600 |
| C03     | GE activated/incoming |   218.9800 |   0.2864 |     0.9496 |     182.7800 |      183.7800 |        0.1851 |     5.8100 |  104.3300 |        0.7300 |            0.2700 |

`Onset idx0` and `Onset step1` preserve the raw-index ambiguity explicitly; the normalized column is a separate analysis. The structural boundary family did create genuine pre-onset eligibility—21%–31% of runs had no onset by the quarter cutoff—but it overcorrected the comparator. Occupancy fell from 98.1%–98.3% to 22.7%–28.9%, persistence fell from 858–914 to 163–219, and consistency rose from 0.071–0.091 to 0.936–0.950 rather than approaching the paper targets of 88%, 716, and 0.38. Raw onset moved from about 1 to 159–201, beyond the paper's approximately 37; normalized onset moved partway toward the separately preserved 0.37 interpretation. Candidate 2 and candidate 3 agreed closely on this wrong joint fingerprint.

### Episode and recurring-boundary descriptors

| Cand.   | Label                 |   Entries |   Exits |   Mean dur. |   Median dur. |   Boundary freq. |   Activated span |
|:--------|:----------------------|----------:|--------:|------------:|--------------:|-----------------:|-----------------:|
| C02     | Adjacent H>.9         |   16.8600 | 15.9700 |     79.2762 |       49.6900 |          n/a     |          n/a     |
| C03     | Adjacent H>.9         |   16.3000 | 15.4000 |    102.0275 |       70.7600 |          n/a     |          n/a     |
| C02     | PF backfill/incoming  |    5.9400 |  5.6400 |     40.9031 |       36.9650 |         27.6800 |           0.6927 |
| C03     | PF backfill/incoming  |    5.4900 |  5.2300 |     42.8486 |       36.6300 |         26.6200 |           0.6600 |
| C02     | PF activated/incoming |    5.6500 |  5.3500 |     40.4345 |       36.8150 |         27.6800 |           0.6577 |
| C03     | PF activated/incoming |    5.2800 |  5.0200 |     41.6862 |       35.3250 |         26.6200 |           0.6278 |
| C02     | PF activated/outgoing |    5.6500 |  5.3500 |     42.6950 |       39.2500 |         27.6800 |           0.6577 |
| C03     | PF activated/outgoing |    5.2800 |  5.0200 |     44.0633 |       37.9250 |         26.6200 |           0.6278 |
| C02     | GE activated/incoming |    6.0000 |  5.6000 |     53.3493 |       48.3650 |         33.2900 |           0.7206 |
| C03     | GE activated/incoming |    5.8100 |  5.5000 |     58.5697 |       50.3150 |         32.3100 |           0.6885 |

Boundary frequency is the number of strict-`H>0.9` members of the locked modal reference; activated span is the normalized molecular-time span from the first to last positive projected post-fission position. Exact per-run values and reference generations are in `fingerprint_results.parquet`, `boundary_reference_results.parquet`, and `boundary_membership_results.parquet`.

### Joint paper-fingerprint comparison

| candidateId       | labelId                                 | onsetMode   |   paperDistance |   distanceImprovementFraction |   closerDimensionCount | occupancyCloser   |
|:------------------|:----------------------------------------|:------------|----------------:|------------------------------:|-----------------------:|:------------------|
| S12F-CANDIDATE-02 | PF_MODAL_MEDOID_BACKFILL_INCOMING_H900  | RAW         |         11.9040 |                       -2.7616 |                      0 | False             |
| S12F-CANDIDATE-02 | PF_MODAL_MEDOID_BACKFILL_INCOMING_H900  | NORMALIZED  |         11.6855 |                       -2.6883 |                      1 | False             |
| S12F-CANDIDATE-02 | PF_MODAL_MEDOID_ACTIVATED_INCOMING_H900 | RAW         |         12.1645 |                       -2.8440 |                      0 | False             |
| S12F-CANDIDATE-02 | PF_MODAL_MEDOID_ACTIVATED_INCOMING_H900 | NORMALIZED  |         11.8103 |                       -2.7277 |                      1 | False             |
| S12F-CANDIDATE-02 | PF_MODAL_MEDOID_ACTIVATED_OUTGOING_H900 | RAW         |         11.9342 |                       -2.7712 |                      0 | False             |
| S12F-CANDIDATE-02 | PF_MODAL_MEDOID_ACTIVATED_OUTGOING_H900 | NORMALIZED  |         11.5438 |                       -2.6435 |                      1 | False             |
| S12F-CANDIDATE-02 | GE_MODAL_MEDOID_ACTIVATED_INCOMING_H900 | RAW         |         11.2721 |                       -2.5620 |                      0 | False             |
| S12F-CANDIDATE-02 | GE_MODAL_MEDOID_ACTIVATED_INCOMING_H900 | NORMALIZED  |         11.0052 |                       -2.4735 |                      1 | False             |
| S12F-CANDIDATE-03 | PF_MODAL_MEDOID_BACKFILL_INCOMING_H900  | RAW         |         12.0019 |                       -2.9065 |                      0 | False             |
| S12F-CANDIDATE-03 | PF_MODAL_MEDOID_BACKFILL_INCOMING_H900  | NORMALIZED  |         11.7937 |                       -2.8346 |                      1 | False             |
| S12F-CANDIDATE-03 | PF_MODAL_MEDOID_ACTIVATED_INCOMING_H900 | RAW         |         12.2648 |                       -2.9921 |                      0 | False             |
| S12F-CANDIDATE-03 | PF_MODAL_MEDOID_ACTIVATED_INCOMING_H900 | NORMALIZED  |         11.9192 |                       -2.8754 |                      1 | False             |
| S12F-CANDIDATE-03 | PF_MODAL_MEDOID_ACTIVATED_OUTGOING_H900 | RAW         |         12.0271 |                       -2.9147 |                      0 | False             |
| S12F-CANDIDATE-03 | PF_MODAL_MEDOID_ACTIVATED_OUTGOING_H900 | NORMALIZED  |         11.6437 |                       -2.7858 |                      1 | False             |
| S12F-CANDIDATE-03 | GE_MODAL_MEDOID_ACTIVATED_INCOMING_H900 | RAW         |         11.3697 |                       -2.7008 |                      0 | False             |
| S12F-CANDIDATE-03 | GE_MODAL_MEDOID_ACTIVATED_INCOMING_H900 | NORMALIZED  |         11.0499 |                       -2.5927 |                      1 | False             |

### Isolated structural contrasts

| candidateId       | contrastId                                          | metric             |   meanDifference |   lower95 |   upper95 |
|:------------------|:----------------------------------------------------|:-------------------|-----------------:|----------:|----------:|
| S12F-CANDIDATE-02 | BACKFILL_MINUS_ACTIVATED_INCOMING                   | occupancy          |           0.0079 |    0.0073 |    0.0086 |
| S12F-CANDIDATE-02 | BACKFILL_MINUS_ACTIVATED_INCOMING                   | consistency        |          -0.0013 |   -0.0037 |    0.0008 |
| S12F-CANDIDATE-02 | BACKFILL_MINUS_ACTIVATED_INCOMING                   | firstOnsetRawScore |         -34.0300 |  -51.0963 |  -20.6438 |
| S12F-CANDIDATE-02 | OUTGOING_MINUS_INCOMING_ACTIVATED                   | occupancy          |           0.0183 |    0.0136 |    0.0237 |
| S12F-CANDIDATE-02 | OUTGOING_MINUS_INCOMING_ACTIVATED                   | consistency        |           0.0045 |    0.0022 |    0.0070 |
| S12F-CANDIDATE-02 | OUTGOING_MINUS_INCOMING_ACTIVATED                   | firstOnsetRawScore |           6.0300 |    5.3500 |    6.7100 |
| S12F-CANDIDATE-02 | GENERATION_END_MINUS_POSTFISSION_ACTIVATED_INCOMING | occupancy          |           0.0546 |    0.0455 |    0.0646 |
| S12F-CANDIDATE-02 | GENERATION_END_MINUS_POSTFISSION_ACTIVATED_INCOMING | consistency        |           0.0088 |    0.0047 |    0.0129 |
| S12F-CANDIDATE-02 | GENERATION_END_MINUS_POSTFISSION_ACTIVATED_INCOMING | firstOnsetRawScore |         -25.2400 |  -53.3663 |    2.1162 |
| S12F-CANDIDATE-03 | BACKFILL_MINUS_ACTIVATED_INCOMING                   | occupancy          |           0.0079 |    0.0072 |    0.0087 |
| S12F-CANDIDATE-03 | BACKFILL_MINUS_ACTIVATED_INCOMING                   | consistency        |          -0.0014 |   -0.0052 |    0.0019 |
| S12F-CANDIDATE-03 | BACKFILL_MINUS_ACTIVATED_INCOMING                   | firstOnsetRawScore |         -35.1200 |  -61.2438 |  -16.1113 |
| S12F-CANDIDATE-03 | OUTGOING_MINUS_INCOMING_ACTIVATED                   | occupancy          |           0.0190 |    0.0134 |    0.0248 |
| S12F-CANDIDATE-03 | OUTGOING_MINUS_INCOMING_ACTIVATED                   | consistency        |           0.0054 |    0.0026 |    0.0085 |
| S12F-CANDIDATE-03 | OUTGOING_MINUS_INCOMING_ACTIVATED                   | firstOnsetRawScore |           6.4300 |    5.8200 |    7.0500 |
| S12F-CANDIDATE-03 | GENERATION_END_MINUS_POSTFISSION_ACTIVATED_INCOMING | occupancy          |           0.0594 |    0.0492 |    0.0710 |
| S12F-CANDIDATE-03 | GENERATION_END_MINUS_POSTFISSION_ACTIVATED_INCOMING | consistency        |           0.0125 |    0.0073 |    0.0182 |
| S12F-CANDIDATE-03 | GENERATION_END_MINUS_POSTFISSION_ACTIVATED_INCOMING | firstOnsetRawScore |         -11.3300 |  -43.9663 |   19.1925 |

Promoted leads: **0** (none). Directional resemblance, when present, remains exploratory and retrospective. An occupancy move toward 0.88 cannot override disagreement in persistence, onset, consistency, episodes, recurrence, cutoff eligibility, uncertainty, or the other simulator candidate.

Here, no boundary candidate even moved occupancy toward 0.88: every structural occupancy error exceeded the adjacent comparator's error in both candidates. All four candidates were therefore classified `EXPLORATORY_NON_SUPPORT`, `RETROSPECTIVE_ONLY_LEAD`, `METHOD_DEPENDENT_LEAD`, `AUTHOR_AMBIGUITY_UNRESOLVED`, and `NOT_PROMOTABLE`.

## Robustness, falsification, and validation

- Exact independent replay of every one of the five labels on all 200 trajectories: PASS.
- Frozen adjacent-H and `H>0.9` comparator identity: PASS.
- Immutable S01–S18/V1/V2/L01/L02 baseline: PASS.
- Paired 4,096-replicate matrix bootstrap, leave-one-matrix-out influence, cross-candidate agreement, adjacent-H overlap, and historical-nondrift overlap: retained in machine-readable form.
- Candidate 2 and candidate 3 stayed separate for every primary gate; pooling was not used.
- Report regeneration, schema/cardinality, artifact hash, compute, and storage checks: PASS.
- Reporting amendment 001 changed only vocabulary compliance and exposure of already-computed fields; hashes of all scientific result tables, labels, gates, and classifications remained unchanged: PASS.

## Commands and runtime

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l03.py
python -m ruff check src/e01_s19_boundary_compotype scripts/e01/prepare_s19_l03_lock.py scripts/e01/run_s19_l03.py tests/e01/test_s19_l03.py
git commit <pre-outcome L03 lock> && git push origin eidosoma/groups/42
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src python scripts/e01/prepare_s19_l03_lock.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src python scripts/e01/run_s19_l03.py --workers 8
```

Scientific CPU time was 0.041060 hours and wall time was 0.029065 hours. CPU float64 was authoritative.

## Caveats, blockers, and limitations

1. The paper does not identify a unique modal reference, boundary substrate, activation point, or projection alignment.
2. Historical public GARD stores pre-fission generation-end traces, but it is not the unavailable target-paper code.
3. The maximum-neighbor medoid is a precise paper/source-informed reconstruction, not author-code identity.
4. Completed-run reference selection uses future observations, so no structural L03 result is eligible as early-warning or online-control evidence.
5. These matrices were previously studied; all L03 selection and inference is exploratory and requires untouched confirmation for any promoted paper-facing lead.
6. No emergence association, prediction, or intervention result was recalculated under these labels.
7. The inherited `postFissionEpisodeCount`/`sameReferenceReentryCount` fields treat post-fission rows separated on the molecular clock as noncontiguous. They are retained for provenance but are not used as authoritative recurrence evidence here; the locked modal boundary frequency, membership table, and activated span are authoritative.

## Provenance

- Pushed pre-outcome repository commit: `9e278ac0a7366939f2520c8ec3f5f23f62f7d368` on `eidosoma/groups/42`.
- Original paper: arXiv `2607.28250v1`; hash retained in `input_manifest.json`.
- Historical GARD commit: `86dff6320d5ae91b4e831471079ff46749b14df9`; source identities and license boundary retained in `source_snapshot_manifest.json`.
- Frozen S13Y trajectory/cache identities and exact replay: `input_manifest.json` and `preanalysis_replay_evidence.parquet`.
- Value-preserving reporting amendment and frozen scientific hashes: `reporting_amendment_001.json`; expanded deterministic summary: `temporal_fingerprint_extended_summary.csv`.

## Recommended next action and mandatory boundary

Return control for human review. The human continuation override permits future bounded S19 loops but authorizes none automatically. S19-L04, S20, E02, author contact, and report-bundle generation remain inactive.
