# S19-L02 — Replicator-definition temporal-fingerprint reconstruction

## Concise top summary

- **Research step ID:** S19-L02
- **Completion status:** COMPLETE; mandatory human-review boundary reached; S20 and every later loop remain inactive
- **Artifacts written:** 40 compact L02 evidence/report files plus append-only S19 root-ledger updates
- **Validation result:** PASS_ALL_LOCK_REPLAY_IMMUTABILITY_STORAGE_AND_REGENERATION_CHECKS
- **Outcome classification:** EXPLORATORY_CONSTRAINING_NO_PROMOTABLE_LEAD
- **Caveats or blockers:** No label passed the complete source-grounding and joint-fingerprint promotion gate; the two-loop default is S20 closeout-only. The paper does not uniquely specify clustering, recurrence, threshold, reference, or molecular/generation alignment; completed-run clustering is future-dependent.
- **Recommended next action:** Human review must choose the next program action. Current default: `ACTIVATE_S20_CLOSEOUT_ONLY_TWO_CONSECUTIVE_LOOPS_WITH_NO_PROMOTABLE_LEAD`. Do not begin it automatically.
- **Lay summary:** This loop tested whether the gap between roughly 98% replication in the current adjacent-similarity label and the paper's roughly 88% state could be explained by a genuinely different definition of a replicator. It compared four fixed definitions and judged the whole temporal pattern—not occupancy alone. The analysis did not tune a threshold, generate simulations, or use causal-emergence results to pick a label.


## Frozen question and scientific boundary

The loop asked whether a recurring-attractor or historical GARD definition jointly improves the paper-facing fingerprint—persistence 716, occupancy 0.88, consistency 0.38, and first onset 37—relative to adjacent molecular `H>0.9`, in both simulator candidates. Because Table 1 prints onset as a percentage while its note says molecular steps, raw-step and normalized-onset distances were locked as separate analyses. Neither could replace the other.

This is an exploratory reconstruction on previously studied S13Y matrices. It does not revise S18, adjudicate Figures 3–6 under a new label, establish early warning, or establish causal control. No S19-L01 scientific value was repaired, rerun, reinterpreted, or used.

## Inputs

- Frozen S13Y trajectory manifest: 100 shared matrix identities, 200 candidate-specific trajectories, 100 completed fissions each.
- Candidate 2 and candidate 3 were analyzed separately. Pooling was not used for a primary gate.
- Frozen S13Y adjacent-H and historical technique-1 label arrays supplied exact replay comparators.
- Original v1 paper, S08 label contracts, S13X label implementations, S18 Matrix A, and pinned historical GARD v10 source identity supplied context and provenance.
- New GARD trajectories: **0**. New PhiRL/emergence values: **0**. GPU use: **0**.

## Detailed methods

### Pre-outcome lock and replay gate

The complete layout, four labels, compositional coordinates, distance/linkage, missing-data rules, target-distance formula, seeds, bootstrap, outlier checks, and promotion gates were committed and pushed before new label outcomes were opened. The preanalysis gate then reloaded all 200 trajectory caches and required exact candidate/trajectory identities, selected molecular clocks, adjacent-H float arrays, and strict `H>0.9` labels. Any mismatch would have failed the loop closed.

### Exactly four label families

1. **Adjacent molecular `H>0.9` comparator.** Incoming consecutive-state cosine H on the selected molecular clock. This is ordinary local smoothness, not a global attractor definition.
2. **Dominant recurring-composition centroid.** The frozen S13X completed-run post-fission cosine-component/centroid implementation. It is retrospective and method-dependent; the paper says Euclidean and omits the exact graph and centroid mechanics.
3. **Recurring Euclidean composition cluster.** The frozen S13X completed-run Euclidean silhouette/K-means implementation. It is retrospective and method-dependent; the paper omits K-means, K selection, and tie rules.
4. **Historical technique-1 compotype/non-drift.** The frozen historical GARD v10 adjacent post-fission average-H rule, propagated onto the molecular clock. The initial state remained explicitly ineligible. Interior labels depend on the outgoing neighbor, so this is not an online current-state label.

No `H>0.97` candidate or threshold grid was computed. The earlier 0.97 sensitivity served only as motivation already known before lock; it was not an L02 result or selection option.

### Temporal fingerprint

For every matrix and label, the loop retained occupancy, persistence, raw and normalized first onset, consecutive-label Pearson consistency, entries/exits, episode count and durations, longest episode, state and no-onset status at the 25% cutoff, and post-fission recurrence diagnostics. Full-run labels were explicitly marked retrospective.

The paper-distance score was the root mean square of errors standardized by the paper's control-table plus/minus values. It used persistence, occupancy, consistency, and one onset interpretation. Runs without onset retained null observed onset but received a right-censored score of total clock length (raw) or 1.0 (normalized), preventing missing onsets from improving the score. Undefined consistency was never imputed and failed promotion if fewer than 95/100 runs were defined.

Promotion required, in **both candidates and both onset interpretations**, at least 10% lower distance than adjacent H, a paired matrix-bootstrap 95% upper bound below zero, at least three of four target dimensions closer including consistency or onset, improvement under every leave-one-matrix-out omission, exact replay, adequate defined trajectories, and source-grounded label identity. Occupancy alone could not pass.

## Results

### Candidate-specific full temporal fingerprints

| Cand.   | Label             |   Persistence |   Occupancy |   Consistency |   Onset raw |   Onset norm. |   Episodes |   Longest |   Nonrep. at 25% |   No onset by 25% |   Distance raw |   Distance norm. |
|:--------|:------------------|--------------:|------------:|--------------:|------------:|--------------:|-----------:|----------:|-----------------:|------------------:|---------------:|-----------------:|
| C02     | Adjacent H>.9     |        858.41 |      0.9809 |        0.0713 |       1.1   |        0.0016 |      16.86 |    251.89 |             0.03 |              0    |         3.1646 |           3.1683 |
| C03     | Adjacent H>.9     |        913.78 |      0.9827 |        0.0906 |       0.97  |        0.0014 |      16.3  |    281.16 |             0.02 |              0    |         3.0723 |           3.0756 |
| C02     | Dominant centroid |        212.73 |      0.2872 |        0.8861 |     125.57  |        0.1384 |      13.27 |     68.17 |             0.66 |              0.16 |        10.9416 |          10.8265 |
| C03     | Dominant centroid |        209    |      0.2709 |        0.8872 |     131.2   |        0.138  |      12.84 |     76.7  |             0.78 |              0.15 |        11.2076 |          11.0793 |
| C02     | Euclidean cluster |        545.82 |      0.6254 |        0.9165 |      30.51  |        0.0387 |      13.54 |    191.34 |             0.37 |              0.02 |         6.1795 |           6.2087 |
| C03     | Euclidean cluster |        592.76 |      0.6367 |        0.9277 |      23.86  |        0.0338 |      12.28 |    238.16 |             0.33 |              0.03 |         6.1177 |           6.1445 |
| C02     | Historical T1     |        231.56 |      0.317  |        0.9363 |      85.02  |        0.0855 |       7.92 |     89.02 |             0.67 |              0.07 |        10.5747 |          10.5505 |
| C03     | Historical T1     |        237.82 |      0.3087 |        0.9409 |     117.758 |        0.115  |       7.32 |     97.07 |             0.67 |              0.16 |        10.8171 |          10.6851 |

### Joint fingerprint improvement relative to adjacent H

| Cand.   | Label             | Onset mode   |   Distance gain |   Closer dims | Structure improved   |   Bootstrap low |   Bootstrap high | All LOO improved   |
|:--------|:------------------|:-------------|----------------:|--------------:|:---------------------|----------------:|-----------------:|:-------------------|
| C02     | Dominant centroid | RAW          |         -2.4575 |             0 | False                |          7.0241 |           8.4902 | False              |
| C02     | Dominant centroid | NORMALIZED   |         -2.4171 |             1 | True                 |          6.9127 |           8.3496 | False              |
| C02     | Euclidean cluster | RAW          |         -0.9527 |             1 | True                 |          2.6167 |           3.4564 | False              |
| C02     | Euclidean cluster | NORMALIZED   |         -0.9596 |             1 | True                 |          2.6462 |           3.4827 | False              |
| C02     | Historical T1     | RAW          |         -2.3416 |             0 | False                |          6.7261 |           8.0761 | False              |
| C02     | Historical T1     | NORMALIZED   |         -2.33   |             1 | True                 |          6.6998 |           8.0282 | False              |
| C03     | Dominant centroid | RAW          |         -2.648  |             0 | False                |          7.4138 |           8.8449 | False              |
| C03     | Dominant centroid | NORMALIZED   |         -2.6023 |             1 | True                 |          7.2956 |           8.6341 | False              |
| C03     | Euclidean cluster | RAW          |         -0.9913 |             2 | True                 |          2.6293 |           3.4971 | False              |
| C03     | Euclidean cluster | NORMALIZED   |         -0.9978 |             2 | True                 |          2.6672 |           3.5226 | False              |
| C03     | Historical T1     | RAW          |         -2.5209 |             0 | False                |          7.0813 |           8.424  | False              |
| C03     | Historical T1     | NORMALIZED   |         -2.4741 |             1 | True                 |          6.9835 |           8.2354 | False              |

The complete distributional summaries, raw/normalized distance bootstraps, leave-one-out checks, cutoff measures, episode data, label overlap with exact H, recurrence diagnostics, and cross-candidate comparisons are machine-readable. Directional resemblance was evaluated even where exact numerical agreement failed, but a favorable direction in one candidate could not rescue the other.

### Classification and promotion

Promoted lead count: **0**. Promoted IDs: **none**.

- `MOL_ADJACENT_INCOMING_H900`: POSSIBLE_STABILITY_PROXY, NOT_PROMOTABLE; promoted=false.
- `PF_DOMINANT_COMPONENT_CENTROID_H900`: EXPLORATORY_NON_SUPPORT, RETROSPECTIVE_ONLY_LEAD, METHOD_DEPENDENT_LEAD, AUTHOR_AMBIGUITY_UNRESOLVED, NOT_PROMOTABLE; promoted=false.
- `PF_EUCLIDEAN_KMEANS_DOMINANT`: EXPLORATORY_NON_SUPPORT, RETROSPECTIVE_ONLY_LEAD, METHOD_DEPENDENT_LEAD, AUTHOR_AMBIGUITY_UNRESOLVED, NOT_PROMOTABLE; promoted=false.
- `PF_HISTORICAL_ADJACENT_AVERAGE_H090`: EXPLORATORY_NON_SUPPORT, AUTHOR_AMBIGUITY_UNRESOLVED, NOT_PROMOTABLE; promoted=false.


## Robustness and falsification

- Exact independent re-execution of every label on every trajectory: PASS.
- Frozen adjacent-H and historical-label identity: PASS.
- Candidate 2 and candidate 3 remained separate through all primary calculations: PASS.
- Paired bootstrap unit: catalytic matrix; molecular rows were not treated as replicates.
- Leave-one-matrix-out influence: retained for every alternative label and both onset interpretations.
- Adjacent-H overlap: retained to identify definitions that remain proxies for ordinary local stability.
- New trajectories, emergence, association, prediction, and intervention outcomes used for selection: none.
- Immutable S01–S18/V1/V2/S19-L01 validation: PASS.

## Commands and runtime

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 pytest -q tests/e01/test_s19_l02.py
git commit <pre-outcome L02 lock> && git push origin eidosoma/groups/42
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/prepare_s19_l02_lock.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s19_l02.py --workers 8
```

Scientific CPU time was 0.096595 hours and wall time was 0.069290 hours. Retained and temporary storage remained within the locked ceilings. CPU float64 was authoritative.

## Validation

Overall validation: **PASS_ALL_LOCK_REPLAY_IMMUTABILITY_STORAGE_AND_REGENERATION_CHECKS**. The report was rendered twice from machine-readable results and matched exactly. Required schemas, row counts, label cardinality, candidate cardinality, hashes, storage ceilings, replay, and promotion-limit checks passed.

## Caveats, blockers, and limitations

1. The paper's phrase “most recurring composition” does not identify a unique centroid, medoid, cluster algorithm, threshold, persistence rule, or tie rule. The two cluster implementations are fixed forensic reconstructions, not recovered author code.
2. Completed-run centroid and K-means labels use future observations and are eligible only for retrospective paper-facing interpretation.
3. Historical technique 1 is source-traceable to a public older GARD implementation, not the unavailable target-paper implementation; its outgoing-neighbor term is not cutoff-causal.
4. The paper's onset unit is internally inconsistent. Raw and normalized results remain separate.
5. The target-distance calculation necessarily uses known paper fingerprints and therefore can overfit those fingerprints. Untouched S20 data would be required for any confirmation.
6. No label was evaluated by its association with emergence or by downstream prediction/intervention performance, by design.
7. S18 and S19-L01 classifications remain unchanged. L02 is an additive exploratory record.

## Provenance

- Repository commit fixed before outcomes: `b244d905df31124b74d3169614ab1939ae5d4ebe` on `eidosoma/groups/42`, equal to the pushed remote at access.
- Historical GARD v10: commit `86dff6320d5ae91b4e831471079ff46749b14df9`; no detected repository license; identity/hash only.
- Original paper: arXiv `2607.28250v1`, retained SHA-256 in `input_manifest.json`.
- S13Y trajectory and label hashes: `input_manifest.json` and `preanalysis_replay_evidence.parquet`.
- Complete source relationships and licensing boundaries: root `source_search_ledger.parquet` and L02 `source_snapshot_manifest.json`.

## Recommended next action and mandatory boundary

Return control for human review now. Recommended choice: `ACTIVATE_S20_CLOSEOUT_ONLY_TWO_CONSECUTIVE_LOOPS_WITH_NO_PROMOTABLE_LEAD`. This recommendation is not activation. Do not start S19-L03, S20, E02, author contact, or report-bundle generation automatically.
