# E01/S19-L10 — MATLAB-compatible recurring-attractor reconstruction

## Concise top summary

- **Research step ID:** `S19-L10` (`E01-S19-L10-MATLAB-COMPATIBLE-RECURRING-ATTRACTOR-RECONSTRUCTION-v1.0.0`).
- **Completion status:** COMPLETE; frozen at the mandatory post-L10 human-review boundary.
- **Artifacts written:** all required method-lock, fixture, source, seed, trajectory, clustering, label, fingerprint, comparator, control, bootstrap, validation, classification, report, manifest, and 12 figure artifacts under `/artifacts/research_steps/S19/loops/L10`; root S19 ledgers and handoff were appended.
- **Validation result:** PASS_ALL_FIXTURES_SEED_FIREWALL_400_TRAJECTORY_REPLAYS_14_RESULT_TABLE_REPLAYS_IMMUTABILITY_SOURCE_SCOPE_RUNTIME_STORAGE_AND_HASH_GATES — 16/16 fixture checks, 400/400 trajectory replays, 14/14 scientific-table regenerations, zero-overlap seed firewall, immutable-prior, scope, runtime, storage, source-hash, and artifact-integrity gates passed. The first regeneration attempt is preserved: it passed 400/400 trajectories and 13/14 table hashes, while the sole fingerprint-table failure had zero differing cells after diagnostic column alignment. Following explicit human authorization, technical repair 001 fixed only schema-order canonicalization and the complete fresh rerun passed 400/400 trajectories and 14/14 tables without a scientific value change.
- **Outcome classification:** `RECURRING_ATTRACTOR_LABEL_NOT_RECONSTRUCTED`; S19 vocabulary: `EXPLORATORY_NON_SUPPORT`, `AUTHOR_AMBIGUITY_UNRESOLVED`, `NOT_PROMOTABLE`; promoted lead IDs: none.
- **Caveats or blockers:** Both labels use complete-run post-fission compositions and are retrospective; exact author code and the paper's onset/dispersion semantics remain unavailable. No L10 result establishes author-code identity, prediction, intervention efficacy, or causal control. The initial column-order replay failure and its explicitly authorized value-preserving repair are retained as provenance; they do not strengthen the negative scientific result.
- **Lay summary:** The MATLAB/scikit-learn mismatch was resolved prospectively and safely: MATLAB-compatible singleton scores were retained for software selection, while all-singleton or tied-largest outcomes could not become biological labels. On 100 new matched matrices, mean molecular occupancy was R1 0.3980/0.4039 and R2 0.2716/0.2819 for candidates 2/3. The complete locked fingerprint—not occupancy alone—produced `RECURRING_ATTRACTOR_LABEL_NOT_RECONSTRUCTED`.
- **Recommended next action:** Mandatory human review; do not activate another loop, S20, E02, author contact, report generation, emergence, prediction, or intervention work automatically.

## Lay summary

L09 did not answer the recurring-attractor question because Python's silhouette implementation rejected a cluster assignment that MATLAB documents as valid. L10 fixed only that compatibility boundary before creating any new outcome, tested the fix on mandatory fixtures, generated a wholly new seed-firewalled dataset, and evaluated both the historical and paper-Euclidean reconstructions. Crucially, software is allowed to select an all-singleton solution, but the separate scientific recurrence gate refuses to call it a recurring compotype. The final classification reflects availability, full temporal fingerprints, negative controls, cross-candidate behavior, and exact regeneration—not target proximity alone.

## Frozen question and evidentiary boundary

The question was whether either of exactly two already specified pipelines could form a scientifically valid dominant recurring composition and jointly improve the paper-facing control fingerprints. R1 used the pinned historical GARD non-drift/compotype lineage with a clean-room MATLAB-compatible singleton silhouette. R2 used the unchanged L09 paper-Euclidean specification. Both directly labelled molecular states by strict `H(x_t,c*)>0.9`; neither projected a boundary label. Causal emergence, prediction, intervention outcomes, and target-guided threshold search were absent. All labels are full-run retrospective constructions.

## Inputs and provenance

- 100 new catalytic matrices and matched initial states, all identities frozen before labels.
- Four retained trajectory groups: candidate 2 and candidate 3 at their original exposures (primary), plus both continuation rules at fixed `h=2.875` (comparator only).
- Historical GARD commit `86dff6320d5ae91b4e831471079ff46749b14df9`, the original paper/Figure 1, references 63–65, official MathWorks silhouette documentation, and scikit-learn 1.9.0 documentation were hashed; unlicensed historical source remains cache-only.
- Exact source/file identities are in `source_snapshot_manifest.json`; exact inputs and seeds are in `input_manifest.json`, `input_units.parquet`, `seed_manifest.parquet`, and `seed_firewall.json`.

## Methods

### MATLAB-compatible implementation

For singleton observations, `matlab_compatible_silhouette` returns exactly 1. For a nonsingleton, it computes within-cluster mean distance `a`, nearest-other-cluster mean distance `b`, and `(b-a)/max(a,b)`, with the locked identical-distance value 0. R1 uses float64 cosine distance, permits `k=n`, preserves the historical k=1 mean-H path, tests k=1–10 with ten deterministic replicas, and preserves the historical early stop. R2 retains Euclidean Lloyd k-means, k=1–10, ten deterministic replicas, fixed initialization, convergence, tie, and silhouette semantics.

After software k-selection, a separate gate requires a unique largest cluster with at least two assigned boundaries. Every-singleton and tied-largest fits remain explicit status-bearing units and emit no molecular label. Undefined consistency and incomplete/extinct units remain undefined rather than imputed or replaced.

### Measurements and statistics

The catalytic matrix was the independent unit. Candidate 2 and candidate 3 were separate. Primary molecular metrics included occupancy, persistence, raw zero-/one-based and normalized onset, onset generation, Pearson consecutive-label consistency, transitions, episode topology, pre-onset time, 10/20/25/33% no-onset availability, boundary diagnostics, and parent-daughter H. Both sample SD and SE were reported without choosing the closer dispersion interpretation. Exactly 4,096 domain-separated matrix bootstrap replicates were used. Random-reference, second-largest-cluster, and time-permutation controls were frozen; Holm correction was applied across the two pipelines within candidate/control/outcome.

## Results

| Pipeline | Candidate | Defined | Eligible recurrence | Selected k=n | All-singleton | Occupancy | Persistence | Consistency | First onset (1-based) | No onset through 25% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 2 | 81/100 | 81 | 15 | 15 | 0.3980 | 266.47 | 0.8609 | 106.09 | 0.111 |
| R1 | 3 | 83/100 | 83 | 12 | 12 | 0.4039 | 282.23 | 0.8807 | 95.08 | 0.120 |
| R2 | 2 | 99/100 | 99 | 0 | 0 | 0.2716 | 169.07 | 0.7698 | 127.15 | 0.394 |
| R2 | 3 | 99/100 | 99 | 0 | 0 | 0.2819 | 185.27 | 0.7895 | 132.93 | 0.444 |

The per-candidate promotion-gate counts were {"R1-C2": "6/12", "R1-C3": "5/12", "R2-C2": "6/12", "R2-C3": "6/12"}. Recurrence status counts are preserved machine-readably; their compact mapping is `{"('R1_MATLAB_COMPATIBLE_HISTORICAL_DOMINANT_COMPTYPE_H090', 'CANDIDATE_2', 'ELIGIBLE')": 81, "('R1_MATLAB_COMPATIBLE_HISTORICAL_DOMINANT_COMPTYPE_H090', 'CANDIDATE_2', 'NO_NONDRIFT_COMPOSITIONS')": 1, "('R1_MATLAB_COMPATIBLE_HISTORICAL_DOMINANT_COMPTYPE_H090', 'CANDIDATE_2', 'NO_RECURRING_COMPTYPE')": 15, "('R1_MATLAB_COMPATIBLE_HISTORICAL_DOMINANT_COMPTYPE_H090', 'CANDIDATE_2', 'NO_UNIQUE_RECURRING_COMPTYPE')": 3, "('R1_MATLAB_COMPATIBLE_HISTORICAL_DOMINANT_COMPTYPE_H090', 'CANDIDATE_3', 'ELIGIBLE')": 83, "('R1_MATLAB_COMPATIBLE_HISTORICAL_DOMINANT_COMPTYPE_H090', 'CANDIDATE_3', 'NO_NONDRIFT_COMPOSITIONS')": 3, "('R1_MATLAB_COMPATIBLE_HISTORICAL_DOMINANT_COMPTYPE_H090', 'CANDIDATE_3', 'NO_RECURRING_COMPTYPE')": 12, "('R1_MATLAB_COMPATIBLE_HISTORICAL_DOMINANT_COMPTYPE_H090', 'CANDIDATE_3', 'NO_UNIQUE_RECURRING_COMPTYPE')": 2, "('R2_PAPER_EUCLIDEAN_DOMINANT_ATTRACTOR_H090', 'CANDIDATE_2', 'ELIGIBLE')": 99, "('R2_PAPER_EUCLIDEAN_DOMINANT_ATTRACTOR_H090', 'CANDIDATE_2', 'NO_UNIQUE_RECURRING_COMPTYPE')": 1, "('R2_PAPER_EUCLIDEAN_DOMINANT_ATTRACTOR_H090', 'CANDIDATE_3', 'ELIGIBLE')": 99, "('R2_PAPER_EUCLIDEAN_DOMINANT_ATTRACTOR_H090', 'CANDIDATE_3', 'NO_UNIQUE_RECURRING_COMPTYPE')": 1}`. Of 36 aggregate negative-control tests, 28 passed their direction, uncertainty, and multiplicity contract. Pipeline-specific classifications and every gate are in `classification.json` and `scientific_gate_results.parquet`.

### Paper-target interpretation

The targets were kept distinct: occupancy 0.88, persistence 716, consistency 0.38, raw first onset 37, normalized onset as an unresolved companion, episode topology, and trajectory length. `paper_target_comparison.csv` reports raw and standardized differences, sample SD, SE, and bootstrap intervals. `complete_fingerprint_distances.parquet` reports both raw-onset and normalized-onset distances and counts dimensions improved over every frozen comparator. No smallest-distance pipeline was selected unless semantic, availability, control, cross-candidate, and validation gates also passed.

## Illustrated results

![MATLAB and scikit-learn singleton behavior](figures/figure_01_matlab_vs_sklearn_singleton.png)

*Figure 1. Prospectively validated singleton-silhouette distinction. MATLAB-compatible values are one; scikit-learn treats k=n as outside its valid domain.*

![Selected k and cluster sizes](figures/figure_02_selected_k_cluster_sizes.png)

*Figure 2. Selected-k and dominant-cluster-size distributions across both pipelines and candidates.*

![Recurrence statuses](figures/figure_03_singleton_no_recurring_rates.png)

*Figure 3. Eligibility, all-singleton/nonrecurrence, tie, and other explicit status frequencies.*

![Representative recurring clusters](figures/figure_04_representative_dominant_clusters.png)

*Figure 4. Diagnostic two-dimensional projections of representative post-fission composition sets; color indicates direct dominant-centroid membership.*

![Similarity to dominant centroid](figures/figure_05_h_to_dominant_over_time.png)

*Figure 5. Molecular-time H to the completed-run dominant centroid, with the fixed strict 0.9 threshold.*

![Adjacent and attractor labels](figures/figure_06_adjacent_vs_recurring_labels.png)

*Figure 6. The frozen adjacent molecular label versus the direct recurring-attractor label on representative trajectories.*

![Occupancy and persistence](figures/figure_07_occupancy_persistence.png)

*Figure 7. Candidate-specific occupancy and persistence; dashed lines are paper-facing targets.*

![Consistency and onset](figures/figure_08_consistency_onset.png)

*Figure 8. Candidate-specific consecutive-label consistency and raw one-based onset.*

![Episode topology](figures/figure_09_episode_topology_preonset.png)

*Figure 9. Positive and negative episode durations; quarter-cutoff availability is tabulated in the main results.*

![Negative controls](figures/figure_10_negative_controls.png)

*Figure 10. Registered random-reference, second-cluster, and permuted-time controls; color encodes the Holm-aware directional gate.*

![Cross-candidate agreement](figures/figure_11_candidate_agreement.png)

*Figure 11. Candidate-2 versus candidate-3 means for occupancy and onset.*

![Decision matrix](figures/figure_12_final_fingerprint_decision_matrix.png)

*Figure 12. Complete preregistered scientific gate matrix; green means passed and red means failed.*

## Validation

- Mandatory fixtures F01–F12: 16/16 checks passed before scientific trajectory generation.
- Opaque ten-matrix benchmark passed before label access and projected total use below ceilings.
- Exactly 100 shared inputs, 400 trajectory attempts, no replacements, and no seed/input overlap.
- Exact regeneration: all 400 trajectory identities/hashes and all 14 authoritative result tables matched.
- Frozen historical source, paper, Figure 1, and documentation hashes passed; prior S01–S18/V1/V2/L01–L09 artifacts passed the immutable baseline.
- Total CPU 0.348413 h, wall 0.082148 h, GPU 0 h; runtime and storage ceilings passed with at least 10% validation reserve.
- Required-artifact, schema, hash, and report regeneration checks passed.

## Commands, software, and reproduction

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l10.py
ruff check src/e01_s19_matlab_attractor scripts/e01/run_s19_l10.py tests/e01/test_s19_l10.py
python scripts/e01/run_s19_l10.py prepare
git commit ... && git push origin eidosoma/groups/42
python scripts/e01/run_s19_l10.py generate --workers 8
python scripts/e01/run_s19_l10.py analyze --workers 8
python scripts/e01/run_s19_l10.py regenerate --workers 8
python scripts/e01/run_s19_l10.py finalize
```

Python 3.13.14, NumPy 2.4.6, SciPy 1.18.0, pandas 2.3.3, scikit-learn 1.9.0, and PyArrow 24.0.0 were recorded. Eight processes and one numerical-library thread per worker were used; CPU float64 was authoritative and no GPU was used.

## Caveats, failures, and limitations

- L10 is an adaptive exploratory continuation after prior label investigations. Even an untouched result cannot retroactively erase specification multiplicity.
- R1 is source-lineage compatible, not author code. The exact target MATLAB release, RNG, and author modifications remain unknown.
- R2 follows paper-Euclidean wording but remains a reconstruction where the paper is incomplete.
- Completed-run cluster discovery uses future observations and cannot support early warning, future-suffix independence, online intervention, or causal control.
- Exact H membership is deterministic conditional on a selected centroid; it does not demonstrate incremental information from causal emergence.
- Paper Table 1 onset units and the printed `±` identity remain unresolved. Raw and normalized onset and SD/SE were not substituted based on target proximity.
- Status-bearing ineligible units were retained, not silently dropped or reassigned.

## Provenance and artifact map

Repository code is pinned by `implementation_lock.json` and the pushed commit recorded in `run_release_gate.json`. Source provenance is in `source_snapshot_manifest.json`; seeds/inputs/trajectories are in their manifests; all numerical results are in Parquet/CSV tables; validation records include regeneration, immutability, source, scope, runtime, storage, and artifact integrity; `artifact_manifest.json` hashes the complete compact loop package. Cached trajectory payloads remain under `/cache/e01_s19_l10` and are represented by hashes rather than copied into artifacts.


## Authorized technical regeneration repair

The first complete regeneration reproduced all 400 trajectories and 13 of 14 authoritative table hashes. `label_fingerprint_results.parquet` had the same 400 rows, the same column set, and zero differing cells after a diagnostic alignment, but its raw column order differed because the first completed parallel worker established DataFrame insertion order. The locked comparison had canonicalized rows but not columns, so it correctly raised instead of silently relaxing the gate.

The human then explicitly directed: “if it was a technical problem, fix and rerun.” Repair `S19-L10-TECHNICAL-REPAIR-001` was frozen, committed, and pushed before rerun. It preserved the failed validation, comparison, trajectory-replay, and runtime artifacts under `*_failed_attempt_001.*`; left the scientific runner, core, config, seeds, trajectories, estimands, labels, controls, bootstrap, and gates unchanged; and added only lexicographic column canonicalization to the table comparator. A fresh cache reran all 400 trajectories and all 14 tables. The repaired comparison passed 400/400 trajectory and 14/14 table gates with exact dtypes and cells after schema alignment and reported no scientific value change. This is a disclosed post-outcome technical repair, not extra scientific specification search.

The combined runtime counts both regeneration attempts. `technical_repair_001.json`, `technical_repair_lock_001.json`, `technical_repair_release_gate_001.json`, `regeneration_validation_failed_attempt_001.json`, `regeneration_validation.json`, and `technical_repair_completion_001.json` provide the audit chain.

## Outcome and next action

The locked decision is `RECURRING_ATTRACTOR_LABEL_NOT_RECONSTRUCTED`. This remains exploratory, does not identify author code, and does not change S18's prospective-prediction or causal-control conclusions. Mandatory human review; do not activate another loop, S20, E02, author contact, report generation, emergence, prediction, or intervention work automatically.
