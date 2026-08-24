# E01/S19-L08 — Untouched occupancy-mechanism discrimination

## Concise top summary

- **Research step ID:** S19-L08 (`E01-S19-L08-UNTOUCHED-OCCUPANCY-MECHANISM-DISCRIMINATION-v1.0.0`)
- **Completion status:** COMPLETE; frozen at the mandatory post-L08 human-review boundary
- **Artifacts written:** a complete loop package including the preregistration/method lock, 100-unit input and seed manifests, 400 trajectory attempts, 600 temporal fingerprints, episode and mechanistic diagnostics, exactly 4,096-replicate bootstrap evidence, full regeneration, decision gates, hashes, this report, and the one-page handoff
- **Validation result:** PASS — 400/400 complete trajectories, 400/400 exact trajectory replays, 600/600 exact fingerprint replays, aggregate/result replay exact, seed firewall and immutable-prior checks pass, and all scope/runtime/storage/hash checks pass
- **Outcome classification:** `NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA`; S19 vocabulary: `EXPLORATORY_NON_SUPPORT`, `AUTHOR_AMBIGUITY_UNRESOLVED`, `NOT_PROMOTABLE`
- **Caveats or blockers:** neither mechanism is author code; A changes the label object and uses an undocumented molecular projection; B uses an undocumented exposure; the paper's onset units and recurring-attractor semantics remain unresolved; this is exploratory discrimination after L07 selection and cannot confirm, predict, or establish causal control
- **Lay summary:** On 100 wholly new matched matrices, A's boundary-unit occupancy and B's molecular occupancy reproduced the approximate band, but A's locked molecular projection did not: 0.8470/0.8457 fell just below 0.85 (boundary-only 0.8581/0.8608), while B was 0.8750/0.8698. The preregistered joint gate therefore passed 4/6 object-candidate cells and returned `NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA`. This is a scope-specific locked decision, not an author-code identification.
- **Recommended next action:** Mandatory human review. Do not activate L09, S20, E02, author contact, report-bundle generation, emergence, prediction, or intervention work automatically.

## Lay summary

L07 found two ways to turn the approximately 98% label occupancy into approximately 88%: measure inheritance once per fission and project it over the next growth interval, or keep the molecular label but make Poisson updates much larger. L08 tested those exact two ideas on 100 new catalytic matrices without searching or changing anything. A retained its boundary-unit match and B retained its molecular match, but A's separate molecular projection fell marginally outside the band. Occupancy therefore remains measurement-object dependent rather than a reproduced mechanism-wide result. The complete comparison used trajectory length, persistence, onset, consistency, episodes, fission fidelity, mass, overshoot, cross-candidate agreement, and exact replay. Its locked result is `NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA`.

## Frozen question and scope

Can the two L07 occupancy mechanisms independently reproduce `0.88±0.03` in both candidate pipelines on untouched matrices, and does the complete prospectively locked fingerprint favor one without treating occupancy alone as sufficient?

Mechanism A is strict parent-to-selected-daughter `H>0.9` at fission boundaries under the original candidate exposures, reported both as a boundary series and under the already frozen following-interval molecular projection. Mechanism B is strict all-molecular adjacent-incoming `H>0.9` at fixed `h=2.875`. Exactly these two mechanisms were run. There was no threshold, exposure, clock, projection, simulator, label, emergence, prediction, or intervention search.

## Inputs and provenance

- Original paper: local extracted arXiv v1 attachment, SHA-256 recorded in `source_snapshot_manifest.json`.
- Historical GARD lineage: pinned commit `86dff6320d5ae91b4e831471079ff46749b14df9`, retained in cache and not redistributed because no compatible license file was found.
- Untouched L08 input: exactly 100 new catalytic matrices and matched mass-40 distinct-type initial states under the locked root. Exact hash overlap with every discoverable prior matrix, initial state, seed material, seed root, and derived seed was zero.
- Candidate 2: first daughter; A uses `h=0.6031526490073492`, B uses `h=2.875`.
- Candidate 3: random nonempty daughter; A uses `h=0.5613315384859516`, B uses `h=2.875`.
- Both use the frozen trim-new-entrants-to-`nmax` overshoot rule, 100 species, 100 requested fissions, CPU float64, and one numerical-library thread per worker.

## Detailed methods

The complete contract was tested, committed, and pushed before outcomes. A pre-outcome seed firewall generated only inputs and seed identities. A one-matrix/four-simulation benchmark calculated no label and projected the primary plus complete regeneration below the reserved 90 CPU-hour scientific ceiling.

Every attempt was retained. An incomplete or extinct trajectory would have contributed its observed locked prefix when the label object remained defined, while a missing object would have received an explicit null; no unit could be replaced. In fact, 400/400 trajectories completed all 100 fissions.

The independent unit was catalytic matrix. Each candidate and mechanism remained separate. The three noninterchangeable label summaries were A-boundary, A-projected-molecular, and B-molecular. For each, the analysis retained occupancy, persistence, one- and zero-based onset, normalized onset, Pearson consistency, positive/negative episode counts, durations and spacing. Trajectory discriminants included selected-clock/boundary length, parent-daughter similarity, pre/post-fission mass, pre-trim overshoot, max-step terminations, and candidate agreement.

Uncertainty used exactly 4,096 domain-separated PCG64DXSM matrix bootstrap replicates. Raw-onset and normalized-onset paper distances remained separate. A preference required lower distances with paired intervals excluding zero in both candidates and onset modes, at least four of five non-occupancy target wins including persistence and onset, nondegenerate episode topology, completion/mass/max-step/cross-candidate gates, and every integrity check. Source coherence was frozen and reported separately; it could not override numerical results.

## Commands

```text
PYTHONPATH=src:. pytest -q tests/e01/test_s19_l08.py
python -m compileall -q src/e01_s19_untouched_mechanism scripts/e01/run_s19_l08.py
PYTHONPATH=src python scripts/e01/run_s19_l08.py prepare
PYTHONPATH=src python scripts/e01/run_s19_l08.py run --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l08.py regenerate --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l08.py finalize
```

No dependency was installed. Python 3.13.14, NumPy 2.4.6, pandas 2.3.3, SciPy 1.18.0, PyArrow 24.0.0, and Matplotlib 3.11.1 were used.

## Results

### Locked label fingerprints

| Object | Cand. | Length | Persistence | Occupancy | Onset step 1 | Onset normalized | Consistency | Positive episodes | Negative episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A boundary | 2 | 100.00 | 85.81 | 0.858100 | 1.81 | 0.0082 | 0.1320 | 9.70 | 9.38 |
| A boundary | 3 | 100.00 | 86.08 | 0.860800 | 1.97 | 0.0098 | 0.1856 | 9.07 | 8.68 |
| A projected molecular | 2 | 891.56 | 731.95 | 0.846981 | 15.13 | 0.0156 | 0.9098 | 9.70 | 9.38 |
| A projected molecular | 3 | 946.59 | 777.09 | 0.845676 | 19.28 | 0.0187 | 0.9200 | 9.07 | 8.68 |
| B molecular | 2 | 327.48 | 285.99 | 0.874951 | 3.41 | 0.0076 | 0.1047 | 32.70 | 32.81 |
| B molecular | 3 | 329.19 | 285.68 | 0.869845 | 3.54 | 0.0079 | 0.0888 | 34.49 | 34.66 |

Four of six primary occupancy object-candidate cells passed the frozen inclusive `[0.85, 0.91]` band, with 100 defined matrices in every cell. Both A boundary cells and both B molecular cells passed; A's projected molecular means, 0.846981 and 0.845676, fell below the 0.85 floor. Boundary-unit and molecular-projection values remain separate; neither was substituted based on closeness.

### S19-L08 reporting amendment 001

The initial generated prose incorrectly said 6/6 occupancy gates passed even though `occupancy_gate_results.csv`, `decision_gate_results.csv`, and the terminal classification always recorded the correct 4/6 failure. This value-preserving amendment corrects only that narrative inconsistency. No trajectory, label, statistic, gate, decision, or classification changed.

### Simulator and fission discriminants

| Mechanism | Cand. | Selected clock | Boundaries | Parent-daughter H | Post-fission mass | Mean overshoot | Q95 overshoot | Max-step fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A fission-boundary | 2 | 891.56 | 100.00 | 0.9462 | 40.013 | 6.628 | 18.303 | 0.0000 |
| A fission-boundary | 3 | 946.59 | 100.00 | 0.9473 | 39.956 | 6.107 | 16.836 | 0.0000 |
| B high-exposure | 2 | 327.48 | 100.00 | 0.9467 | 40.030 | 38.496 | 111.825 | 0.0000 |
| B high-exposure | 3 | 329.19 | 100.00 | 0.9457 | 39.978 | 40.522 | 108.421 | 0.0000 |

### Paper-distance discrimination

| Mechanism | Candidate | Onset mode | Normalized paper distance |
| --- | --- | --- | ---: |
| A_FISSION_BOUNDARY | CANDIDATE_2 | NORMALIZED_ONSET | 4.02567 |
| B_HIGH_EXPOSURE | CANDIDATE_2 | NORMALIZED_ONSET | 2.54042 |
| A_FISSION_BOUNDARY | CANDIDATE_2 | RAW_ONSET | 3.99909 |
| B_HIGH_EXPOSURE | CANDIDATE_2 | RAW_ONSET | 2.53039 |
| A_FISSION_BOUNDARY | CANDIDATE_3 | NORMALIZED_ONSET | 4.10946 |
| B_HIGH_EXPOSURE | CANDIDATE_3 | NORMALIZED_ONSET | 2.63941 |
| A_FISSION_BOUNDARY | CANDIDATE_3 | RAW_ONSET | 4.07863 |
| B_HIGH_EXPOSURE | CANDIDATE_3 | RAW_ONSET | 2.62944 |

| Candidate | Onset mode | Mean A-minus-B | CI 2.5% | CI 97.5% | P(A lower) |
| --- | --- | ---: | ---: | ---: | ---: |
| CANDIDATE_2 | NORMALIZED_ONSET | 1.48937 | 1.31057 | 1.66162 | 0.0000 |
| CANDIDATE_2 | RAW_ONSET | 1.47116 | 1.30690 | 1.64004 | 0.0000 |
| CANDIDATE_3 | NORMALIZED_ONSET | 1.47385 | 1.29834 | 1.64517 | 0.0000 |
| CANDIDATE_3 | RAW_ONSET | 1.45195 | 1.27161 | 1.62823 | 0.0000 |

![Locked mechanism comparison](mechanism_comparison.png)

**Figure 1.** Untouched candidate-specific means and 95% matrix-bootstrap intervals. Dashed lines show paper control anchors. The panels retain occupancy, persistence, onset, consistency, and the inferred clock-length target separately.

### Paper and source coherence

Mechanism A directly matches the paper's growth-fission inheritance language and the historical generation trace with `H=0.9`, but literal parent-daughter fidelity is not the paper's “most recurring composition,” and the molecular projection is unrecovered. Mechanism B matches the paper's Poisson/molecular-step language and tests a genuinely omitted exposure, but neither paper nor retained source specifies `h=2.875`; it also changes clock length and overshoot. These facts were locked before results and did not override the numerical gates.

## Validation

- Exactly 100 new shared matrix/initial identities; zero prior beta, initial-state, seed-material, root, or derived-seed overlap.
- Exactly 400 attempts and 400 retained trajectory manifests; no replacement.
- Exactly 400/400 independent trajectory replays and 600/600 label/fingerprint replays.
- Aggregate and mechanism-comparison hashes regenerated exactly.
- Exactly 4,096 bootstrap replicates per registered aggregate and contrast.
- S01–S18, V1/V2, and S19-L01–L07 immutable baseline: 1672 files unchanged.
- Runtime: 0.0475 CPU-hours and 0.0118 wall-hours; 0 GPU-hours. Retained and cache storage stayed below their ceilings.
- Repository scientific lock remained the clean pushed commit `975a2cbd1cff4cc09e312bb884a5e3fd86d3a249`.

## Outcome classification and interpretation

The directed decision is **`NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA`**. The decision vocabulary was prospectively locked so that any failure of the joint all-six occupancy gate maps to this token; it does not mean that every individual readout missed the band, because 4/6 passed. Under the existing S19 vocabulary, the result is `EXPLORATORY_NON_SUPPORT`, `AUTHOR_AMBIGUITY_UNRESOLVED`, `NOT_PROMOTABLE`. It is not labelled confirmed, is not promoted to S20, and does not identify author code. The result cannot alter S18 prediction or causal-control classifications.

## Caveats, blockers, and limitations

1. These mechanisms were selected adaptively in L07; L08 is untouched only with respect to its new matrices and fixed comparison.
2. Matching 88% is necessary for this comparison but not proof of the paper's replicator definition.
3. A's following-interval projection and B's exposure value remain author ambiguities.
4. The paper's recurring-attractor description is not exactly either tested rule.
5. The Table 1 first-onset heading and note disagree on units; both analyses remain separate.
6. Simulator evidence is not experimental origin-of-life validation, biological replication, or causal evidence.
7. No authors were contacted; unlicensed public source was not redistributed.

## Artifact and software provenance

Machine-readable evidence includes `trajectory_fingerprints.parquet`, `episode_results.parquet`, `trajectory_diagnostics.parquet`, `results.parquet`, `occupancy_gate_results.csv`, `mechanism_discrimination_results.parquet`, `paired_distance_bootstrap_results.parquet`, `cross_candidate_results.parquet`, `decision_gate_results.csv`, `regeneration_validation.json`, `storage_validation.json`, and `artifact_manifest.json`. Repository-backed code and the full lock are on `eidosoma/groups/42` at `975a2cbd1cff4cc09e312bb884a5e3fd86d3a249`.

## Recommended next action and mandatory boundary

Return to human review. L08 is complete and frozen. No L09, S20, E02, author contact, report-bundle generation, causal-emergence calculation, prediction model, or intervention experiment is active.
