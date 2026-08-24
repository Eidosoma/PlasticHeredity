# S12H Full Results: Candidate 1 Boundary-Clock Revalidation

## Top summary

- **Research step ID:** `E01-S12H-CANDIDATE1-BOUNDARY-CLOCK-REVALIDATION-v1.0.0` (S12H)
- **Completion status:** `STOPPED_FAIL_CLOSED_AT_STAGE1_TIMEBASE_REVALIDATION`; stage 2 was not authorized and S13 was not begun.
- **Artifacts written:** 54 status-bearing files under `/artifacts/research_steps/S12H/`, including the preregistration and method lock, 96-input manifest, 32-row candidate-1 clock audit, upstream gate decision, replay/provenance/immutability evidence, schema-bearing suppressed scientific tables, six stop-state figures, manifests, status JSON, and this report.
- **Validation result:** **Fail closed.** Thirteen of fourteen stage-1 gates passed. The combined aggregate-support gate failed because 2/32 trajectories (6.25%) exceeded 1,314 C1 transitions, above the frozen 5% ceiling. All replay, identity, completion, sample-endpoint, mass, max-step, distance, provenance, runtime, storage, schema, manifest, and immutability checks passed.
- **Outcome classification:** `BOUNDARY_INCLUSIVE_CANDIDATE1_NOT_UPSTREAM_CONFIRMED` (constraining/contradictory). All label and emergence associations are `NOT_EVALUATED`.
- **Caveats or blockers:** Candidate 1 is a post-S12G, boundary-inclusive derivative rather than the original S12FR C0 candidate. The miss is narrow but preregistered and cannot be waived: matrices 7 and 18 had 1,355 and 1,381 C1 transitions. No S12G task cache supplied a scientific result.
- **Recommended next action:** Return for mandatory human review. Do not repair or continue S12H, and keep S13, prediction, MLP work, interventions, estimator repair, and scale-up blocked.

## Lay summary

S12H asked whether recording the selected daughter after every fission would give candidate 1 a coherent endpoint without spoiling its earlier match to the paper's visible time scale. That clock worked mechanically: every trajectory had all 100 daughter boundaries, all three paper-visible sample endpoints were covered, and replay and provenance were exact. It nevertheless missed one unchanged upstream criterion. Two of 32 trajectories extended beyond the paper-axis ceiling, while the preregistered rule allowed at most one. Because every upstream gate had to pass, S12H stopped before computing a single replicator label or information-theory value. The proposed fresh three-candidate ensemble therefore did not run.

## Frozen question and decision rule

The frozen question was whether the raw S12FR candidate-1 dynamics remain a paper-compatible time-base candidate after replacing only its analysis clock:

- original identity: `S12F-CANDIDATE-01`, `h=0.5081160391061118`, first-daughter continuation, retained overshoot, `C0_BATCH_UPDATES_ONLY`;
- proposed derivative: `S12H-CANDIDATE-01-C1-DERIVATIVE-v1.0.0`, with exactly the same dynamics, matrices, initial states, seeds, and raw trajectories, but `C1_SELECTED_DAUGHTER_RETAINED` after every fission.

The design froze an all-gates rule before raw-trajectory access. Stage 2 could start only after the derivative passed the original S12FR confirmation envelope and a second candidate lock was committed and pushed. Any failed upstream gate required permanent termination without labels or information theory.

## Inputs and provenance

- Governing files: `AGENTS.md`, `FULL_PLAN.md`, `RESEARCH_PLAN.md`, `input-attachments/MANIFEST.json`, and the attachment sidecar.
- Paper: arXiv `2607.28250v1`, PDF SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- Historical GARD source: commit `86dff6320d5ae91b4e831471079ff46749b14df9`.
- IIGR source: commit `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`.
- PhiRL source: commit `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`.
- Safe lattice: SHA-256 `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`; no pickle was loaded.
- Source validation: S12C 14/14 equivalence rows and S12D 40/40 emergence-identity rows passed before stage 1.
- Raw inputs: exactly 32 locked candidate-1 S12FR confirmation pickles (6,828,700 bytes). The broader manifest retained all 96 candidate inputs and verified 32/32 shared catalytic-matrix and initial-state identities.
- Immutable baseline: 514 S01-S12G artifact files, all 96 S12FR raw caches, and all 950 forbidden S12G task-cache files were hash-frozen. Final revalidation found zero changes.
- Pre-outcome repository lock: commit `9cceb86f42d8251915e03088c8ebf68adee57c2a`, pushed to `origin/eidosoma/groups/42` before raw access.

Two pre-outcome operational issues are retained in `preoutcome_issue_ledger.json`: a JSON scalar-serialization defect and a direct-script import-path defect. Both occurred before raw trajectory, label, information-theory, or S12G cache-payload access; neither changed the scientific method or simulator. Each repair was committed and pushed before the method lock was regenerated.

## Detailed methods

For every raw trajectory, S12H selected the initial state, every Poisson batch-update state, and every selected post-fission daughter state. Thus

`T_Phi(C1) = total_batch_updates + 100`

for each 100-fission lineage. Every generation endpoint had to be the actual stored `post_fission` observation, never an imputation, duplicate, skipped generation, or special zero-update fallback. Each selected sequence was extracted twice and hashed for exact clock replay.

The unchanged S12FR gate required:

1. exactly 32 locked inputs and at least 31/32 100-fission completions;
2. at least two of the three digitized endpoints (800, 800, 1,000) inside the C1 q05–q95 interval;
3. aggregate support with observed maximum at least 1,090, q95 no greater than 1,314, and at most 5% of trajectories above 1,314;
4. median selected-daughter mass between 35 and 45;
5. max-step termination fraction at most 0.05;
6. the original S12F/S12FR distance no greater than 1.0;
7. no synthetic duplicate clock records;
8. exact state cardinality, raw-cache hashes, candidate identity, S12FR replay, C1 extraction replay, seed/provenance, runtime, and storage.

No label, CLR representation, partition, PhiID atom, emergence scalar, local Phi-r value, association, spike statistic, or S12G cache payload was accessed in stage 1.

## Results

### Clock summary

| Quantity | Result | Gate |
| --- | ---: | --- |
| Complete 100-fission lineages | 32/32 | Pass (at least 31) |
| C1 q05 | 531.65 | Descriptive |
| C1 median | 912.0 | Descriptive |
| C1 q95 | 1,282.95 | Pass (at most 1,314) |
| C1 maximum | 1,381 | Pass lower-support requirement (at least 1,090) |
| Paper sample endpoints inside q05–q95 | 3/3 | Pass (at least 2) |
| Trajectories above 1,314 | 2/32 = 0.0625 | **Fail** (at most 0.05) |
| Median post-fission mass | 42.0 | Pass (35–45) |
| q95 overshoot summary | 29.25 | Diagnostic |
| Max-step termination fraction | 0.0 | Pass (at most 0.05) |
| Confirmation distance | 0.86475 | Pass (at most 1.0) |

Candidate 1's uniform C1 clock adds exactly 100 transitions to every original C0 trajectory. Matrices 7 and 18 changed from C0 lengths 1,255 and 1,281 to C1 lengths 1,355 and 1,381, respectively. This makes the fraction beyond 1,314 equal to 6.25%. With 32 units, the 5% rule permits at most one exceedance; two therefore fail the frozen aggregate-support criterion. No threshold was rounded, relaxed, or reinterpreted after access.

### Gate matrix

| Gate | Result |
| --- | --- |
| Exactly 32 locked trajectories | Pass |
| At least 31/32 complete | Pass |
| At least 2/3 sample endpoints covered | Pass (3/3) |
| Aggregate support compatible | **Fail** |
| Median post-fission mass 35–45 | Pass |
| Max-step fraction at most 0.05 | Pass |
| Confirmation distance at most 1.0 | Pass |
| Uniform C1 without synthetic duplicate | Pass |
| State cardinality | Pass (32/32; 100 endpoints each) |
| Exact replay | Pass (32/32 S12FR and 32/32 C1 extraction) |
| Cache hashes | Pass (32/32) |
| Candidate identity | Pass (32/32) |
| Seed and provenance | Pass |
| Runtime and storage | Pass |

Outcome: `BOUNDARY_INCLUSIVE_CANDIDATE1_NOT_UPSTREAM_CONFIRMED`.

### Conditional stage 2

Stage 2 was not authorized. No derivative candidate lock was committed; `derivative_candidate_lock_validation.json` records `NOT_AUTHORIZED_STAGE1_FAILED`. Every scientific output table is schema-bearing and has zero rows. The six required figures are explicit stop-state figures. No candidate-specific or ensemble association, replicator-versus-drift contrast, spike analysis, metric-identity comparison, or full-versus-prefix result exists, and all such questions remain `NOT_EVALUATED`.

## Validation

- Candidate-1 cache hashes, raw candidate identities, and S12FR replay flags: 32/32 pass.
- C1 selected-sequence replay hashes: 32/32 pass.
- C1 cardinality: 100 actual post-fission endpoints per trajectory and `updates + 101` selected observations in all 32.
- Finite/discrete/forbidden-nonfinite/seed divergences: 0/0/0/0.
- Prior immutability: 514 prior artifact files, 96 raw caches, and 950 S12G cache files unchanged.
- S12G scientific cache reuse: 0 files.
- New GARD trajectories: 0.
- Label or information-theory outcomes opened: none.
- Schema validation: all 21 declared tables present with required columns; 32 upstream rows, one failure row, and zero scientific rows.
- Artifact validation: all 54 required artifacts present; retained size below 1 MiB and below the 30 GiB ceiling.
- Stage-1 runtime: 3.340829 wall-seconds and 3.332916 CPU-seconds; GPU use 0.

## Commands and dependencies

```bash
PYTHONPATH=src python -m pytest -q \
  tests/e01/test_s12h_candidate1_boundary_clock_revalidation.py \
  tests/e01/test_s12g_frozen_timebase_ensemble.py
python -m ruff check \
  src/e01_boundary_clock_revalidation \
  scripts/e01/freeze_s12h_preregistration.py \
  scripts/e01/run_s12h_candidate1_boundary_clock_revalidation.py \
  tests/e01/test_s12h_candidate1_boundary_clock_revalidation.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src \
  python scripts/e01/freeze_s12h_preregistration.py \
  --design-commit 9cceb86f42d8251915e03088c8ebf68adee57c2a
ARTIFACTS_DIR=/artifacts PYTHONPATH=src \
  python scripts/e01/run_s12h_candidate1_boundary_clock_revalidation.py \
  --stage revalidate
```

Focused validation passed 7/7 tests and Ruff passed. No dependency was installed. The runtime was Python 3.13.14, NumPy 2.4.6, pandas 2.3.3, PyArrow 24.0.0, and SciPy 1.18.0 on Linux 6.12.95/glibc 2.39. CPU float64 was authoritative; BLAS and OpenMP thread counts were frozen to one. Stage 2's six-worker source campaign was never launched.

## Artifacts and storage contract

`candidate1_clock_revalidation.parquet` is the lossless 32-row upstream audit. `candidate1_timebase_confirmation.json` is the canonical gate decision. `classification.json`, `failure_ledger.csv`, and `status.json` preserve the fail-closed outcome. All scientific tables retain the frozen S12G schemas but are empty because their phase was prohibited. `artifact_manifest.json` records bytes and SHA-256 for every collectible artifact except itself. No large intermediate, source checkout, trajectory cache, or environment was copied into the artifact directory.

## Caveats, blockers, and interpretation

- This derivative was authorized after S12G exposed the C0 endpoint problem. Upstream revalidation limits but cannot erase that post-outcome flexibility.
- The aggregate miss is caused by a discrete sample fraction: 2/32 exceeds the 5% ceiling, while 1/32 would pass. That makes the miss narrow but not optional.
- The result rejects only this newly defined boundary-inclusive candidate under the frozen paper-timebase envelope. It does not alter S12FR's original C0 confirmation or S12G's fail-closed result.
- No evidence was produced about self-replication labels, emergence, local Phi-r, retrospective association, prospective survival, prediction, intervention, or author implementation identity.
- S01–S12G, including S12F `SIMULATOR_IDENTIFICATION_FAILED`, S12FR `NONIDENTIFIABLE_TIMEBASE_ENSEMBLE`, and S12G `S12G_VALIDATION_FAILED_CLOSED`, remain byte-for-byte unchanged.

## Recommended next action

Return for mandatory human review. Under the explicit stop rule, do not repair S12H, do not run the three-candidate scientific ensemble, and do not begin S13 or any prediction, MLP, intervention, estimator-repair, simulation, or scale-up step automatically. S13 remains `BLOCKED_PENDING_S12H_HUMAN_REVIEW`.
