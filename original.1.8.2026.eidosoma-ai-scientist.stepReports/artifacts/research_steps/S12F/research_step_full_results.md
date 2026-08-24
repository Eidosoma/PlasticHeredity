# S12F full results — Latent time-base inference

## Top summary

- **Research step ID:** `E01-S12F-LATENT-TIMEBASE-INFERENCE-v1.0.0`.
- **Completion status:** `COMPLETED_FAIL_CLOSED_AT_DEVELOPMENT_REPLAY_GATE`; S12F stopped before posterior inference, candidate locking, confirmation, S12G, S13, labels, emergence, or interventions.
- **Artifacts written:** all 39 preregistered required paths, six figures, and one reporting-only finalization record under `/artifacts/research_steps/S12F`; `artifact_manifest.json` freezes every non-self artifact.
- **Validation result:** `PASS_THROUGH_BENCHMARK_THEN_TERMINAL_EXACT_REPLAY_GATE_FAILED` — figure extraction, pushed preregistration, source/input identities, read-only clock cardinality, prior/cache immutability, and 16/16 benchmark replay passed; the global 2,048-pair development replay gate failed.
- **Outcome classification:** `SIMULATOR_IDENTIFICATION_FAILED` (constraining/contradictory operational result).
- **Caveats or blockers:** At least one round-1 pair returned false, but the exact count and pair identity were not retained or inspected after the mandated stop. Therefore this is not evidence that fixed/adaptive exposure fails scientifically.
- **Lay summary:** Adding post-fission states alone did not explain the paper’s time axis. A small benchmark showed that lowering the Poisson exposure can move trajectory lengths into the paper-like range, but the formal inverse problem had to stop when its exact-repeat safeguard failed. No exposure was selected.
- **Recommended next action:** Mandatory human review. Keep S13 blocked; do not repair or rerun S12F, begin S12G, calculate labels/emergence, or continue automatically.

## Frozen question and boundary

S12F asked whether observation-clock accounting, a common fixed exposure `h`, or the single conditional adaptive gross-event exposure can recover the paper-visible upstream time base. It was frozen as `PAPER_AS_DATA_SIMULATOR_IDENTIFICATION`. Self-replication labels, clustering, source-defined emergence, corrected local Phi-r, prediction, spikes, interventions, S12G, and S13 were prohibited. S12E remains `TIME_BASE_MISMATCH_CONFIRMED` for its five original candidates and clock.

## Lay summary

The sample traces in Figure 2 end at about 800, 800, and 1,000 molecular steps, while the aggregate trace visibly extends to roughly 1,100. Recounting the already generated S12E states with daughter boundaries raised K1–K3 medians by only 100 steps, still too short. A one-matrix operational benchmark suggested exposure near 0.25 can produce roughly 1,100–1,350 batch updates, whereas exposure 0.75 produces roughly 400–540. That benchmark is not a posterior. The preregistered ABC campaign executed its first round, but its global exact-replay comparator returned false for at least one pair. The stop rule was enforced before any distances or candidates were inspected.

## Inputs and provenance

- Original arXiv v1 PDF: `/cache/e01_s03/downloads/paper-2607.28250v1.pdf`, SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- Rendered Figure 2 raster: SHA-256 `0e4aac507ccf6e10ced31edd6d7e5ba8c876d9d0c8d420b145dfc27c7d040778`.
- S12E artifact manifest: SHA-256 `ebcc731ba9e3b3543e4748b1cdc844e512d561d66bbfc579e0f088d236b8accc`.
- Historical GARD/IIGR/PhiRL commits: `86dff6320d5ae91b4e831471079ff46749b14df9`, `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`, and `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`.
- S12C safe lattice SHA-256: `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`; it was not loaded or executed in S12F.
- Method/target lock commit: `cf5b27b370a2d8d12e6867034d6ec8f4f96b3fc7`, pushed before new simulation access.
- Prior evidence: 434 artifact files and 6107 S12E cache files checked with zero changes/missing.
- No dataset or previous-artifact mount was present. Authors were not contacted.

## Detailed methods

### Phase 0 — paper as data

The original 926×569 Figure-2 raster was calibrated from panel-specific tick centroids by affine least squares. A second calibration used an independently rendered 300-DPI PDF page and manual grid reading. The methods agreed within the larger of two original-raster pixels or one percent of each x range. The sample endpoints were frozen as 800±8, 800±8, and 1,000±10. The aggregate terminal was retained as 1,090–1,120, its display upper boundary as 1,300–1,314, and a late visible discontinuity as 980–1,070. The Table-1 ratio `716/0.88 = 813.636...` was secondary only, with the endpoint-propagated descriptive interval 569.23–1,075.29.

### Phase 1 — read-only observation clocks

For each of the 72 S12E K1–K3 trajectories, C0 was `sum U`, C1 was `sum U + 100`, and forensic C2 was `sum U + 200`. C2 was never primary-eligible because it introduces an unmaterialized duplicated pre-fission boundary. C3 and C4 could not be reconstructed because S12E did not retain the sampled join/loss vectors. Exact state-cardinality and prior replay evidence passed for C0–C2.

| engineId                         | clockId                          |   q05TPhi |   medianTPhi |   q95TPhi |   maximumTPhi |   sampleEndpointsInsideQ05Q95 | aggregateCompatible   | materializedNaturally   | clockOnlyGatePassed   |
|:---------------------------------|:---------------------------------|----------:|-------------:|----------:|--------------:|------------------------------:|:----------------------|:------------------------|:----------------------|
| K1_PAPER_POISSON_RANDOM_NONEMPTY | C0_BATCH_UPDATES_ONLY            |      96.7 |        440.5 |    620.8  |           652 |                             0 | False                 | True                    | False                 |
| K1_PAPER_POISSON_RANDOM_NONEMPTY | C1_SELECTED_DAUGHTER_RETAINED    |     196.7 |        540.5 |    720.8  |           752 |                             0 | False                 | True                    | False                 |
| K1_PAPER_POISSON_RANDOM_NONEMPTY | C2_EXPLICIT_PRE_AND_POST_FISSION |     296.7 |        640.5 |    820.8  |           852 |                             2 | False                 | False                   | False                 |
| K2_PAPER_POISSON_FIRST_DAUGHTER  | C0_BATCH_UPDATES_ONLY            |      92.4 |        434.5 |    612.85 |           628 |                             0 | False                 | True                    | False                 |
| K2_PAPER_POISSON_FIRST_DAUGHTER  | C1_SELECTED_DAUGHTER_RETAINED    |     192.4 |        534.5 |    712.85 |           728 |                             0 | False                 | True                    | False                 |
| K2_PAPER_POISSON_FIRST_DAUGHTER  | C2_EXPLICIT_PRE_AND_POST_FISSION |     292.4 |        634.5 |    812.85 |           828 |                             2 | False                 | False                   | False                 |
| K3_PAPER_POISSON_RANDOM_LITERAL  | C0_BATCH_UPDATES_ONLY            |      92.2 |        443.5 |    613    |           623 |                             0 | False                 | True                    | False                 |
| K3_PAPER_POISSON_RANDOM_LITERAL  | C1_SELECTED_DAUGHTER_RETAINED    |     192.2 |        543.5 |    713    |           723 |                             0 | False                 | True                    | False                 |
| K3_PAPER_POISSON_RANDOM_LITERAL  | C2_EXPLICIT_PRE_AND_POST_FISSION |     292.2 |        643.5 |    813    |           823 |                             2 | False                 | False                   | False                 |

No admissible clock-only branch passed: C0/C1 placed zero of the three sample endpoints inside Q05–Q95 and missed aggregate support. C2 placed the duplicate 800 endpoint twice inside K1–K3 ranges but still missed aggregate support and required synthetic duplication.

### Benchmark

Exactly 16 preregistered fixed-exposure configurations used one common catalytic matrix/initial state, three daughter semantics where specified, retained versus newly joined trimming, and exposures 0.10, 0.25, 0.75, or 1.25. Every trajectory completed 100 fissions and every same-seed replay comparison passed.

| particleId                                                               |    h | daughterRule    | overshootRule             |   clockC0 |   clockC1 |   medianPostFissionMass |   q95Overshoot | exactReplayPassed   |
|:-------------------------------------------------------------------------|-----:|:----------------|:--------------------------|----------:|----------:|------------------------:|---------------:|:--------------------|
| BENCH-00-RANDOM_NONEMPTY-RETAIN_OVERSHOOT-h=0.25                         | 0.25 | RANDOM_NONEMPTY | RETAIN_OVERSHOOT          |      1148 |      1248 |                    41.5 |          11.1  | True                |
| BENCH-01-RANDOM_NONEMPTY-TRIM_NEW_ENTRANTS_TO_NMAX-h=0.25                | 0.25 | RANDOM_NONEMPTY | TRIM_NEW_ENTRANTS_TO_NMAX |      1083 |      1183 |                    40   |          10.05 | True                |
| BENCH-02-FIRST_DAUGHTER-RETAIN_OVERSHOOT-h=0.25                          | 0.25 | FIRST_DAUGHTER  | RETAIN_OVERSHOOT          |      1278 |      1378 |                    42   |           9.05 | True                |
| BENCH-03-FIRST_DAUGHTER-TRIM_NEW_ENTRANTS_TO_NMAX-h=0.25                 | 0.25 | FIRST_DAUGHTER  | TRIM_NEW_ENTRANTS_TO_NMAX |      1225 |      1325 |                    39.5 |          10    | True                |
| BENCH-04-RANDOM_LITERAL-RETAIN_OVERSHOOT-h=0.25                          | 0.25 | RANDOM_LITERAL  | RETAIN_OVERSHOOT          |      1090 |      1190 |                    42   |          10.05 | True                |
| BENCH-05-RANDOM_LITERAL-TRIM_NEW_ENTRANTS_TO_NMAX-h=0.25                 | 0.25 | RANDOM_LITERAL  | TRIM_NEW_ENTRANTS_TO_NMAX |      1358 |      1458 |                    39.5 |          10    | True                |
| BENCH-06-RANDOM_NONEMPTY-RETAIN_OVERSHOOT-h=0.75                         | 0.75 | RANDOM_NONEMPTY | RETAIN_OVERSHOOT          |       403 |       503 |                    45   |          23    | True                |
| BENCH-07-RANDOM_NONEMPTY-TRIM_NEW_ENTRANTS_TO_NMAX-h=0.75                | 0.75 | RANDOM_NONEMPTY | TRIM_NEW_ENTRANTS_TO_NMAX |       539 |       639 |                    40   |          24.05 | True                |
| BENCH-08-FIRST_DAUGHTER-RETAIN_OVERSHOOT-h=0.75                          | 0.75 | FIRST_DAUGHTER  | RETAIN_OVERSHOOT          |       404 |       504 |                    45   |          29    | True                |
| BENCH-09-FIRST_DAUGHTER-TRIM_NEW_ENTRANTS_TO_NMAX-h=0.75                 | 0.75 | FIRST_DAUGHTER  | TRIM_NEW_ENTRANTS_TO_NMAX |       443 |       543 |                    39.5 |          22.1  | True                |
| BENCH-10-RANDOM_LITERAL-RETAIN_OVERSHOOT-h=0.75                          | 0.75 | RANDOM_LITERAL  | RETAIN_OVERSHOOT          |       461 |       561 |                    45   |          19.05 | True                |
| BENCH-11-RANDOM_LITERAL-TRIM_NEW_ENTRANTS_TO_NMAX-h=0.75                 | 0.75 | RANDOM_LITERAL  | TRIM_NEW_ENTRANTS_TO_NMAX |       469 |       569 |                    40   |          29.05 | True                |
| BENCH-12-RANDOM_NONEMPTY-RETAIN_OVERSHOOT-h=0.10000000000000001          | 0.1  | RANDOM_NONEMPTY | RETAIN_OVERSHOOT          |      2519 |      2619 |                    41.5 |           5    | True                |
| BENCH-13-RANDOM_NONEMPTY-TRIM_NEW_ENTRANTS_TO_NMAX-h=0.10000000000000001 | 0.1  | RANDOM_NONEMPTY | TRIM_NEW_ENTRANTS_TO_NMAX |      2728 |      2828 |                    40   |           5    | True                |
| BENCH-14-RANDOM_NONEMPTY-RETAIN_OVERSHOOT-h=1.25                         | 1.25 | RANDOM_NONEMPTY | RETAIN_OVERSHOOT          |       231 |       331 |                    46   |          45    | True                |
| BENCH-15-RANDOM_NONEMPTY-TRIM_NEW_ENTRANTS_TO_NMAX-h=1.25                | 1.25 | RANDOM_NONEMPTY | TRIM_NEW_ENTRANTS_TO_NMAX |       343 |       443 |                    40   |          46.35 | True                |

The worst-case campaign projection was 1.4654 CPU-hours and 0.2442 wall-hours at six workers, far below the 250/72-hour ceilings. Benchmark length changes are feasibility evidence only and were not used as an ABC posterior or manual candidate selection.

### Phase 2 — ABC-SMC and terminal gate

The frozen fixed-family schedule was 256/128/64 particles across three rounds, eight common development matrices per particle, with a log-uniform `h in [0.10,1.25]`. Round 1 executed 256×8 = 2,048 true/replay trajectory pairs. Before `particle_summary_and_distance` was called, the global exact-replay check found at least one false comparison and raised the terminal error. Consequently:

- zero particle distances were opened;
- zero posterior weights or accepted particles were produced;
- the conditional adaptive family was not run;
- zero candidate identities were locked;
- zero 32-matrix confirmation trajectories were generated.

`abc_particle_results.parquet` retains all 256 preregistered particle identities with numeric outcomes suppressed; `abc_round_summary.csv` records the global failure without inventing an exact failing-pair count.

### Static comparator caveat

After stopping, source-only inspection noted that the frozen dataclass equality includes exposure extrema set to IEEE NaN when a generation contains zero growth updates, while IEEE NaN is unequal to itself. This is a plausible comparator-policy risk, not an established cause: no failing pair was rerun, its records were not inspected, and no code was patched. The gate remains failed exactly as executed.

### Phases 3–4

Untouched confirmation and the downstream timebase lock were not reached. Their required tables contain explicit `NOT_REACHED` rows and reasons. `candidate_timebase_pipeline_lock.json` contains zero confirmed candidates. No label or information-theory outcome exists.

## Results

The first failed dependency was the development exact-replay comparator, after target extraction and benchmark validation but before likelihood-free scientific scoring. The only allowed terminal classification that describes this state is `SIMULATOR_IDENTIFICATION_FAILED`. This does not support `NO_PAPER_TIMEBASE_RECONSTRUCTION`, because the fixed and adaptive candidate families were not scientifically adjudicated.

## Commands

```bash
python scripts/e01/freeze_s12f_preregistration.py
PYTHONPATH=src python -m pytest -q tests/e01/test_s12f_latent_timebase.py
PYTHONPATH=src python -m pytest -q tests/e01/test_s12e_paper_pipeline_detective.py tests/e01/test_s12f_latent_timebase.py
ruff check src/e01_latent_timebase scripts/e01/freeze_s12f_preregistration.py scripts/e01/run_s12f_latent_timebase.py tests/e01/test_s12f_latent_timebase.py
git commit -m "Preregister S12F latent time-base inference"
git push origin eidosoma/groups/42
python scripts/e01/freeze_s12f_preregistration.py --record-commit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=src python scripts/e01/run_s12f_latent_timebase.py \
  --stage development --require-pushed-preregistration
```

The last command stopped with `RuntimeError: development exact replay failed`. It was not relaunched. The post-failure artifact finalization generated only status/provenance/reporting outputs from frozen inputs, completed clock/benchmark files, the terminal traceback, and deterministic seed identities; it did not rerun simulations or inspect suppressed outcomes.

## Dependencies, parameters, and runtime

CPU float64 was authoritative with six process workers and one BLAS/OpenMP thread per worker. Python `3.13.14`, NumPy `2.4.6`, SciPy `1.18.0`, pandas `2.3.3`, and PyArrow `24.0.0` were used. The L4 was not used. Frozen GARD parameters were `N_g=100`, `n_min=40`, `n_max=80`, 100 generations, 1,000 maximum updates/generation, `k_f=0.01`, `k_b=0.0001`, `rho_i=0.01`, and complementary species-wise binomial fission at p=0.5.

## Validation

- Two independent Figure-2 calibrations: **PASS**.
- Preregistration/targets committed and pushed before new simulation: **PASS**.
- Focused tests: **9/9 PASS**; S12E+S12F regression set: **19/19 PASS**; Ruff: **PASS**.
- Read-only S12E state cardinality and prior replay evidence: **PASS**.
- Benchmark exact replay: **16/16 PASS**.
- Full development exact replay: **FAIL (>=1/2,048 pairs)**; terminal stop enforced.
- S12E/S12F recorded seed-material intersection: **0, PASS**.
- Prior artifacts and S12E caches: **PASS (434 + 6107 files unchanged)**.
- Labels, emergence, local Phi-r, prediction, intervention, S12G, and S13 access: **NONE**.
- Runtime/storage ceilings: **PASS; stopped early**.

## Caveats, blockers, and failed assumptions

The replay failure is a validation blocker, not a scientific null. Its exact scope is unknown because the global check did not persist pair-level results before raising. Static NaN behavior may explain a false inequality but was not tested after the stop. Figure targets remain raster-derived; the aggregate terminal is interval-censored; the Table-1 ratio is not a trajectory-length estimator. C3/C4 are unrecoverable from the S12E caches. C2 is a forbidden synthetic duplication. The one-matrix benchmark cannot identify `h`, daughter semantics, overshoot handling, or an author implementation.

## Provenance and artifact completeness

`source_input_snapshot_manifest.json`, `immutable_prior_baseline.json`, and `s12e_cache_manifest.json` pin the source/context/prior state. `development_seed_manifest.parquet` distinguishes the completed benchmark from the attempted stopped round. All conditional outputs are status-bearing. `post_failure_reporting_finalization.json` documents that no scientific code or result was changed after the failure. `artifact_manifest.json` records SHA-256 and size for every retained artifact except itself. The workspace plan hash at finalization is `cd550403f5c4bf81d0578c62c18342c88e1fa65802c375cc618d3f00d782aec6`.

## Recommended next action

Return for mandatory human review with S13 `BLOCKED_PENDING_S12F_HUMAN_REVIEW`. Do not automatically repair or rerun S12F, start S12G/S13, compute labels or emergence, or broaden the model family.
