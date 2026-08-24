# S11 — Reconstruct time-localized Phi-r: full results

## Top summary

| Field | Result |
| --- | --- |
| Research step | **S11 — Reconstruct time-localized Phi-r** |
| Completion status | **RETURN_FOR_REVIEW_VALIDATION_BLOCKED.** S11 execution and artifact production are complete; S12 was not begun. |
| Artifacts written | Canonical preregistration and record; exact-pair eligibility; finite-sample calibration, truth, CCS-oracle, regularization, shuffle, invariance, partition, CPU/GPU, causality, failure-injection, reproducibility, runtime, seed, and registry evidence; 92,808 representative partition-candidate rows; 17,208 partition-history rows; 34,416 status-bearing estimate rows; 6,912 strict-scope atom rows; two figures; versioned contract and eligibility registry; provenance/attempt and artifact manifests; this report. |
| Validation result | **Constrained: 11/16 gate families passed.** The canonical output/schema audit passed, but no fixed-window branch passed every frozen gate: **0/576 branch rows eligible**, **0/16 fixed pairs eligible**, and **0/33,984 fixed estimate rows numeric**. The unchanged strict S10 branch produced 432 eligible synthetic expanding/whole-scope estimates at \(n_\mathrm{eff}\ge512\). |
| Outcome classification | **Constraining/contradictory.** The preregistered small-window reconstruction is not eligible for scientific use. |
| Caveats or blockers | Known-truth failures affected 18/64 summaries; every exact pair failed the structured time-shuffle gate; all 48 partition summary gates failed; high-dimensional relabel invariance passed only 62/256 attempts; dimension-8 approximate search agreement was 0.326–0.850, below 0.90. Fixtures are synthetic, CCS remains experimental, whole-trajectory results are non-prospective, and no author/PhiID/MATLAB or paper-primary identity was recovered. |
| Recommended next action | **Do not begin S12.** Return to the Chief Scientist for a method decision. Any future work must either accept strict expanding/whole scopes only, or preregister a new follow-up estimator/partition/null-control experiment; it must not weaken S11 gates post hoc or reinterpret failed fixed branches as eligible. |

## Lay summary

The paper-like windows contain only 24–255 usable past/future pairs, while the previously validated reference method requires at least 512. S11 therefore built and preregistered a separate regularized estimator for these smaller windows, then challenged it on systems where the correct answer was known and on 99- and 100-dimensional partition problems representative of E01.

Some infrastructure worked very well: null centering, covariance regularization, exact repeatability, causality-safe indexing, CPU/GPU agreement, and the original strict method at 512 or more samples all passed. The fixed-window reconstruction nevertheless failed its mandatory truth, shuffle, partition-search, and relabel-stability checks. In particular, time-shuffled systems with strong instantaneous redundancy did not behave like the independent-null calibration, and the approximate partition search did not agree sufficiently with exhaustive search. The correct scientific action is therefore to emit no fixed-window Phi-r numbers. S11 did exactly that and returned 33,984 explicit ineligible rows instead of silently filling in estimates.

## Frozen question and decision rule

The frozen question was whether past-only expanding and sliding windows could recover a stable, interpretable Phi-r trajectory. The user additionally required a separately versioned small-window/high-dimensional branch at all exact \(n_\mathrm{eff}=w-\tau\) values before any fixed-window estimate, without changing S10's strict \(n_\mathrm{eff}\ge512\) branch.

The preregistration was committed and pushed as `f257146cd845f49ec01efe08175a15cae64ccf39` (SHA-256 `1c21ad91927929626edb6b2e14dfa745674decbaa89c3ac57f2cfdc678458f40`) before benchmark outcomes were inspected. A fixed branch was eligible only if its exact-size estimator, truth, null, regularization, shuffle, partition, approximation, invariance, reproducibility, causality, closure, and CPU/GPU gates all passed. A failed branch had to retain status and reason but no numeric estimate.

The fixed grid was:

| Window | Lags | Exact effective samples |
| ---: | --- | --- |
| 32 | 1, 2, 4, 8 | 31, 30, 28, 24 |
| 64 | 1, 2, 4, 8 | 63, 62, 60, 56 |
| 128 | 1, 2, 4, 8 | 127, 126, 124, 120 |
| 256 | 1, 2, 4, 8 | 255, 254, 252, 248 |

All remain below S10's strict threshold. The strict gate ID `E01-S10-SAMPLE-GATE-STRICT-v1.0.0` was not renamed, relaxed, or applied to a fixed pair.

## Inputs

The 21 frozen inputs and their hashes are enumerated in `preregistration.yaml` and reverified in `preregistration_record.json`. They include:

- `AGENTS.md`, `FULL_PLAN.md`, and the pre-S11 `RESEARCH_PLAN.md`;
- the attachment manifest, attachment sidecar, supplied paper extraction, and official arXiv v1 PDF (SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`);
- the S09 report, compositional contract, transform specifications, and valid-transform registry;
- the complete S10 report, preregistration/record, eligibility evidence, and information-dynamics contract;
- the source and environment manifests, S06 seed contract, and specification registry v0.3.0.

No S12 GARD trajectory existed or was generated. All S11 time-localized histories are validation fixtures, not E01 baseline or intervention results.

## Detailed methods

### Distinct small-window estimator

`E01-S11-PHIID-GAUSSIAN-OAS-CROSSFIT-v1.0.0` is a new validation branch, not a modified S10 source wrapper. For each pair it constructs ((x_t,y_t,x_{t+\tau},y_{t+\tau})), divides the exact lagged rows into four contiguous balanced evaluation folds, standardizes from the other three folds, fits an identity-target Oracle Approximating Shrinkage covariance, and evaluates held-out Gaussian local surprisals. No row is deleted and no numerical fallback is allowed.

The local quantities enter the pinned phyid 16-atom linear lattice under two separate redundancy identities:

- MMI, selecting the lower ensemble-mean local mutual-information array;
- experimental CCS, preserving the source's sign-filtered common-change formula and its “To be implemented” caveat.

All 16 atoms and nine intermediate mutual informations are retained. The paper equation is reported only as the derived aggregate

\[
\mathrm{str}+\mathrm{stx}+\mathrm{sty}+\mathrm{sts}
-\mathrm{rtr}-\mathrm{rtx}-\mathrm{rty}-\mathrm{rts},
\]

with a direct closure check against (I(XY;X'Y')-I(X;X'Y')-I(Y;X'Y')). It is not relabeled as a source-named atom.

### Finite-sample calibration and known truth

For each of 16 pairs and each redundancy branch, 512 independent-white calibration replicates defined an exact-pair atom/MI mean correction and 99th-percentile aggregate envelope. A disjoint set of 256 held-out null replicates evaluated bias and false-positive rates.

Known-truth testing used 256 replicates per pair for noisy redundant AR(1) and directional VAR systems. MMI targets came from analytic lag-specific Gaussian covariances. CCS targets came from two independently scrambled (2^{18}=262{,}144)-draw Sobol population calculations per system/lag. OAS shrinkage multipliers 0.5, 1.0, and 1.5 were assessed without replacing the canonical multiplier after outcomes.

### High-dimensional partition and search

`E01-S11-MIB-SEARCH-STABILITY-SPECTRAL-SINGLE-FLIP-v1.0.0` standardizes components, forms an absolute contemporaneous plus bidirectional lag-correlation affinity, generates eight Bayesian-bootstrap spectral partitions, builds a co-assignment consensus, and evaluates base/bootstrap/consensus/single-flip candidates. A separate 16-bootstrap run tests approximation stability.

Validation covered dimensions 99 and 100, 16 signal and 16 null replicates at every exact pair. The planted splits were 49/50 and 50/50. Dimension 8 used 64 replicates per pair to compare the candidate search with all 127 unordered bipartitions. Every cross-product of two scalar mappings, three objectives, and three normalizations remained separately identified:

- mappings: standardized group mean and deterministic standardized PC1;
- objectives: synchronous MI, bidirectional lagged MI, and absolute paper-equation aggregate;
- normalizations: none, minimum part entropy, and geometric part size.

There was no post hoc paper-primary designation. OmegaID discrete and multivariate-doublet paths were excluded. Guarded OmegaID Gaussian was used only for strict scalar 1-by-1 equal-width cross-checks.

### Temporal scopes

Every fixed history used an inclusive past-only window and required `futureIndexMax <= windowEnd`. The 2,048-row piecewise two-block validation fixture was evaluated at the first complete window, half-window cadence, and final row. Every mapping/objective/normalization evaluation retained its partition or an explicit ineligible reason.

Expanding scopes used the unchanged pinned phyid branch only at 512 and 1,024 effective samples. Whole-trajectory results used the same strict branch and carry `NON_PROSPECTIVE_WHOLE_TRAJECTORY_DESCRIPTION` plus `prospective=false`.

### Randomness, precision, and compute

All fixtures and bootstrap streams use S06-domain-separated PCG64DXSM estimator identities rooted at `11` repeated for 32 bytes. The compact manifest covers 29,518 unique streams and the complete records are in `seed_records.parquet`. Computation used eight process workers with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` fixed to one.

The canonical interpreter was Python 3.13.14 in `/cache/e01_s10/venv`, with NumPy 2.4.6, SciPy 1.18.0, pandas 2.3.3, and CuPy 13.6.0. Device 0 was NVIDIA L4 `GPU-1f5bed19-d63f-be98-470c-f670e3d4aafd`; a second L4 was visible but unused. CPU/GPU comparisons used IEEE-754 binary64.

## Commands

Key reproducible commands were:

```bash
git commit -m "Preregister S11 time-localized Phi-r validation"
git push origin eidosoma/groups/42

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 /cache/e01_s10/venv/bin/python \
  scripts/e01/run_s11_time_localized_phir.py --workers 8

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 /cache/e01_s10/venv/bin/python -m pytest -q \
  tests/e01/test_time_localized_phir.py \
  --deselect=tests/e01/test_time_localized_phir.py::test_preregistration_is_frozen_complete_and_does_not_relax_s10

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 /cache/e01_s10/venv/bin/python -m pytest -q tests/e01 \
  --deselect=tests/e01/test_time_localized_phir.py::test_preregistration_is_frozen_complete_and_does_not_relax_s10 \
  --deselect=tests/e01/test_compositional_preprocessing.py::test_builder_writes_complete_lossless_status_bearing_artifacts \
  --deselect=tests/e01/test_information_dynamics.py::test_preregistration_is_frozen_and_complete \
  --deselect=tests/e01/test_rng_schema.py::test_generated_artifacts_and_fresh_process_regeneration_when_present

python -m ruff check src/e01_time_localized_phir \
  scripts/e01/run_s11_time_localized_phir.py \
  tests/e01/test_time_localized_phir.py
```

Before the required S11 handoff update mutated `RESEARCH_PLAN.md`, the focused suite passed 18/18 and the repository suite passed 109/109 non-state-sensitive tests with three earlier-step plan-identity checks deselected. After the handoff update, the final regression rerun passed 17/17 focused computational tests and 108/108 non-state-sensitive E01 tests with four historical plan-identity checks deselected. The added deselection is the S11 test that deliberately asserts the preregistered pre-S11 plan hash; `preregistration_record.json` independently records its 21/21 successful pre-outcome hash audit. One final full-suite invocation accidentally omitted the frozen thread variables and correctly failed 11 S06 fail-closed environment checks; the corrected command above passed with one known pinned-phyid divide-by-zero warning from enumerating absent binary states. No code, gate, or result artifact was changed in response.

## Results

### Anchor results

| Result | Observation | Gate |
| --- | ---: | --- |
| Frozen inputs | 21/21 hashes matched | Pass |
| Held-out null calibration | 32/32 pair/redundancy rows passed | Pass |
| Known-truth summaries | 46/64 passed; 18 failed | Fail |
| CCS population oracle | 8/8 cross-scramble checks passed; maximum difference (1.93\times10^{-4}) nats | Pass |
| Regularization | 32/32 sensitivity rows passed | Pass |
| High-dimensional/exact-search summaries | 0/48 passed | Fail |
| Invariance attempts | 356/576 passed | Fail |
| Structured shuffle rows | 28/64 passed | Fail |
| CPU/GPU comparisons | 190/190 passed | Pass |
| Causality rows | 956/956 passed | Pass |
| Failure injections | 12/12 passed | Pass |
| Fixed eligibility | 0/576 branches; 0/16 pairs | Fail closed |
| Fixed numeric estimates | 0/33,984; every row has status/reason | Correct suppression |
| Strict synthetic scopes | 432/432 estimates eligible at \(n_\mathrm{eff}\ge512\) | Pass |
| Maximum strict closure errors | lattice (2.22\times10^{-16}), equation (1.94\times10^{-16}) nats | Pass |

### Calibration and estimator behavior

All held-out independent-null rows passed. Maximum absolute held-out atom mean was (9.72\times10^{-4}) nats; maximum absolute aggregate mean was (1.08\times10^{-4}) nats; false-positive rates ranged from 0 to 0.0234 against the frozen 0.05 gate.

Known-truth failures were concentrated rather than universal. The maximum total-MI error was 0.29494 nats for the 31-effective-sample directional system, exceeding 0.20. Other failed rows were largely directional-sign gates at weak long-lag signals: 18/64 summaries failed in total, while aggregate and atom errors were often small. Both MMI and experimental CCS therefore remain ineligible for the affected pair branches; the good numerical behavior of one summary was never substituted for a failed directional gate.

### Shuffle control

Every exact pair failed at least one structured shuffle branch. For noisy redundant AR, only 0.0547–0.3125 of shuffled replicates fell inside the independent-null 99% envelope, versus the required 0.95. Directional VAR performed much better (0.9297–1.0), but four pair/redundancy combinations still fell below 0.95.

The redundant-system result is an important constraint: permuting time destroys ordered lag structure but preserves strong within-row source/target redundancy, whereas the calibration null has independent within-row variables. The observed mismatch shows that an independent-white envelope is not exchangeable with this structured shuffle control for the new estimator. This is an interpretation of the frozen evidence, not a replacement calibration; no conditional-null correction was introduced after seeing the result.

### Partition and approximation behavior

The null partition fixtures failed closed: eligible null counts were zero in every dimension/pair summary. Signal recovery was nonetheless insufficient under the simultaneous gates.

- In D=100, all 235 signal replicates that passed the stability gate recovered the planted split exactly, but 21/256 signal attempts were ineligible, mostly at short effective sizes.
- In D=99, 227/256 signal attempts were eligible. The typical ARI was 0.95959, while exact recovery occurred in only 2 cases; the consensus commonly assigned one extra component to the 49-member block.
- Dimension-8 candidate-versus-exhaustive agreement was 0.326–0.850 across exact pairs, below the frozen 0.90 threshold for all 16 pairs.
- Primary-versus-16-bootstrap winner ARI was usually 1 when both searches were eligible, but this did not rescue failures of eligibility, exact recovery, exhaustive agreement, or invariance.

All 576 preregistered invariance attempts are retained. Scalar signed-affine and source/target relabel controls passed 64/64. Positive componentwise affine partition controls passed 230/256; 26 were explicitly ineligible. Feature relabel controls passed only 62/256, with 97 unavailable comparisons and additional non-unit ARIs down to 0.2647. Spectral degeneracy/tie behavior and stability-gate interactions are therefore material implementation constraints, not harmless relabelings.

### Fixed, expanding, and whole histories

`partition_histories.parquet` contains 17,208 rows: 16,992 fixed, 144 expanding, and 72 whole-trajectory histories. Fixed partition search itself was eligible for 7,236 rows and ineligible for 9,756, but global pair/branch gates prevented all fixed Phi-r estimates regardless of a local partition's status.

`phir_estimates.parquet` contains 34,416 rows. The 33,984 fixed rows are all `INELIGIBLE` with null numeric fields. The 288 expanding and 144 whole-trajectory strict rows are eligible synthetic validation outputs. Whole rows are explicitly non-prospective. These strict results demonstrate implementation feasibility only; they are not GARD results and do not validate a fixed-window scientific analysis.

### CPU/GPU and source guard

All 96 NumPy-versus-CuPy comparisons for the new estimator passed; maximum absolute difference was (1.08\times10^{-15}). All 94 guarded OmegaID Gaussian CPU-versus-GPU strict-scope comparisons passed; maximum absolute difference was (1.33\times10^{-15}). OmegaID discrete was never called, and no multivariate doublet was accepted as a 16-atom result.

### Runtime

The canonical run took 598.6 seconds before final serialization. The largest stages were fixed-window history construction (257.8 seconds), dimension-8 exhaustive validation (88.5 seconds), high-dimensional validation (74.4 seconds), CPU/GPU validation (32.5 seconds), and strict scopes (28.1 seconds). Runtime values are performance observations, not acceptance criteria.

## Validation and completeness

The 16 validation families were:

| Gate family | Result |
| --- | --- |
| Preregistration and frozen inputs | Pass |
| Held-out null calibration | Pass |
| MMI known truth | **Fail** |
| CCS population oracle | Pass |
| Experimental CCS known truth | **Fail** |
| Regularization | Pass |
| High-dimensional partition and null | **Fail** |
| Affine and relabel invariance | **Fail** |
| Time shuffle | **Fail** |
| CPU/GPU | Pass |
| Causality indexing | Pass |
| Strict expanding/whole scope | Pass |
| Reproducibility | Pass |
| Failure injection | Pass |
| Registry preservation | Pass |
| Output schema and suppression | Pass |

Output validation confirmed 576 fixed eligibility rows, all 16 pair IDs, 34,416 estimate rows, 6,912 atom rows, exactly 16 atoms per eligible strict estimate, no atoms or numeric fields for any ineligible estimate, four successful Parquet round trips, all 576 invariance attempts, and no S12 artifact directory.

The first canonical attempt reached the same scientific outcome but omitted 97 relabel-failure rows when ARI was unavailable. It is preserved at `/cache/e01_s11/attempt1`; its validation summary is copied into the canonical artifacts. The second attempt changed only status-row retention and completeness assertions. Nine scientific anchor files and `validation_summary.json` were byte-identical across attempts. `attempt_history.json` records hashes and reasons.

Both generated figures were visually inspected. `time_localized_phir.png` explicitly states that no fixed numeric estimate was emitted, and `partition_stability.png` shows planted ARI by exact pair and representative dimension.

## Artifacts written

Key canonical artifacts under `/artifacts/research_steps/S11/` are:

- `research_step_full_results.md`, `validation_summary.json`, `output_validation.json`, and `artifact_manifest.json`;
- `preregistration.yaml`, `preregistration_record.json`, `specification_metadata.yaml`, and `exact_pair_eligibility.csv`;
- calibration/truth/CCS/regularization/shuffle/invariance/partition/CPU-GPU/causality/runtime tables;
- `partition_candidate_scores.parquet`, `partition_histories.parquet`, `phir_estimates.parquet`, and `atom_outputs.parquet`;
- `seed_manifest.json`, `seed_records.parquet`, `runtime_manifest.json`, `registry_preservation.json`, `failure_injection.json`, and `reproducibility_validation.json`;
- `time_localized_phir.png`, `partition_stability.png`, `attempt_history.json`, and the first-attempt summary.

Reusable bundle contracts are:

- `/artifacts/E01_forensic_replication_bundle/information_dynamics/time_localized_phir_contract_v1.0.0.yaml`;
- `/artifacts/E01_forensic_replication_bundle/information_dynamics/time_localized_phir_eligibility_registry_v1.0.0.yaml`.

Repository-backed implementation is in `src/e01_time_localized_phir/`, with the canonical runner at `scripts/e01/run_s11_time_localized_phir.py`, preregistration at `configs/e01/s11_time_localized_phir_preregistration.yaml`, and focused tests at `tests/e01/test_time_localized_phir.py`.

## Provenance

- Repository: `Eidosoma/arrival-of-self-replicators`, branch `eidosoma/groups/42`.
- Preregistration commit: `f257146cd845f49ec01efe08175a15cae64ccf39`.
- Canonical-run code head: `cdb51bc3751137f86cd1f00e0c31086abdba63b4`.
- Final visualization/handoff commit: `c7af34694ac74ff6d41e72e427a3f3698634676b`; pushed to `eidosoma/groups/42`.
- Registry v0.3.0 before/after SHA-256: `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891`; byte-for-byte preserved, execution gate still closed, no author sentinel resolved.
- Validation summary SHA-256: `d60bc525c59948b0274f4cb121c6e91d723184c8293c089fc44c122061720674`.
- Seed records: 29,518 unique estimator-stream identities; aggregate and table hashes are in `seed_manifest.json`.
- Complete runtime/package/GPU identity is in `runtime_manifest.json`; every final file hash is in `artifact_manifest.json`.

## Caveats, blockers, and failed assumptions

1. The small-window branch did not pass its frozen validation gates and must not be used for S12 as currently specified.
2. Strict \(n_\mathrm{eff}\ge512\) results are synthetic feasibility checks. They do not answer the fixed-window question and whole-trajectory rows are non-prospective.
3. The structured time-shuffle null is not calibrated by the independent-white envelope when strong contemporaneous redundancy remains. A different conditional null would be a new preregistered method, not a repair to S11.
4. Spectral partitioning is sensitive to feature relabeling and odd-dimensional/tied graph structure. High planted ARI alone is insufficient because exact recovery, exhaustive agreement, and invariance also failed.
5. CCS remains experimental, and neither MMI nor CCS is identified as the paper-author redundancy.
6. Mapping, objective, normalization, search, zero treatment, coordinate system, and atom identity remain separate versioned branches. No branch is paper-primary.
7. No unavailable author code, author estimator mapping, MATLAB RNG semantics, or causal biological meaning is claimed.
8. The paper-facing specification registry remains non-executable; S11 did not mutate source evidence or resolve non-source sentinels.

## Recommended next action

Hand control back and keep S12 unstarted. The Chief Scientist should choose explicitly among:

1. constrain future E01 work to the already validated strict expanding scopes after 512 effective samples, treating whole-trajectory results as non-prospective;
2. authorize a separately numbered follow-up that preregisters a conditional structured-null calibration and a permutation-equivariant partition method, then repeats exact-size validation; or
3. stop the Phi-r reconstruction branch as methodologically underdetermined.

None of these choices may retroactively make an S11 fixed branch eligible, and no S12 run should start until that review is resolved.
