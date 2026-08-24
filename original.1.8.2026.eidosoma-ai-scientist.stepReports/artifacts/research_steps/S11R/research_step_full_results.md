# S11R research-step full results — bounded fixed-window repair validation

## Concise top summary

- **Research step ID:** S11R
- **Completion status:** COMPLETE — FAIL CLOSED; S11 and all failed S11 outputs remain immutable; S12 was not begun.
- **Artifacts written:** All 25 required step files under `/artifacts/research_steps/S11R` (24 payload files plus `artifact_manifest.json`), plus the versioned repair contract and eligibility registry in the information-dynamics bundle. Principal files are `validation_summary.json`, seven result/eligibility CSVs, `seed_records.parquet`, two validation figures, provenance records, and this report.
- **Validation result:** FAIL — 16/19 total preregistered gate families passed. Failed families: known_truth_mmi, known_truth_ccs_experimental, d8_exhaustive_agreement.
- **Outcome classification:** **constraining/contradictory**.
- **Caveats or blockers:** This is a validation-only reconstruction, not the unavailable author implementation or MATLAB RNG. MMI and experimental CCS, all mappings/objectives/normalizations, and author-method uncertainty remain distinct. S10's strict >=512 branch is unchanged. No OmegaID discrete/doublet substitute or GARD scientific estimate was used.
- **Lay summary:** The repair tested a small-sample bias correction, a null matched to each synthetic system, and a feature-order-independent partition rule on a fresh, untouched confirmation set. At least one frozen confirmation check failed, so the repair correctly shut itself down: all 576 fixed-window branches remain ineligible and no scientific values were produced.
- **Recommended next action:** Hand control back for Chief Scientist review. Do not begin S12 without a new explicit instruction. Do not continue this bounded repair path.

## Frozen question

Can the separately versioned Wishart-corrected Gaussian estimator, condition-matched complete-row-shuffle null, and permutation-equivariant threshold-component partition method pass untouched confirmation at all 16 planned effective sample sizes (24–255), without changing S11 or S10?

## Scope and immutable boundaries

S11R was preregistered at Git commit `a4763d0d5c7428897fcb595ae7d25a754d346c31` (SHA-256 `9f8a9424fae41a5ed7ea0d185eb5aaa31449bd12892c027eda2dc83fb57e99e0`) before development or confirmation outcomes. The implementation was frozen at `c244bae2fbdefb8555c8fe8f0eeb11da531afff6`; the method lock was committed and pushed at `0497e9cf84cd4360ba4a36796da2443ae2dae002` before confirmation access. Development and confirmation used separate 256-bit roots. The original S11 artifact manifest, all 34 manifest entries, and eight S11 repository files were verified byte-for-byte before the run and at handoff. S11's 46/64 truth, 28/64 shuffle, 356/576 invariance, 0/48 partition, 0/576 eligibility, and 0/33,984 numeric fixed-window findings were not altered or relabeled.

S10's `E01-S10-SAMPLE-GATE-STRICT-v1.0.0` still requires at least 512 effective samples. S11R is a distinct estimator valid only within its own preregistered validation domain and does not replace that strict branch.

## Inputs

The runner verified all 24 preregistered frozen inputs: workspace governance and plans, attachment manifest and sidecar, paper Markdown and official arXiv v1 PDF, S09 preprocessing report/contracts, S10 report/contracts/eligibility, S11 report/preregistration/results/contracts, S03 provenance/environment manifests, S06 seed contract, and registry v0.3.0. Exact paths and hashes are in `preregistration_record.json`.

No GARD trajectory or mounted dataset was used. All outcomes are validation-fixture evidence.

## Methods

### Small-window estimator

For each scalar pair, the estimator formed `(x_t,y_t,x_t+tau,y_t+tau)`, standardized the complete lagged sample with sample standard deviations, used the unregularized covariance with divisor `n_eff-1`, and evaluated all 15 required Gaussian local entropies in binary64 CPU arithmetic. For a p-dimensional subset and `nu=n_eff-1`, every local entropy received the preregistered exact Gaussian Wishart mean correction

`0.5 * [p/n_eff - (sum_i digamma((nu+1-i)/2) + p log(2) - p log(nu))]`.

There was no row deletion, covariance regularization, fallback, or post-hoc threshold change. MMI and experimental CCS used the same explicit 16-atom lattice identities as S10/S11 but remain separately labeled.

### Condition-matched null

Each calibration replicate came from the same declared structured system, exact window, and lag as its target condition. A uniformly random permutation was applied to complete contemporaneous two-feature rows before lagging. Calibration means were keyed by phase, exact pair, system, and redundancy. Held-out structured-shuffle replicates used disjoint streams. The 99th-percentile absolute centered equation envelope used NumPy's `method='higher'`. Independent white noise was not substituted for either structured system.

### Partition method

The partition affinity was the mean absolute Pearson correlation of lag-aligned past and future rows. An edge existed only above 0.90; any edge within 1e-12 of the threshold was ineligible. Exactly two connected components were required in the base fit and all eight Bayesian exponential-weight bootstraps. The gate additionally required minimum part fraction 0.10, mean bootstrap ARI 0.75, minimum within affinity 0.90, and within-minus-maximum-between affinity 0.10. No feature index or objective-score tie breaker selected the split.

D=8 independently enumerated all 127 unordered bipartitions for each of 18 mapping/objective/normalization branches. Exact objective ties within 1e-12 were ineligible. D=99/100 planted and independent-null fixtures, arbitrary feature permutations, and positive feature-wise affine transforms were directly confirmed at every exact pair.

### Development/confirmation firewall

Development used `d1d1d1d1...`; confirmation used `c1c1c1c1...`. Both used S06 SHA-256 domain separation and NumPy PCG64DXSM with phase, domain, pair, dimension, and replicate in the identity. The full cross-phase stream-ID and seed-material intersections were zero. Confirmation seeds were accessed only after the method-lock file was committed.

The first development execution exposed one harness-only defect: its synthetic S12-write injection searched report text instead of invoking the already-present path guard. Before the method lock, the injection was corrected to call the unchanged guard and development was replayed with identical seeds. All scientific diagnostic outcomes reproduced, all 16 injections then passed, and no estimator, partition rule, gate, sample size, or seed identity changed. This attempt history is preserved in `development_summary.json` and `method_lock.json`.

## Commands

```bash
PYTHONPATH=src pytest -q tests/e01/test_time_localized_phir_repair.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=src python scripts/e01/run_s11r_time_localized_phir_repair.py \
  --phase development --workers 8 --output /cache/e01_s11r/development
# implementation and method-lock commits were made and pushed here
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=src python scripts/e01/run_s11r_time_localized_phir_repair.py \
  --phase confirmation --workers 8 --output /artifacts/research_steps/S11R \
  --method-lock configs/e01/s11r_confirmation_method_lock.yaml \
  --method-lock-commit 0497e9cf84cd4360ba4a36796da2443ae2dae002
PYTHONPATH=src pytest -q tests/e01/test_time_localized_phir_repair.py
PYTHONPATH=src pytest -q tests/e01/test_time_localized_phir.py \
  -k 'not preregistration_is_frozen_complete_and_does_not_relax_s10'
PYTHONPATH=src ruff check src/e01_time_localized_phir_repair \
  scripts/e01/run_s11r_time_localized_phir_repair.py \
  tests/e01/test_time_localized_phir_repair.py
```

## Confirmation results

- **Condition-matched calibration:** 64/64 condition/redundancy rows passed. The maximum held-out atom mean magnitude was 0.005676 (gate 0.040), maximum equation-aggregate mean magnitude was 0.010556 (gate 0.040), and maximum false-positive rate was 0.03125 (gate 0.050). Structured-shuffle envelope coverage ranged from 0.96875 to 1.000, so all 64/64 shuffle rows passed.
- **Known truth:** MMI passed 45/48 and experimental CCS passed 45/48, for 90/96 combined. Every independent-white and noisy-redundant row passed. For each redundancy, the directional-VAR rows at `(w,tau)=(32,8),(256,4),(256,8)` failed only the preregistered directional-sign check; their largest total-MI, equation, and atom-RMSE errors were still below the numeric gates. Across all truth rows, maximum total-MI error was 0.141932, maximum equation error was 0.131890, and maximum atom RMSE was 0.044035.
- **D=8 exhaustive agreement:** 0/16 pair summaries met 0.90. Pair-level agreement ranged from 0.391493 to 0.754340; 11,218/18,432 individual branch comparisons agreed (0.608615). Exact search declared 3,396/18,432 comparisons ineligible because multiple exhaustive objective values tied within the frozen `1e-12` tolerance; retaining those failures was required. No mapping/objective/normalization branch was selected post hoc.
- **High-dimensional partitions:** all 512/512 D=99/100 planted fixtures were eligible with exact planted recovery (ARI 1.0); all 512/512 independent-null fixtures were ineligible, and no null numeric estimate was emitted. Thus all 32/32 pair-by-dimension planted/null summaries passed.
- **Invariance:** all 256 feature-relabel, 256 positive feature-affine, 64 scalar positive-affine, and 64 scalar source/target-relabel rows passed (640/640). Maximum scalar differences were `8.66e-16` and `8.04e-16` against `1e-9` gates.
- **Global decision:** 16/19 gate families passed, but simultaneous acceptance requires 19/19. Exact fixed-window eligibility is therefore 0/576 branches and 0/16 pairs. Numeric GARD scientific estimates: 0.

Detailed condition-level values, atom RMSEs, direction checks, envelope coverage, exact partitions, ARIs, rejection reasons, and every eligibility status are retained in the machine-readable tables. Failures are not pooled away or replaced by another branch.

## Validation

The run verified frozen preregistration bytes, all 24 frozen input hashes, the committed method lock, the complete development/confirmation seed firewall, exact 16-pair sample counts, S11 immutability, registry immutability, strict-boundary preservation, lattice/equation closure inside every eligible estimator call, exact CPU anchor replay, exact seed replay, all 16 failure injections, table cardinalities, figure creation, and absence of numeric scientific estimates. Development contained 11,920 unique stream identities; confirmation contained 39,184, with zero stream-ID and zero seed-material overlap.

All 25 required step files were present. The audited table cardinalities were 64 calibration rows, 96 truth rows, 64 shuffle rows, 18,432 D=8 rows, 1,024 high-dimensional rows, 640 invariance rows, 576 eligibility rows, and eight runtime rows. Confirmation `seed_records.parquet` contained 39,184 unique stream IDs and 39,184 unique seed materials. The 576 eligibility rows were all explicitly `INELIGIBLE_CONFIRMATION_GATE_FAILED`, with empty numeric-estimate and paper-primary fields. The S11 manifest remained at SHA-256 `21e58c969bc511cb620408518f96b5cab8acae02ec269fa376716c01123742ea`; registry v0.3.0 remained at `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891`; no S12 artifact directory existed.

Focused tests passed 9/9 for S11R. Sixteen non-state-sensitive S11 tests passed and one optional GPU test skipped; the one intentionally excluded S11 preregistration-state test compares the live `RESEARCH_PLAN.md` to its historical S11-era hash and is expected to fail after the explicitly authorized S11R plan update. Ruff passed. Both figures were visually inspected and matched their underlying gate tables. Outcome suppression was evaluated after all confirmation families, never branch-by-branch.

## Runtime and dependencies

The confirmation used 8 process workers with BLAS/OpenMP thread counts fixed to one. NumPy 2.4.6, SciPy 1.18.0, pandas 2.3.3, and scikit-learn 1.9.0 ran in Python 3.13.14 on `Linux-6.12.95+deb13-cloud-amd64-x86_64-with-glibc2.39`. GPU use was preregistered as false and no GPU computation occurred. Summed stage wall time was 167.689 seconds; per-stage timings are in `runtime_benchmarks.csv`.

## Caveats, blockers, and limitations

- The paper does not specify the reconstructed local estimator, null calibration, partition affinity/search, MIB mapping/objective/normalization, redundancy, atom mapping, or RNG semantics. None was silently designated as the paper default.
- Pinned phyid labels CCS as not implemented; the S11R CCS branch therefore remains explicitly experimental even if its gates pass.
- Wishart correction is exact for the declared Gaussian in-sample covariance model. Serially overlapping lag rows and non-Gaussian data are outside that theorem; the synthetic confirmation tests its finite-sample behavior rather than proving universal calibration.
- The threshold-component method is intentionally fail-closed and applies to strongly separated block fixtures. It is not evidence that a real GARD composition admits exactly two threshold components.
- D=8 exhaustive agreement tests the frozen 18-way uncertainty grid; poor agreement cannot be repaired by choosing a favorable objective after outcome inspection.
- Known-truth numeric errors were within tolerance, but three directional-VAR conditions per redundancy reversed the required finite-sample directional ordering. The simultaneous sign gate makes those rows failures; good magnitudes cannot override them.
- Validation eligibility, if present, is not author-code identity, MATLAB-RNG identity, a paper-primary designation, causal evidence, or a GARD scientific result.

## Provenance

The preregistration record contains every frozen input hash. `method_lock.json` identifies the committed implementation and development summary. `seed_records.parquet` and `seed_firewall.json` preserve domain-separated random identities. `s11_immutability.json`, `strict_boundary_preservation.json`, `runtime_manifest.json`, `reproducibility_validation.json`, and `artifact_manifest.json` provide the remaining byte, environment, replay, and output provenance. Repository-backed code remains in Git and is not copied into the artifact directory.

## Recommended next action

Hand control back to the Chief Scientist. The bounded S11R path failed closed; retain zero fixed-window eligibility and do not proceed to S12 or another repair without explicit review and authorization.
