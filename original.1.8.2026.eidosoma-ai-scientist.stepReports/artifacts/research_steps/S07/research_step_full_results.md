# S07 Full Results — Validate Stochastic Behavior Before Scientific Analysis

## Top summary

| Field | Result |
| --- | --- |
| Research step | **S07 — Validate stochastic behavior before scientific analysis** |
| Completion status | **Complete** on 2026-08-01; S08 was not begun. |
| Artifacts written | 19 compact files under `/artifacts/research_steps/S07/`: frozen preregistration and calibration, analytical fixtures, seed manifest, goodness-of-fit and moment tables/details, invariant and failure-injection results, four diagnostic plots, registry-preservation evidence, failed-attempt provenance, validation summary, artifact manifest, and this report. |
| Validation result | **PASS:** 26/26 preregistered primary tests, 54/54 deterministic invariants, and 7/7 failure injections passed. The minimum primary p-value was 0.030695 against the Bonferroni threshold 0.0001923077. The S07-focused suite passed 12/12; 60/60 other repository tests passed when the intentionally commit-pinned S06 regeneration test was deselected. |
| Outcome classification | **Supportive**, within the explicit validation profiles. Both engines produced the intended distributions on legitimately matched branches, and independent-only branches passed their own analytical tests. This is not author-code validation or paper-claim replication. |
| Caveats or blockers | The public historical engine is exercised through an explicit NumPy PCG64DXSM harness, not legacy MATLAB RNG. The author implementation remains unavailable. The paper-style vector-Poisson profile is an explicit reconstruction with frozen exposure/clipping/boundary choices, not an inferred default. Exact Monte Carlo p-values have finite resolution. Registry v0.3.0 remains non-executable and sentinel-bearing. |
| Lay summary | Before using either simulator for scientific conclusions, this step checked whether its random choices behave like the probability laws it claims to implement. Event choices, catalytic strengths, fission outcomes, modern waiting times, and the explicit paper-style Poisson branch all matched their preregistered targets. Deliberately corrupted results were also caught, showing that the checks can fail when they should. |
| Recommended next action | Return control to the Chief Scientist. S08 is eligible for separate authorization, but **do not begin S08 as part of this handoff**. |

## Frozen question

Do the historical-reference validation harness and independent Python engine generate the intended stochastic behavior—event probabilities proportional to \(a_k/a_0\), lognormal catalytic matrices with log moments \(A\) and \(\sigma^2\), exact single-event mass changes, and the specified fission laws—when, and only when, their explicit branch specifications are legitimately comparable?

The primary success criterion was fixed before canonical outcomes were generated: all 26 multiplicity-controlled primary tests, all deterministic invariants, and all seven failure-injection detectors had to pass. Independent-only branches were tested against analytical targets rather than being forced into an invalid historical comparison.

## Inputs

### Governing and paper inputs

- `/workspace/AGENTS.md`
- `/workspace/FULL_PLAN.md`
- `/workspace/RESEARCH_PLAN.md`
- `/workspace/input-attachments/MANIFEST.json`
- `/workspace/input-attachments/2607.28250v1/2607.28250v1_metadata/ATTACHMENT.md`
- Original equation-bearing paper PDF: `/cache/e01_s03/downloads/paper-2607.28250v1.pdf`, SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.

The paper supplied the lognormal catalytic-matrix parameters \(A=-4\) and \(\sigma=4\), described vector-Poisson state updates, and described binomial fission with probability 0.5. S07 did not treat paper prose as resolving conflicting historical-code behavior.

### Prior research-step inputs

The complete S01–S06 full-results reports and their ledgers/contracts were read before implementation. The most important frozen machine-readable identities were:

| Input | SHA-256 |
| --- | --- |
| `specification_registry_v0.3.0.yaml` | `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891` |
| `source_manifest.yaml` | `50a569e30826fe893258f5c0935469576008f43134eee94e8bf2654d4ef23ed5` |
| `environment_report.json` | `021c6f848e01172c098e615f27babcf6748dbba1f8bd0f1374883d9d392ef2cd` |
| `precision_policy.yaml` | `a49b2f37629382264881bb390dd010b453f1959d792eee180dae6c6c425259e6` |
| S04 historical behavior contract | `e6fe49aba2240047d018e5b619ef07d3e48922fb43a963256b6b2233f07d0a43` |
| S05 independent engine contract | `a35e313cb0685218691397980d1f5d8020fee8c994359e3227b9b1c1ef8605e8` |
| S05 validation profiles | `959fb3171e19087af06b09d21fd499e776b16de85e0b673f9ccde61e5b23ee0c` |
| S06 seed derivation contract | `a4c5586fc6be012afaff21f47fae422c4d6b6c68200c236df4a5b1ea5e736bb1` |
| S06 seed schema | `9eab6f1d796810fa3ba3e58d486d22293f65cac07da5c635c0aa023092e72218` |
| S06 trajectory precision contract | `2c73d7385d7511636cb809cdb1b2b5c0239632faec2f6ff2ffb692a7b3548b4d` |
| S06 trajectory schema | `981807b512bff589a6a693c1da191efad829ccb3294fd4f9297c3ee02a7a5d57` |

The artifact manifest records all 30 input files with exact path, size, role, and SHA-256.

## Preregistration and calibrated tolerances

All tolerances, sample sizes, hypotheses, test identifiers, multiplicity rules, rare-category handling, seeds, and failure injections were frozen before inspecting canonical outcomes. The preregistration was committed and pushed as `6490bb48a2780cdb73afa6ebe8da697496c4803d`; its source and artifact SHA-256 are both `ec86619c197a80ff6da4fa5ad07a464097544ce9691533d1ae26d894289e4104`.

The preregistration record established that the Git worktree was clean, all upstream hashes matched, the preregistered primary test IDs were exactly `S07-T01` through `S07-T26`, and no canonical outcome artifact existed at freeze time.

### Statistical gate

- One global family of 26 primary tests.
- Family-wise alpha: 0.005.
- Bonferroni per-test alpha: \(0.005/26=0.0001923076923076923\).
- Multinomial primary test: parametric Monte Carlo under the exact specified multinomial null, using the likelihood-ratio \(G^2\) statistic and plus-one p-value.
- Monte Carlo replicates per exact test: 199,999, in batches of 5,000.
- Minimum attainable exact Monte Carlo p-value: 0.000005, at least 38 times smaller than the gate.
- Log-beta mean: exact two-sided normal z test.
- Log-beta variance: exact two-sided chi-square test using the unbiased variance.
- Deterministic invariant tolerance: zero failures.
- Failure injection: all 7/7 preregistered defects had to be detected.

### Rare categories

Any category with expected count below 25 was declared rare before sampling. Primary inference retained every positive-probability category in the exact unpooled multinomial test. Rare categories were pooled only for a diagnostic table; no asymptotic chi-square gate was used when expected-count adequacy failed. Any observation in a structural-zero target would have caused immediate failure.

This rule applied to both rare-event tests, their cross-engine test, and all six vector-Poisson marginal tests. In the rare event fixture, the three loss categories each had expected count approximately 4.99975 from 300,000 draws. The historical observation was `(5, 0, 4)` and the independent observation was `(4, 4, 7)` across those three loss categories; neither zero observation was silently removed.

## Methods

### Explicit validation profiles

Four immutable profiles were instantiated without altering registry v0.3.0:

1. `E01-S07-MATCHED-COMMON-v1.0.0`: historical orientation with diagonal; categorical single events; fixed-size without-replacement fission with odd discard; event-index clock; first daughter.
2. `E01-S07-MATCHED-RARE-v1.0.0`: the same matched branch with deliberately rare positive event categories.
3. `E01-S07-MODERN-GILLESPIE-v1.0.0`: independent-only direct Gillespie timing; binomial-complement fission; uniformly random daughter.
4. `E01-S07-PAPER-POISSON-v1.0.0`: independent-only vector-Poisson reconstruction with exposure 0.25, componentwise loss clipping, retained batch overshoot, and explicit branch identity.

The fixtures instantiate named test branches only. They do not turn the sentinel-bearing registry into an executable default specification.

### Event selection

For each categorical/direct-Gillespie fixture, an analytical oracle independently computed join and leave propensities and normalized them to \(a_k/a_0\). The common matched fixture used 150,000 draws per engine. The rare matched fixture used 300,000 draws per engine. The independent modern fixture used 150,000 event and waiting-time draws.

The two engines were compared only on the two matched categorical fixtures. Their observed total-variation distance was tested against the null distribution obtained from paired independent multinomial samples from the same known analytical target. Exact cross-RNG trajectories were neither expected nor tested.

### Catalytic-matrix moments

Each engine generated 32,768 independent \(3\times3\) matrices, giving 294,912 logged beta entries per engine. The target was

\[
\log \beta_{ij}\sim\mathcal N(A=-4,\sigma^2=16).
\]

Mean and variance acceptance intervals were calibrated before sampling as `[-4.0274660366, -3.9725339634]` and `[15.8450949226, 16.1558385856]`, respectively.

### Mass and state invariants

Every categorical/direct-Gillespie record was checked for:

- mass change exactly \(+1\) or \(-1\);
- integer post-state reconstructed exactly from the selected event;
- nonnegative post-state;
- branch and RNG compatibility identity.

The vector-Poisson branch was separately checked for exact state reconstruction, componentwise applied-loss clipping, nonnegativity, fixed-exposure clock semantics, and explicit branch identity. It was also required to exhibit at least one non-unit mass change so that a batch kernel could not silently masquerade as a categorical single-event kernel.

### Fission

- Fixed even parent `(2,2,2)`: 120,000 draws per engine against the exact multivariate-hypergeometric law.
- Fixed odd parent `(2,2,1)`: 120,000 draws per engine against the exact joint law of child A and the uniformly discarded molecule.
- Independent binomial-complement parent `(2,3,1)`: 120,000 draws against the exact 24-outcome product-binomial law, plus a separate 50/50 daughter-choice test.

All samples were checked for exact parent/children/discard conservation, the specified child/discard sizes, and correct RNG-stream usage. Cross-engine comparisons were made only for the fixed-size even and odd branches.

### Independent-only branches

For the vector-Poisson reconstruction, 120,000 update batches were sampled and each of the six attempted-count marginals was tested against its exact Poisson law with an explicit tail bin. Applied loss counts were not substituted for attempted counts. For direct Gillespie timing, the probability-integral transform \(1-e^{-a_0\Delta t}\) was divided into ten equiprobable bins and tested for uniformity.

### Random-number semantics

Every raw task and inference task used an isolated S06-derived namespace. Each bundle contained all nine canonical PCG64DXSM streams: catalytic matrix, initial state, event, waiting time, fission, daughter selection, intervention, estimator, and machine learning. The seed manifest records:

- 13 raw-task bundles and 117/117 unique raw stream identities;
- 22 primary inference bundles and 198/198 unique inference stream identities;
- 3 additional independently derived failure-injection estimator streams;
- 318/318 unique recorded stream identities overall.

The historical harness consumes an explicit NumPy stream and remains labeled `NUMPY_GENERATOR_EXPLICIT_NOT_MATLAB_LEGACY`.

### Failure injection

Seven preregistered faults were injected into derived validation records, never into the canonical engine outputs:

1. move 3% event mass from the modal to the next category;
2. add 0.05 standard deviations to every log-beta value;
3. corrupt a categorical post-event mass by +2;
4. move 3% fission mass between categories;
5. add an unmatched molecule to a daughter;
6. move 3% Poisson probability mass between bins;
7. relabel the vector-Poisson branch as categorical single-event.

## Results

### Anchor results

| Result family | Tests | Result |
| --- | ---: | --- |
| Event frequencies versus \(a_k/a_0\) | 5 | 5/5 passed; p-values 0.030695–0.99113. |
| Legitimate cross-engine event comparisons | 2 | 2/2 passed; TV 0.002753 common and 0.001233 rare. |
| Log-beta mean/variance | 4 | 4/4 passed. |
| Fission target distributions and daughter choice | 6 | 6/6 passed; p-values 0.096055–0.91051. |
| Legitimate cross-engine fission comparisons | 2 | 2/2 passed; TV 0.002042 even and 0.004617 odd. |
| Vector-Poisson attempted-count marginals | 6 | 6/6 passed; exact p-values 0.34051–0.99415. |
| Direct-Gillespie waiting-time PIT | 1 | Passed; p=0.56631. |
| **Total primary family** | **26** | **26/26 passed; minimum p=0.030695 > 0.0001923077.** |

Non-rejection is evidence that these calibrated fixtures behave consistently with their specified distributions; it is not proof that the two implementations are identical or that an unavailable author implementation used the same semantics.

### Log-beta moments

| Engine | Entries | Sample mean | Target mean | Sample variance | Target variance | Mean p | Variance p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical-reference harness | 294,912 | -3.9961790 | -4 | 15.9737208 | 16 | 0.603931 | 0.528578 |
| Independent | 294,912 | -4.0018487 | -4 | 16.0738633 | 16 | 0.801824 | 0.076583 |

### Invariants

All 54 deterministic checks passed. This includes zero mass/reconstruction/nonnegativity/branch failures for categorical and direct-Gillespie events; finite positive beta values; exact even, odd-discard, and binomial fission conservation; three catalytic orientation/diagonal oracle checks; exact historical propensity-oracle agreement; and explicit RNG identity boundaries.

The vector-Poisson fixture produced 80,592 non-unit mass-change batches out of 120,000, satisfying the positive control that batch updates remain observably distinct from \(\pm1\) event semantics.

### Failure injection

All 7/7 injected faults were detected. The three distribution shifts reached the minimum possible Monte Carlo p-value, 0.000005; the log-beta shift had p approximately \(1.53\times10^{-168}\); and the three structural/identity corruptions triggered their exact detectors.

### Registry preservation

Registry v0.3.0 remained byte-for-byte unchanged at SHA-256 `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891`. No S07 registry update was made. Its execution flag remains false, all 64 unresolved/conflicting/evidence-deferred parameters remain present, and all 21 branch sets remain unexpanded.

## Diagnostic figures

- `diagnostic_event_probabilities.png`: standardized residuals for all event-frequency target tests, including the unpooled rare-event categories.
- `diagnostic_beta_moments.png`: exact log-beta moment p-values against the Bonferroni gate.
- `diagnostic_fission_probabilities.png`: analytical and sampled fixed-even, fixed-odd/discard, and binomial-complement probabilities.
- `diagnostic_independent_only_branches.png`: vector-Poisson attempted-count means and direct-Gillespie waiting-time PIT frequencies.

All four PNGs were visually inspected after generation. Labels, legends, axes, and plotted values were legible and consistent with the tables.

## Validation and completeness checks

### Canonical checks

- Preregistration contract and all 11 upstream hashes: pass.
- Canonical primary tests: 26/26 pass.
- Deterministic invariants: 54/54 pass.
- Failure injections: 7/7 detected.
- Rare-category exact-test routing: 9/9 applicable primary tests.
- Artifact hash/size audit before the final report: 17/17 manifested outputs matched.
- S07-focused tests after artifact generation: 12/12 pass.
- Remaining repository tests with the identity-pinned S06 regeneration test deselected: 60/60 pass.
- Targeted Ruff lint and format checks for all S07 code: pass.
- S08 artifact directory absence: pass.

### Preserved failed attempt

The first canonical invocation terminated before writing any outcome artifact because the runner attempted to read `rng_compatibility_id` from the historical `EventRecord`; the S04 API places this identity on the explicit `NumpyUniformSource` adapter and aggregate growth/fission records. No scientific outcome was inspected and no preregistered tolerance, fixture, seed, sample size, or analysis rule changed. The accessor was corrected, a real-engine raw-task regression test was added, and the correction was committed and pushed as `08033e3dfe1c5b63389ff91021f24ec5c07ea194`. The failed attempt is preserved in `canonical_attempt_01_failure.json`.

### Frozen S06 identity isolation

An unfiltered repository test run produced 60 passes and one expected failure: the S06 fresh-process exact-regeneration test rejects the current S07 Git commit because the S06 example is deliberately pinned to repository commit `1728b345da4b50e329c36983a436034bc25d6507`. The independent-engine and reproducibility-adapter source hashes themselves remain unchanged. S07 did not rewrite frozen S06 evidence to make a later repository identity appear identical. This is a provenance-bound isolation result, not an S07 stochastic failure.

Global `ruff check .` passed. Global `ruff format --check .` still reports five inherited S03 files as unformatted; S07 did not mechanically rewrite unrelated frozen-step sources. All five S07-owned files pass targeted formatting.

## Commands

Commands were run from `/workspace/arrival-of-self-replicators` with numeric libraries constrained to one thread:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python scripts/e01/freeze_s07_preregistration.py --artifacts-dir /artifacts

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python scripts/e01/run_s07_stochastic_validation.py --artifacts-dir /artifacts

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
pytest -q tests/e01/test_stochastic_validation.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
pytest -q -k 'not test_generated_artifacts_and_fresh_process_regeneration_when_present'

ruff check src/e01_gard_validation \
  scripts/e01/freeze_s07_preregistration.py \
  scripts/e01/run_s07_stochastic_validation.py \
  tests/e01/test_stochastic_validation.py

ruff format --check src/e01_gard_validation \
  scripts/e01/freeze_s07_preregistration.py \
  scripts/e01/run_s07_stochastic_validation.py \
  tests/e01/test_stochastic_validation.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python scripts/e01/run_s07_stochastic_validation.py \
  --artifacts-dir /artifacts --finalize-manifest
```

## Dependencies, runtime, and resources

- Python 3.13.14.
- NumPy 2.4.6; SciPy 1.18.0; Matplotlib 3.11.1; PyYAML 6.0.3; jsonschema 4.26.0; pytest and Ruff from the supplied environment.
- CPU-only S07 execution using four process workers, with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` all fixed to 1.
- Visible runtime: 24 logical Intel Xeon CPUs; two NVIDIA L4 GPUs were visible but unused by S07.
- Canonical wall time recorded by the runner: 162.94 seconds. The longest raw task was the 300,000-draw independent rare-event task at 115.50 worker-seconds.
- No dependency installation, network fetch, GPU computation, or source-registry update occurred in S07.

## Artifacts written

All outputs are under `/artifacts/research_steps/S07/`:

- `preregistration.yaml`
- `preregistration_record.json`
- `calibrated_tolerances.json`
- `validation_fixtures.json`
- `seed_manifest.json`
- `goodness_of_fit_summary.csv`
- `goodness_of_fit_details.json`
- `moment_tests.csv`
- `invariant_checks.csv`
- `failure_injection.json`
- `diagnostic_event_probabilities.png`
- `diagnostic_beta_moments.png`
- `diagnostic_fission_probabilities.png`
- `diagnostic_independent_only_branches.png`
- `registry_preservation.json`
- `canonical_attempt_01_failure.json`
- `validation_summary.json`
- `artifact_manifest.json`
- `research_step_full_results.md`

Repository-backed reproducible code remains in Git rather than being copied into the artifact directory:

- `configs/e01/s07_stochastic_validation_preregistration.yaml`
- `src/e01_gard_validation/`
- `scripts/e01/freeze_s07_preregistration.py`
- `scripts/e01/run_s07_stochastic_validation.py`
- `tests/e01/test_stochastic_validation.py`

## Provenance

- Repository: `/workspace/arrival-of-self-replicators`
- Branch: `eidosoma/groups/42`
- Preregistration commit: `6490bb48a2780cdb73afa6ebe8da697496c4803d`
- Canonical corrected runner commit: `08033e3dfe1c5b63389ff91021f24ec5c07ea194`
- Both commits were pushed to `origin/eidosoma/groups/42` before handoff.
- The final self-excluding artifact manifest records SHA-256 and size for every compact S07 output plus all governing inputs and Git-backed code files.

## Caveats, blockers, failed assumptions, and limitations

1. **No author-code identity.** Neither engine is the unavailable author implementation, and passing these tests cannot establish that identity.
2. **No MATLAB RNG identity.** The historical engine uses explicit NumPy PCG64DXSM draws for distributional validation. Exact legacy MATLAB algorithm, global-state ordering, and trajectories remain unresolved.
3. **Branch-limited comparison.** Cross-engine tests were restricted to identical explicit categorical and fixed-size-fission branches. Modern direct-Gillespie, binomial-complement, and reconstructed vector-Poisson behavior were validated independently.
4. **Paper reconstruction remains conditional.** The Poisson exposure, clipping, stopping, and daughter choices are explicit fixture values, not silent paper or author defaults.
5. **Finite statistical power and resolution.** Passing goodness-of-fit tests does not prove equality. Exact Monte Carlo p-values are resolved to 0.000005, and the campaign covers fixed small systems rather than the full scientific parameter space.
6. **Floating-point scope.** Calculations used CPU float64 under the S06 one-thread precision policy. No claim of byte-identical cross-platform outcomes is made.
7. **Registry remains closed.** All 64 unresolved sentinels and 21 branch sets remain unresolved/unexpanded; S07 introduced no registry default.
8. **Historical licensing remains unresolved.** S07 adds no redistribution claim for the pinned historical source.
9. **No scientific-analysis claim.** S07 validates stochastic machinery only. It does not reconstruct self-replicator labels, compute \(\Phi^r\), or assess any paper association, prediction, or intervention result.

There is no blocker to returning control. Any S08 work requires a separate instruction.
