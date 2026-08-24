# S05 full results — Construct an independent Python GARD engine

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S05** — Construct an independent Python GARD engine |
| Completion status | **Complete** on 2026-08-01. Only S05 was executed; S06 was not begun. |
| Artifacts written | Git-backed independent engine and tests at commit `be698bd2c3b0dc9b46676b99b2230c5a2274662c`; versioned engine contract and three validation-only profiles; API surface and branch catalog; complete diagnostic event/fission logs; 22-check invariant/provenance audit; five substantive matched-branch comparisons with eight gates; benchmark; registry-preservation audit; validation summary; artifact manifest; and this canonical report. |
| Validation result | **PASS** — 22/22 unit, source-independence, frozen-evidence, and registry checks; 8/8 predeclared distributional gates; exact agreement across 512 propensity cases; 120,000 event draws per engine; 50,000 even and 30,000 odd fissions per engine; 20,000 one-generation draws per engine; 11/11 focused tests; 35/35 repository tests; Ruff; formatting; whitespace; Git push; remote-ref equality; and S06-absence checks passed. |
| Outcome classification | **Supportive** for S05's bounded hypothesis: a separately structured NumPy engine reproduces the S04 public-historical model-level distributions when both are configured to the same explicit branch. This is not a replication verdict for the paper's scientific claims. |
| Caveats or blockers | The author implementation is unavailable; legacy MATLAB and cross-RNG trajectories are not expected to match; the registry remains closed with 64 unresolved items and 21 unexpanded branch sets; paper vector-Poisson and modern Gillespie fixtures were not compared with S04 because S04 does not implement those branches; S05 diagnostic records are not the canonical S06 seed/serialization schema. |
| Lay summary | A second GARD simulator was written independently and given explicit switches for the disputed update, growth, fission, daughter, initialization, and timing rules. When both simulators were asked the same historical-reference question, their event, split, and one-generation distributions agreed well inside thresholds fixed before the run. |
| Recommended next action | **Stop after S05 and return control.** S06 is eligible but unstarted; freeze random-number semantics and the canonical data schema only if separately authorized. |

## Frozen question

Can a separately written implementation reproduce the same model-level distributions?

The completion criterion was no unexplained model-level disagreement on small, explicitly matched benchmarks. Exact paths were neither required nor appropriate because S04 uses legacy-compatible uniform selection semantics while S05 uses separate modern NumPy generators and different stochastic primitives.

## Decision

Yes, within the matched public-historical branch and the declared Monte Carlo resolution. The independent and S04 engines agreed on:

- all 512 deterministic propensity-array cases to a maximum absolute error of `0.0`;
- the six-category event law with two-engine total variation `0.0021583`;
- even fixed-size fission with total variation `0.01274`;
- odd fixed-size fission plus discard identity with total variation `0.0141333`;
- one-generation followed-daughter endpoints with total variation `0.0077`; and
- growth event counts, whose means were `3.3156` and `3.3121` and differed by only `0.40986` pooled standard errors.

Every value passed its prospectively recorded S05 tolerance. The result establishes distributional compatibility for that complete validation profile. It does not collapse historical, paper-reconstruction, and modern Gillespie branches into one model, and it does not establish identity with unavailable author code.

## Lay summary

The two engines reach the same kinds of outcomes at the same frequencies when they are given the same explicit historical rules. They do not produce the same molecule-by-molecule movie, because their random-number machinery is deliberately different. That is expected: matching distributions is the scientifically relevant test here.

The new engine also exposes the choices that the paper leaves open. A run must say, for example, whether updates are single events, Gillespie events with waiting times, or vector-Poisson batches; whether fission fixes daughter size or flips binomial coins; and whether the first, second, or a randomly selected daughter continues. Unknown registry values cannot be passed through as if they were defaults.

## Inputs

### Governing and paper inputs

| Input | S05 use | Identity or boundary |
| --- | --- | --- |
| `/workspace/AGENTS.md` | Execution, artifact, testing, Git, and handoff contract | Refreshed before work |
| `/workspace/FULL_PLAN.md` | Frozen catalytic transform, boost and join/leave equations, Gillespie interpretation, fission reconstruction, logging, and seed-separation constraints | Refreshed in full; historical `k_f`, `k_b`, and uniform `rho` remain reference-only |
| `/workspace/RESEARCH_PLAN.md` | S05 question, outputs, validation, completion, and strict S06 boundary | Refreshed before S05 and updated only after completion |
| `/workspace/input-attachments/MANIFEST.json` | Attachment route and original-file metadata | Read before attachment content |
| Attachment `_metadata/ATTACHMENT.md` | Required sidecar and materialization boundary | Original binary not mounted through the attachment path |
| Attachment `pdf-markdown.md` | Accessible complete paper extraction | Methods lines 61–69 state without-replacement initialization, vector-Poisson updates, binomial `p=0.5` fission, and paper constants |
| Official original paper PDF | Equation-bearing original source | arXiv `2607.28250v1`; 18 pages; 1,117,911 bytes; SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4` |

The official PDF was rehashed and its GARD Methods pages were re-extracted directly with `pdftotext -layout`. This confirmed the paper wording without treating it as a complete executable algorithm.

### Prior research-step inputs

- S01–S04 canonical full-results reports.
- S01 claim ledger and source reconciliation.
- S02 ambiguity/discrepancy ledger and registry lineage.
- S03 source and environment manifests, dependency locks, precision policy, and clean-environment result.
- S04 historical behavior contract, compatibility notes, source-traceability table, verified small cases, and Git-backed reference API.

### Frozen machine-readable evidence

| Evidence | SHA-256 | S05 action |
| --- | --- | --- |
| `specification_registry_v0.3.0.yaml` | `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891` | Preserved byte-for-byte; no S05 updates |
| `source_manifest.yaml` | `50a569e30826fe893258f5c0935469576008f43134eee94e8bf2654d4ef23ed5` | Verified before artifact generation |
| `environment_report.json` | `021c6f848e01172c098e615f27babcf6748dbba1f8bd0f1374883d9d392ef2cd` | Verified before artifact generation |
| `historical_behavior_contract.yaml` | `e6fe49aba2240047d018e5b619ef07d3e48922fb43a963256b6b2233f07d0a43` | Used only to bound legitimate S04 comparisons |

Registry v0.3.0 still contains 120 parameters, 64 unresolved/conflict/evidence-deferred items, 21 unexpanded branch sets, 85 execution blockers, `executable: false`, and `noSilentDefaults: true`.

## Detailed methods

### 1. Independent architecture

The production package is `src/e01_gard_independent/`, split into five modules:

- `specification.py`: enums, complete immutable specification, and fail-closed mapping loader;
- `rng.py`: six caller-owned, pairwise-distinct NumPy generator inputs and state digests;
- `records.py`: immutable propensity, event, growth, fission, generation, and lineage records;
- `engine.py`: matrix generation, initialization, propensity calculation, update kernels, growth, fission, daughter selection, and lineage execution; and
- `__init__.py`: explicit public API and version.

None of these modules imports or invokes `e01_gard_historical`. The S05 validation builder is the only new file allowed to import both packages, and it does so solely for matched-branch comparisons. S05 uses NumPy's categorical choice, Poisson, binomial, and multivariate-hypergeometric primitives rather than S04's explicit-tape and strict cumulative-selection control path.

### 2. Fail-closed specification contract

`GardSpecification` has 27 required fields and no dataclass defaults. Construction validates all numeric values and every compatible branch combination. Mapping-based construction rejects:

- missing or extra fields;
- raw `UNRESOLVED::`, `CONFLICT::`, `BRANCH_SET::`, and `UNAVAILABLE::` sentinels;
- raw strings where a typed branch enum is required;
- shared generator objects or duplicated stream IDs;
- inconsistent kernel/clock/loss/boundary combinations;
- missing or irrelevant Poisson exposure;
- missing or irrelevant fission probability;
- inconsistent profile-role and propensity/update branches; and
- incompatible bounded/unbounded maximum-step configurations.

Three fully explicit small profiles were frozen before full comparison execution:

| Specification ID | Purpose | Update/time | Fission/daughter | Comparison status |
| --- | --- | --- | --- | --- |
| `E01-S05-HISTORICAL-DISTRIBUTION-COMPARISON-v1.0.0` | Match the public S04 transition law at explicit fixture values | Categorical single event; event index only | Fixed-size without replacement with odd discard; first daughter | Eligible for S04 comparison |
| `E01-S05-MODERN-GILLESPIE-FIXTURE-v1.0.0` | Validate the modern interpretation of the frozen rates | Direct Gillespie; independent exponential waiting stream | Binomial complement; uniformly selected daughter | Internal S05 validation only |
| `E01-S05-PAPER-POISSON-FIXTURE-v1.0.0` | Validate one completely specified paper-prose reconstruction | Vector Poisson at exposure `0.25`; clipping; retained overshoot | Binomial complement; first daughter | Internal S05 validation only |

Their small kinetics and reservoir vectors are fixture values, not recovered author values or a primary paper configuration. The historical comparison profile uses `k_f=0.4`, `k_b=0.02`, and uniform three-species `rho` solely to give well-conditioned, short validation trajectories.

### 3. Integer states and catalytic matrices

All state-bearing APIs copy into signed 64-bit integer arrays after rejecting empty vectors, negative counts, nonintegral counts, nonfinite values, and dimension mismatches. The catalytic matrix generator uses a caller-owned generator and the frozen transform

\[
\beta=\exp(A+\sigma\epsilon),\qquad \epsilon\sim\mathcal N(0,1).
\]

Orientation and diagonal behavior are explicit branches: historical orientation with diagonal, transposed with diagonal, or historical orientation with a zero diagonal. The engine does not infer one from matrix shape or source identity.

### 4. Explicit propensity arrays

For state mass \(N>0\), the engine computes

\[
b=1+\frac{\beta n}{N},\qquad
a^+=(k_f\rho N)\odot b,\qquad
a^-=(k_b n)\odot b.
\]

Every `PropensityArrays` record contains the boost, join, leave, concatenated, normalized probability arrays, total propensity, and the specification's equation-branch ID. Zero-count loss rates are exactly zero. A nonempty zero-total state follows an explicit `stop` or `raise` branch.

### 5. Separated modern RNG inputs

`RNGStreams` requires six distinct `numpy.random.Generator` objects with distinct caller-supplied IDs:

1. catalytic matrix;
2. initialization;
3. event or Poisson update;
4. Gillespie waiting time;
5. fission; and
6. daughter choice.

The engine creates no implicit generator and has no seed convenience constructor. Diagnostic records hash each used generator's bit-generator state before and after use. Literal S05 fixture seeds are recorded only as validation inputs; no canonical seed tuple, hierarchy, serialization contract, or exact-regeneration promise was defined, because those are the frozen purpose of S06.

### 6. Update kernels and growth limits

The independent engine supports three deliberately separate update families:

- **Categorical single event:** sample one join or leave from `a/a0`; record a mass change of exactly `+1` or `-1`; no physical-time claim.
- **Direct Gillespie:** use the same categorical law and draw `Delta t ~ Exponential(rate=a0)` from the separate waiting-time generator.
- **Vector-Poisson batch:** draw per-species join and attempted-loss counts at an explicit exposure, then apply the chosen loss-nonnegativity and batch-boundary rules.

Growth stops on the explicit `n_max`, extinction, zero propensity, or bounded maximum-step result. Maximum-step behavior is separately identified as fission-current-state, stop-without-fission, or raise. An unbounded branch exists only for the matched historical comparison, whose S04 source has no `max_steps` condition.

### 7. Fission and daughter semantics

The two fission branches are:

- **Fixed-size without replacement:** NumPy's multivariate-hypergeometric sampler assigns `floor(N/2)` molecules to the first daughter; an independent one-molecule hypergeometric draw is recorded as discard for odd parent mass; the second daughter is the remainder.
- **Binomial complement:** independent per-species `Binomial(n_i,p)` counts define the first daughter and the second is the exact complement.

First, second, and uniform-random daughter selection are separate choices. The random choice uses the dedicated daughter generator. Deterministic first/second selection records equal pre/post daughter-stream digests, proving that no hidden daughter draw was consumed. Post-fission continuation either accepts the exact selected daughter or raises if it is empty.

### 8. Complete diagnostic event logging

Every event record contains:

- specification, propensity branch, update kernel, clock branch, generation, and step identity;
- full integer pre/post states, masses, and mass delta;
- boost, join, leave, normalized event, and total propensity arrays;
- selected event/species/probability or attempted/applied batch join/loss arrays;
- boundary action and time increment/cumulative time where applicable; and
- event and waiting stream IDs plus before/after RNG-state hashes.

Every fission record contains the parent, both daughters, discard, selected daughter label and state, all fission/daughter/post-fission branch IDs, conservation flag, stream IDs, daughter-consumption flag, and before/after RNG-state hashes. The artifact fixture exercises all three profiles. These are complete S05 engine diagnostics, not the canonical storage or regeneration schema reserved for S06.

### 9. Distributional comparison design

Comparisons used only `E01-S05-HISTORICAL-DISTRIBUTION-COMPARISON-v1.0.0`. The paper vector-Poisson and modern Gillespie profiles were excluded because there is no like-for-like S04 branch.

The Monte Carlo sizes and gates were written to `s05_specification_profiles.yaml` before the full run:

| Scope | Full size per engine | Metric | Gate |
| --- | ---: | --- | ---: |
| Random propensity fixtures | 512 cases | Maximum absolute error | `<=1e-12` |
| Fixed-state event categories | 120,000 | Two-engine and per-engine target TV | `<=0.015` |
| Even fixed-size fission | 50,000 | Two-engine TV | `<=0.04` |
| Odd fission plus discard | 30,000 | Two-engine TV | `<=0.04` |
| One-generation endpoint | 20,000 | Two-engine TV | `<=0.04` |
| One-generation event count | 20,000 | Absolute mean difference / pooled SE | `<=5.0` |

These are conservative S05 engineering-agreement tolerances, not the calibrated multinomial and goodness-of-fit program assigned to S07. No p-value was used as a brittle pass/fail gate.

## Commands and dependencies

Principal implementation and validation commands were:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
  -q -p no:cacheprovider tests/e01/test_independent_engine.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python scripts/e01/build_independent_engine_artifacts.py \
  --artifacts-dir /artifacts --workers 4

ruff check scripts/e01 tests/e01 src/e01_gard_historical src/e01_gard_independent
ruff format --check src/e01_gard_independent \
  scripts/e01/build_independent_engine_artifacts.py \
  tests/e01/test_independent_engine.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider

git diff --check
git commit -m "Add E01 S05 independent GARD engine"
git push origin eidosoma/groups/42
git ls-remote origin refs/heads/eidosoma/groups/42
```

| Dependency or resource | Value |
| --- | --- |
| Python | 3.13.14 |
| NumPy | 2.4.6 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| Ruff | 0.16.0 |
| Worker processes | 4 for independent comparison tasks |
| Numeric-library threads | 1 each for OMP, OpenBLAS, MKL, and Numba |
| GPU | Not used; integer stochastic simulation and small-array comparisons were CPU tasks |
| New installed dependencies | None |

The artifact builder also has a quick mode used only by the repository round-trip test. Canonical S05 artifacts were generated without `--quick` at the full declared sizes.

## Results

### Anchor result 1: exact propensity agreement

Across 512 randomly generated three-species integer states and positive catalytic matrices, the independent and S04 engines had maximum absolute errors of exactly zero for boost, join, leave, and total arrays.

The independent hand oracle also reproduced:

| Quantity | Expected and actual |
| --- | --- |
| Boost | `(7/3, 4/3)` |
| Join propensities | `(0.175, 0.300)` |
| Leave propensities | `(14/15, 4/15)` |
| Total propensity | `1.675` |

### Anchor result 2: event distributions

For 120,000 draws from each engine on the same fixed state, matrix, rates, and reservoir:

| Comparison | Observed TV | Gate | Result |
| --- | ---: | ---: | --- |
| Independent versus S04 | `0.0021583` | `0.015` | PASS |
| Independent versus analytical `a/a0` | `0.0020884` | `0.015` | PASS |
| S04 versus analytical `a/a0` | `0.0008062` | `0.015` | PASS |

The engines consumed unrelated RNG streams and were never compared event by event.

### Anchor result 3: fission distributions

| Parent and outcome | Draws per engine | Support size | Two-engine TV | Gate | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Even parent `(4,3,2,1)`, first child | 50,000 | 22 | `0.01274` | `0.04` | PASS |
| Odd parent `(3,2,2)`, first child plus discard identity | 30,000 | 19 | `0.0141333` | `0.04` | PASS |

Every independent and historical split conserved the parent when the explicit odd-parent discard was included. The comparison shows that S05's multivariate-hypergeometric construction matches the distribution of S04's sequential without-replacement algorithm without copying its draw-by-draw control flow.

### Anchor result 4: one-generation model outcomes

Across 20,000 independent draws per engine from initial state `(1,1,1)` to `n_max=6` and fixed-size fission:

| Metric | Independent | S04 | Agreement result |
| --- | ---: | ---: | --- |
| Mean growth events | `3.3156` | `3.3121` | Difference `0.0035`; standardized difference `0.40986 <= 5.0` |
| Followed-daughter endpoint distribution | — | — | TV `0.0077 <= 0.04` |

Both gates passed. This is the closest S05 test to an end-to-end model-level transition, while remaining small enough to diagnose and restricted to truly matched semantics.

### Unit, independence, and provenance checks

`unit_invariants.json` contains 22 passing checks:

- nine state, fission-conservation, and log-schema checks across the three profiles;
- one exact hand-propensity oracle;
- one raw-sentinel rejection check;
- five package-file plus one module-set source-independence checks; and
- four frozen-file hash checks plus one registry-gate preservation check.

The focused test file contains 11 tests covering specification completeness, sentinel rejection, separated RNGs, integer states, both initializers, catalytic branches, hand propensities, categorical/Gillespie/Poisson logs, explicit batch-loss failure, growth limits, both fission families, daughter consumption, empty-daughter handling, lineage logs, source independence, frozen registry identity, artifact round trip, and S06 absence.

### Benchmark result

The final four-worker artifact run completed its slowest individual task—the 120,000-draw-per-engine event comparison—in about 76 seconds on this runtime. Recorded per-task diagnostic throughputs ranged from roughly 303 one-generation draws/s/engine to 1,770 propensity cases/s. These timings include immutable-record and RNG-state hashing overhead and are not scientific pass criteria. Exact canonical timings are in `benchmark.json` and may vary across reruns.

### Registry result

No registry parameter was changed. The preservation audit records:

- registry SHA-256 unchanged;
- 120 parameters;
- 64 unresolved parameters;
- 21 unexpanded branch sets;
- `executable: false`;
- `noSilentDefaults: true`; and
- `s05RegistryUpdates: []`.

## Validation

| Validation layer | Result |
| --- | --- |
| Complete specification fields | PASS; 27/27 mandatory, no dataclass defaults |
| Sentinel/conflict/branch rejection | PASS |
| Pairwise-distinct RNG objects and IDs | PASS |
| Integer/nonnegative state checks | PASS |
| Hand propensities | PASS |
| Categorical mass change | PASS; every tested event exactly `+1` or `-1` |
| Gillespie separated waiting stream | PASS |
| Vector-Poisson nonnegative state and explicit error injection | PASS |
| Growth boundary and max-step branches | PASS |
| Fixed-size and binomial fission conservation | PASS |
| Deterministic versus random daughter-stream consumption | PASS |
| Complete diagnostic record fields | PASS across all three profiles |
| Production-package S04 import prohibition | PASS across five modules |
| Frozen evidence hashes | PASS, 4/4 |
| Registry preservation | PASS |
| Matched-branch distribution gates | PASS, 8/8 |
| Focused S05 tests | PASS, 11/11 |
| Full repository tests | PASS, 35/35 |
| Ruff lint | PASS |
| Scoped Ruff format | PASS |
| Git whitespace | PASS |
| Commit, push, remote equality | PASS at `be698bd2c3b0dc9b46676b99b2230c5a2274662c` |
| S06 absence | PASS |

The canonical machine-readable results are `validation_summary.json`, `unit_invariants.json`, `distributional_agreement.csv`, `distributional_agreement_details.json`, and `registry_preservation.json`.

## Failed attempts, refinements, and recoveries

- An initial Ruff invocation passed the two YAML files as direct operands, causing Ruff to parse YAML as Python and emit syntax errors. The YAML itself loaded correctly. Final lint targets Python paths only, matching the repository convention.
- The first staged diff check identified one trailing blank line at the end of each new YAML file. Both were removed and the clean staged diff check passed.
- Validation initially used distinct generator objects but repeated literal seeds for several unused streams. Before commit, those fixture inputs were changed to distinct literal seeds as well, making both object separation and visible fixture identities unambiguous. The scientifically used event/fission seeds and all reported distributions were unchanged.
- A quick-mode artifact build was run under `/cache` to validate the writer and schemas before the full run. It was not promoted as evidence; canonical artifacts use the preregistered full sizes.

No failed unit invariant or scientific agreement gate was hidden. Final machine-readable failure lists are empty.

## Artifacts written

### Shared independent-engine directory

| Path | Purpose |
| --- | --- |
| `/artifacts/E01_forensic_replication_bundle/software/independent_engine/engine_pointer.json` | Git identity, package hashes, and author/RNG identity boundary |
| `/artifacts/E01_forensic_replication_bundle/software/independent_engine/independent_engine_contract.yaml` | Versioned equations, branches, fail-closed rules, logging contract, and comparison boundary |
| `/artifacts/E01_forensic_replication_bundle/software/independent_engine/validation_profiles.yaml` | Three complete validation-only specifications plus full Monte Carlo sizes and gates |
| `/artifacts/E01_forensic_replication_bundle/software/independent_engine/branch_catalog.csv` | Machine-readable implemented-branch inventory and S04 eligibility |
| `/artifacts/E01_forensic_replication_bundle/software/independent_engine/api_surface.json` | Public symbols, function signatures, and no-default field audit |
| `/artifacts/E01_forensic_replication_bundle/software/independent_engine/diagnostic_event_log_fixture.json` | Complete event and fission records for all three profiles |
| `/artifacts/E01_forensic_replication_bundle/software/independent_engine/benchmark.json` | Worker/thread policy and task timing/throughput |

### S05 handoff directory

| Path | Purpose |
| --- | --- |
| `/artifacts/research_steps/S05/research_step_full_results.md` | This canonical S05 handoff |
| `/artifacts/research_steps/S05/validation_summary.json` | Machine-readable workflow fields and validation rollup |
| `/artifacts/research_steps/S05/unit_invariants.json` | Twenty-two unit, independence, evidence-hash, and registry checks |
| `/artifacts/research_steps/S05/distributional_agreement.csv` | Eight compact pass/fail comparison gates |
| `/artifacts/research_steps/S05/distributional_agreement_details.json` | Counts, supports, sample sizes, distances, means, and boundary statement |
| `/artifacts/research_steps/S05/registry_preservation.json` | Exact v0.3.0 unchanged/gate audit |
| `/artifacts/research_steps/S05/artifact_manifest.json` | Input, code, output, size, and SHA-256 provenance |

`RESEARCH_PLAN.md` does not require a distinct S05 `status.json`, so none was written. `validation_summary.json` includes the requested workflow fields without being relabeled as a status file.

### Repository-backed source

| Repository path | Role |
| --- | --- |
| `src/e01_gard_independent/` | Independent engine, records, RNG inputs, and specification model |
| `configs/e01/s05_independent_contract.yaml` | Versioned implementation/evidence contract |
| `configs/e01/s05_specification_profiles.yaml` | Explicit profiles, validation sizes, literal fixture seeds, and gates |
| `scripts/e01/build_independent_engine_artifacts.py` | Reproducible parallel validation and artifact builder |
| `tests/e01/test_independent_engine.py` | Focused engine, contract, invariant, and artifact tests |

Repository code was kept in Git and was not copied into artifacts.

## Provenance

- Date: 2026-08-01 UTC.
- Repository: `/workspace/arrival-of-self-replicators`.
- Branch: `eidosoma/groups/42`.
- S05 commit: `be698bd2c3b0dc9b46676b99b2230c5a2274662c`.
- Push: successful; remote `refs/heads/eidosoma/groups/42` resolved to the same commit.
- Independent engine contract: `E01-independent-engine-v1.0.0`.
- Engine version: `1.0.0`.
- Profile collection: `E01-S05-profiles-v1.0.0`.
- Historical comparison source: public GARD commit `86dff6320d5ae91b4e831471079ff46749b14df9`, never relabeled as author code.
- Official paper: arXiv `2607.28250v1`, PDF SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- Exact input, repository-code, and output hashes are in `/artifacts/research_steps/S05/artifact_manifest.json`; the manifest excludes its own hash.

## Caveats, blockers, failed assumptions, and limitations

### Caveats and blockers

- The exact author implementation remains `UNAVAILABLE::NO_AUTHOR_CODE_RELEASE_FOUND`. Neither engine is labeled as that code.
- S05 uses modern NumPy generators and primitives. It does not reproduce legacy MATLAB `rand`/`randn`, array-fill order, global-state resets, or exact historical draw consumption.
- Distributional agreement is confined to one fully matched public-historical branch and explicit fixture values. It does not establish global equivalence for all parameter regimes.
- The historical comparison's unbounded growth semantics can theoretically run indefinitely. The validated profile has strong positive growth and completed all 20,000 draws; future production execution still requires the S06/S12 run-management contract.
- Paper vector-Poisson exposure, loss clipping, boundary handling, maximum-step behavior, daughter choice, and environment remain author-unknown. S05's paper profile names one fixture branch and never promotes it to default.
- The modern Gillespie profile is an independent interpretation of the FULL_PLAN equations, not a claim about the pinned modern GARD repository's complete behavior.
- Registry v0.3.0 remains intentionally non-executable. S05 did not use outcome data to choose among any branch set.
- Complete S05 diagnostics do not replace S06's planned canonical seed and trajectory schema or exact same-engine regeneration tests.
- Benchmark throughput includes cryptographic state hashing and Python record construction; it is not a scale-up performance promise.

### Failed assumptions avoided or exposed

- Exact cross-RNG trajectories were not used as a success criterion.
- Agreement with the S04 historical engine was not generalized to paper vector-Poisson or modern Gillespie semantics.
- S04's first-child, fixed-size fission, with-replacement initialization, or historical rate constants were not adopted as author defaults.
- NumPy's multivariate-hypergeometric fission did not need to copy S04's sequential control flow to reproduce its distribution.
- A single shared RNG was not necessary for a historical-law comparison; separated streams retained the same model-level transition distribution.

### Limitations

- S05 is an implementation and cross-engine engineering step, not the planned calibrated stochastic validation in S07.
- No catalytic-matrix moment goodness-of-fit, rare-event pooled test, or broad parameter sweep was performed; those remain S07.
- No seed schema, canonical event serialization, or exact regeneration contract was frozen; those remain S06.
- No self-replicator labels, compositional transforms, Phi-r values, prediction model, intervention, or paper figure/table claim was evaluated.
- Simulation remains computational proxy evidence and does not validate real prebiotic chemistry or biological agency.

## Recommended next action

Stop after S05 and return control to the Chief Scientist. S06 is eligible but has not started. If separately authorized, S06 should freeze the seed derivation and canonical trajectory schema around this engine without changing S05's model semantics, and it should retain the author-code and legacy-MATLAB RNG boundaries.
