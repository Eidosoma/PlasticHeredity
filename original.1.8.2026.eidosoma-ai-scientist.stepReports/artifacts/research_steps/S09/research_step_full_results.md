# S09 full results — Resolve compositional zeros transparently

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S09 — Resolve compositional zeros transparently** |
| Completion status | **Complete** on 2026-08-01. Only S09 was executed; S10 was not begun. |
| Artifacts written | A preregistered, Git-backed transform library and Draft 2020-12 schema; 13 explicit zero treatments; all dimension-2 and dimension-4 dropped-CLR choices; full CLR, Helmert ILR, raw, Hellinger, and data-scoped principal-log-ratio controls; 4,901 checksum-protected lossless transform rows plus CSV; a 247-entry fixture-level valid-specification registry; zero, eligibility, conditioning, covariance-rank, inverse, representation, and replacement diagnostics; eight failure injections; exact-replay and visual-validation evidence; three figures; registry-preservation and provenance manifests; and this report. |
| Validation result | **PASS.** All 45 canonical checks passed: 28/28 frozen inputs and 17/17 provenance, scope, row-coverage, grid, dropped-component, finite, inverse, basis, isometry, schema/checksum, failure-injection, zero-reconciliation, diagnostics, figure, and S10-absence gates. All 4,901 expected rows were emitted; 4,745 eligible results were finite; all 156 ineligible results had reasons; maximum inverse error was `1.9984e-15`; 178/178 evaluable isometries and 8/8 failure injections passed. Four anchor outputs replayed byte-exactly. Focused tests passed 8/8 and the non-identity-pinned repository suite passed 82/82. |
| Outcome classification | **Supportive** for the bounded S09 numerical-preprocessing hypothesis. The result is also constraining: full CLR and raw closure are structurally covariance-singular, dropped CLR is non-isometric, multiplicative replacement cannot define an all-zero composition, and one no-replacement PLR fixture branch was not evaluable. |
| Caveats or blockers | No author zero policy, pseudocount, replacement rule, feature orientation/scaling, invalid-transform policy, or paper-primary specification was recovered. Registry v0.3.0 remains unchanged and non-executable. These are validated fixture branches, not trajectory-scale estimator validation or retroactive S08 labels. |
| Lay summary | Zero counts no longer disappear or turn silently into infinities. Every tested observation is represented under every declared branch, either by finite reversible coordinates or by an explicit reason that the branch is mathematically undefined. Small pseudocounts and multiplicative replacement were close, while larger values increasingly changed the geometry. Removing one CLR coordinate fixes linear dependence but makes distances depend on which coordinate was removed; ILR preserves the full Aitchison geometry without that defect. |
| Recommended next action | **Stop and return control to the Chief Scientist.** S10 is eligible but unstarted and requires separate authorization. When authorized, use only immutable accepted S09 specification IDs and retain zero-treatment/coordinate sensitivity; do not select a paper default from fixture outcomes. |

## Frozen question and decision

**Question:** Can explicit zero treatments and compositional coordinate systems transform every validated input observation without infinities, hidden row deletion, or unexplained singular failure, while retaining all author-method uncertainty and branch identity?

**Decision:** Yes, within each branch's declared mathematical domain. The canonical output contains exactly one status-bearing record for each input observation and fixture-level complete specification. Every eligible value was finite and invertible within the frozen `1e-12` absolute/relative tolerances. Undefined cases were retained as `INELIGIBLE` with reasons rather than dropped or coerced.

This decision does not identify the authors' preprocessing. The paper's source-supported branch is limited to relative composition, natural-log-like CLR prose, and removal of the final component. Its zero handling is unstated. S09 therefore validates separately named reconstruction branches and leaves registry sentinels intact.

## Lay summary

Log-ratio methods cannot take the logarithm of zero. A tempting implementation shortcut is to delete those rows, replace zeros without recording how, or let infinities propagate. S09 prevents all three. It records the zero frequency first, applies six declared pseudocount values and a directly matched multiplicative alternative, and retains a no-replacement control.

The transforms were then expressed in several coordinate systems. Full CLR and ILR produce the same Aitchison distances, but full CLR has one redundant direction. Dropping a CLR coordinate is reversible and reproduces the paper's stated “remove the last component” approach, yet ordinary Euclidean distances change with the removed component. ILR uses an orthonormal simplex basis, preserves the full Aitchison distance exactly within floating-point tolerance, and avoids the redundant coordinate.

This is a numerical and provenance result on validated fixtures. It is not evidence that one delta, zero rule, or coordinate system reproduces the unavailable author code or the eventual paper results.

## Inputs and evidence boundary

### Governing and paper inputs

The preregistration froze and verified 28 inputs before canonical outcomes:

- `/workspace/AGENTS.md`, `/workspace/FULL_PLAN.md`, and the pre-S09 `/workspace/RESEARCH_PLAN.md`;
- `/workspace/input-attachments/MANIFEST.json` and the required `_metadata/ATTACHMENT.md` sidecar;
- the supplied Docling paper extraction and the S03-recovered official arXiv `2607.28250v1` PDF, SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`;
- all S01–S08 canonical full-results reports;
- registry v0.3.0, source/environment manifests, S08 label contract/configurations/schema, S08 fixture/label/zero validation artifacts, and S06 seed/precision/trajectory contracts.

The paper's Data preprocessing section states that relative molecular compositions were CLR-transformed and that the final component was removed because CLR makes covariance singular. It does **not** state how zero counts were handled. No source evidence was used to promote a validation setting into an author default.

### Frozen upstream identities

| Input | SHA-256 |
| --- | --- |
| S01 full results | `4e7025fcb2aaa63eb9fad6b0760e5051b857245cfc4fd4c8840d628e44d72a97` |
| S02 full results | `9bcd86ef86c371ba57e991a3bc8295cd92ef2fe05b9edf53327814b0c52f2cfa` |
| S03 full results | `7ef7837bf0b2b65fd011ec5b530b90e07089df637050155b81fc08d3c99992ee` |
| S04 full results | `62a1f862e17579769aca3939b69eba3ff725078d593ece6b29ccb58a29b8c59d` |
| S05 full results | `e83620da619d05687d218186cd2d2789ce26bbac6234de71990948915ea95196` |
| S06 full results | `206482bce8e8a47d5050e83c4a99370c2bedf7dd4244a066078f9ad5c3233595` |
| S07 full results | `9262b881f1dc15392d1d674e3ac15222dc6dfef69d82ca6aeb74f9fe90fd876d` |
| S08 full results | `aed1335f670ca695634aed6023247a6b75c5873a96aa0ff3be8bb5fb22cddbb6` |
| Registry v0.3.0 | `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891` |
| S08 fixture catalog | `b699f7e4e12c8691f10511957cae8785a3f6b93603968471e37e7fb023e4e89f` |
| S08 label outputs | `24fd8ed34ab49e9fbde511d78bb653445ce37872990a8fe676827ed0d800e08b` |
| S06 precision contract | `2c73d7385d7511636cb809cdb1b2b5c0239632faec2f6ff2ffb692a7b3548b4d` |
| S06 trajectory schema | `981807b512bff589a6a693c1da191efad829ccb3294fd4f9297c3ee02a7a5d57` |

Exact paths, sizes, and all 28 hashes are in `preregistration_record.json` and the final artifact manifest.

### Validated observations

S09 preserved the five S08 fixtures exactly: 43 observations in dimensions 2 and 4. Integer count fixtures stayed counts. The S04 HC11 noninteger source-oracle vectors stayed nonnegative weights; their count-scale pseudocount results are explicitly validation diagnostics, not relabeled molecular counts.

No dataset, external web source, new dependency, GPU computation, or S10 input was used.

## Preregistration chronology

1. The outcome-blind contract `E01-S09-compositional-preregistration-v1.0.0` froze inputs, formulas, grids, IDs, eligibility, tolerances, diagnostics, acceptance rules, and failure injections.
2. It was validated, committed, and pushed at `e83b4e9de3e8077cb6aa41b0975adc93e4b6d560` while canonical `/artifacts/research_steps/S09/` and S10 outputs were absent.
3. The transform library, schema, builder, and tests were implemented and pushed at `8646f4cdf12b2633e0ea1ca1b158c070d9817640` before canonical S09 outcomes were generated.
4. No zero treatment, delta, basis, tolerance, acceptance rule, or diagnostic changed after outcome inspection.

The artifact preregistration has SHA-256 `2f9ed2d8c4f326e0ec162545ed73beada55905505efb91eafb49d993aa815070`.

## Detailed methods

### 1. Complete-row policy

Inputs had to be finite and nonnegative. Negative or nonfinite fixtures failed the specification rather than producing partial outputs. For every valid input row and every complete fixture specification, the library emitted exactly one of:

- `ELIGIBLE`, with finite treated composition, coordinates, reconstructed composition, and inverse errors; or
- `INELIGIBLE`, with null numerical outputs and a machine-readable reason.

Rows were never deleted, implicitly imputed, or recoded as drift. The validation feature layout was explicitly observations-as-rows and components-as-columns with no extra scaling. This validation setting does not resolve registry sentinel `UNRESOLVED::E01-A032`. The status policy likewise does not resolve `UNRESOLVED::E01-A033`.

### 2. Zero treatments

The frozen grid was

\[
\delta\in\{10^{-6},10^{-4},10^{-2},0.1,0.5,1\}.
\]

For additive pseudocounts and supplied nonnegative row \(n\) of dimension \(D\),

\[
x_i^{(\delta)}=\frac{n_i+\delta}{\sum_j n_j+D\delta}.
\]

This maps an all-zero row to the explicitly declared uniform composition. It is count-scale dependent by design.

For the matched-delta multiplicative branch, a positive-mass row was first closed to \(x=n/N\). With \(z\) zero components,

\[
q=\frac{\delta}{N+D\delta},\qquad
x_i'=q\text{ for zeros},\qquad
x_i'=(1-zq)x_i\text{ otherwise}.
\]

This preserves every positive-part ratio and uses the same frozen count-scale delta as its additive comparator, avoiding a second outcome-tuned grid. An all-zero row has no composition to redistribute and remains explicitly ineligible.

The thirteenth branch applies no zero replacement. Positive-mass rows are closed without changing zeros. Raw and Hellinger coordinates can retain component zeros; log-ratio coordinates cannot. Zero-sum rows remain ineligible.

### 3. Coordinate families

For a strictly positive closed composition,

\[
\operatorname{clr}(x)_i=\log x_i-\frac{1}{D}\sum_j\log x_j.
\]

S09 materialized:

- full CLR as a structural-rank diagnostic;
- every dropped-component CLR, including the paper-like final-component drop;
- Helmert ILR;
- raw closed proportions;
- Hellinger coordinates \(\sqrt{x}\); and
- covariance-eigenvector principal log-ratio (PLR) coordinates.

For every dropped component \(k\), the inverse inserts

\[
z_k=-\sum_{j\ne k}z_j
\]

before exponentiation and closure. Therefore dropped CLR remains compositionally invertible even though ordinary Euclidean distances in the retained coordinates are not Aitchison-isometric.

The frozen sequential Helmert simplex basis \(V\in\mathbb R^{D\times(D-1)}\) satisfies \(V^TV=I\) and \(V^T\mathbf1=0\). ILR coordinates and inverse are

\[
y=V^T\log x,\qquad x=\mathcal C(\exp(Vy)).
\]

PLR uses \(B=VQ\), where \(Q\) contains descending covariance eigenvectors fitted separately within each explicitly named fixture or pooled-dimension scope and zero treatment. The largest-absolute loading sign was made positive, with earliest-component tie breaking. All eigenvalues, bases, sign-resolved loadings, orthogonality errors, and near-tie diagnostics are serialized. PLR is a validation-only data-adaptive control, never a paper default.

### 4. Conditioning and covariance rank

Coordinates were centered and covariance used divisor \(n-1\). Numerical rank used

\[
\epsilon_{rank}=\max(n,p)\epsilon_{64}s_{max}.
\]

Both raw condition number and effective condition number on the nonzero singular subspace were recorded. A nonstructural branch was `READY` only if observed covariance rank equaled coordinate dimension and raw condition number was at most `1e12`.

Transform validity and covariance readiness were deliberately separate:

- full CLR is always `STRUCTURALLY_SINGULAR_FULL_CLR` when covariance is estimable;
- raw closed proportions are `STRUCTURALLY_SINGULAR_RAW_CLOSURE`;
- too few eligible rows, fixture/sample rank deficiency, and ill conditioning are reported without regularization or repair.

Diagnostics were computed per fixture and after pooling fixtures by dimension. Pooling is a numerical validation scope, not a scientific estimator window.

### 5. Inverse and representation agreement

Every eligible result was inverted. Maximum absolute error, relative error, and closure error each had frozen tolerance `1e-12`.

Within each zero treatment and scope:

- full-CLR and ILR pairwise distances had to agree within `1e-12`;
- ILR and PLR distances had to agree within `1e-12` when the PLR basis was fit;
- every dropped-CLR inverse had to recover the treated composition;
- Pearson and Spearman distance agreement between each dropped CLR and ILR was reported, not forced to one; and
- raw/ILR and Hellinger/ILR distance correlations were recorded as adversarial controls.

Additive-versus-multiplicative Aitchison distance was recorded by observation and delta. Multiplicative positive-part log-ratio preservation and exact no-change behavior on strictly positive rows were separate validation checks.

### 6. Lossless schema and checksums

`transform_arrays.json` uses the S06 canonical JSON subset. All binary64 values—including inputs, compositions, coordinates, inverses, bases, eigenvalues, and errors—are canonical Python `float.hex` strings. The payload is SHA-256 protected and conforms to the S09 Draft 2020-12 schema. It includes zero, coordinate, complete-specification, and PLR-basis identities, so a coordinate array never loses the branch that produced it.

The payload SHA-256 is `55fdc55320289a0b427413509cf565d251802414a4b1214eebe3e2b4a1ec5cba`; the envelope file SHA-256 is `65cb165e0575e0c03f5aa97f854a71a125ebf946c68059fba5df37d53e7a3ffc`.

### 7. Failure injection

Eight preregistered defects were injected into validation calls or derived records:

1. negative input;
2. nonfinite input;
3. raw unresolved registry sentinel used as an executable specification;
4. log ratio applied to an unreplaced component zero;
5. nonorthonormal ILR basis;
6. one hidden row deletion;
7. corrupted inverse composition; and
8. checksum-tampered canonical JSON.

All eight were detected.

## Parameters, dependencies, and compute

| Item | Value |
| --- | --- |
| Transform contract | `E01-S09-compositional-transform-contract-v1.0.0` |
| Output schema | `E01-S09-transform-output-schema-v1.0.0` |
| Zero treatments | 13: six additive, six matched-delta multiplicative, one no-replacement |
| Dimensions | 2 and 4 |
| Numerical dtype | CPU binary64 / NumPy `float64` |
| Inverse/isometry tolerance | absolute and relative `1e-12` |
| Covariance condition threshold | `1e12` |
| Python | 3.13.14 |
| NumPy | 2.4.6 |
| SciPy | 1.18.0 |
| Matplotlib | 3.11.1 |
| PyYAML | 6.0.3 |
| jsonschema | 4.26.0 |
| pytest / Ruff | 9.1.1 / 0.16.0 |
| CPU/GPU | One process, one thread for OMP/OpenBLAS/MKL/NumExpr; GPU unused |
| New dependencies | None |

The work was compact deterministic fixture analysis; parallel execution or GPU use would not improve evidential quality.

## Commands

Principal commands, run from `/workspace/arrival-of-self-replicators`, were:

```bash
# Freeze before outcomes.
python - <<'PY'
# Parsed the preregistration, verified 28 hashes, eight S09 registry owners,
# the closed registry gate, and S09/S10 absence.
PY
git commit -m "Preregister E01 S09 compositional zero audit"
git push origin eidosoma/groups/42

# Focused validation and repository validation.
ruff format src/e01_compositional_preprocessing \
  scripts/e01/build_s09_compositional_artifacts.py \
  tests/e01/test_compositional_preprocessing.py
ruff check src scripts tests
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python -m pytest -q -p no:cacheprovider \
  tests/e01/test_compositional_preprocessing.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python -m pytest -q -p no:cacheprovider tests/e01 \
  -k 'not generated_artifacts_and_fresh_process_regeneration_when_present and not checksum_tamper_detection_and_exact_same_engine_regeneration'

# Commit implementation before canonical outcomes.
git commit -m "Implement E01 S09 compositional transform audit"
git push origin eidosoma/groups/42

# Canonical build.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python scripts/e01/build_s09_compositional_artifacts.py \
  --artifacts-dir /artifacts

# Isolated exact replay under /cache, followed by byte comparisons.
cmp /artifacts/research_steps/S09/transform_arrays.json \
  /cache/e01_s09_replay.kFZD2e/research_steps/S09/transform_arrays.json
cmp /artifacts/research_steps/S09/transform_outputs.csv \
  /cache/e01_s09_replay.kFZD2e/research_steps/S09/transform_outputs.csv
cmp /artifacts/research_steps/S09/numerical_diagnostics.csv \
  /cache/e01_s09_replay.kFZD2e/research_steps/S09/numerical_diagnostics.csv
cmp /artifacts/research_steps/S09/representation_agreement.csv \
  /cache/e01_s09_replay.kFZD2e/research_steps/S09/representation_agreement.csv
```

## Results

### Zero frequency

| Fixture | Observations | With any zero | Zero-sum | Zero cells / all cells |
| --- | ---: | ---: | ---: | ---: |
| Two attractors | 20 | 0 | 0 | 0 / 80 |
| Rare-species swing | 9 | 0 | 0 | 0 / 36 |
| Zero and extinction | 6 | 4 | 1 | 9 / 24 |
| S04 HC11 | 5 | 4 | 1 | 5 / 10 |
| S06 growth-final | 3 | 1 | 0 | 1 / 12 |
| **Total** | **43** | **9** | **2** | **15 / 162** |

This exactly reconciles the nine S08 Aitchison-ineligible observations and identifies the two stricter zero-sum cases.

### Complete row and eligibility accounting

| Result | Count |
| --- | ---: |
| Expected and emitted lossless transform rows | 4,901 / 4,901 |
| Eligible finite rows | 4,745 |
| Explicit ineligible rows | 156 |
| Fixture-level complete specifications | 247 |
| Accepted with explicit eligibility domain | 246 |
| Not evaluable in declared domain | 1 |

Ineligibility was fully explained:

| Reason | Rows |
| --- | ---: |
| Multiplicative replacement on zero-sum composition | 96 |
| Log-ratio coordinate with unreplaced component zero | 40 |
| No-replacement closure on zero-sum composition | 16 |
| Too few rows to fit fixture PLR basis | 4 |
| **Total** | **156** |

The one fixture-level complete specification classified `NOT_EVALUABLE_DECLARED_DOMAIN` was no-replacement PLR on S04 HC11: only one supplied observation was strictly positive, so a covariance eigenbasis could not be fit. The corresponding pooled dimension-2 diagnostic was also not evaluable. No substitute basis was borrowed from another fixture.

### Inverse validation

| Coordinate family | Maximum absolute inverse error | Scope-level inverse rows |
| --- | ---: | ---: |
| Full CLR | `2.220e-16` | 91 pass |
| Every dropped CLR | `8.327e-16` | 312 pass |
| Helmert ILR | `1.055e-15` | 91 pass |
| Raw proportions | `1.110e-16` | 91 pass |
| Hellinger | `1.110e-16` | 91 pass |
| Principal log ratio | `1.998e-15` | 89 pass; 2 not evaluable |

All applicable errors were at least about 500 times smaller than the `1e-12` gate.

### Representation agreement

- Full CLR versus Helmert ILR and ILR versus PLR produced 182 preregistered isometry rows.
- Four had fewer than two common eligible observations and were explicitly not evaluable.
- All **178/178 evaluable** rows passed.
- The maximum pairwise-distance discrepancy was `1.066e-14`, about 94 times below the gate.
- All 89 fitted PLR bases satisfied orthonormality: maximum Gram error `1.110e-15` and maximum all-ones orthogonality error `2.220e-16`; the other two scope/treatment combinations were explicitly ineligible because fewer than two rows were available to fit a basis.
- There were 89 fit and two insufficient PLR bases; three near eigenvalue ties occurred in small no-replacement fixtures and are explicitly recorded.

Dropped CLR was reversible but not isometric. Across all evaluable scope/treatment/drop rows, Pearson distance correlation with ILR ranged `0.561–1.000` and Spearman ranged `0.500–1.000`. In the pooled dimension-4 evidence, the ranges narrowed to Pearson `0.936–0.998` and Spearman `0.937–0.998`, still demonstrating dependence on the removed component. Therefore paper-like drop-final cannot be treated as coordinate-choice invariant.

Raw versus ILR distance Pearson correlations ranged from `-0.203` to `1.000` across the deliberately small/heterogeneous fixtures; Hellinger versus ILR ranged `0.533–1.000`. These controls reinforce that coordinate-family choice can materially change ordinary Euclidean geometry.

### Additive versus multiplicative replacement

Multiplicative replacement preserved positive-part log ratios to maximum error `1.665e-16` and changed strictly positive rows by exactly zero. Its two all-zero observations across six deltas produced 12 explicit ineligible treatment comparisons.

| Delta | Eligible comparisons | Median Aitchison distance | Maximum Aitchison distance |
| ---: | ---: | ---: | ---: |
| `1e-6` | 41 | `1.433e-7` | `8.183e-7` |
| `1e-4` | 41 | `1.433e-5` | `8.183e-5` |
| `1e-2` | 41 | `0.001431` | `0.008141` |
| `0.1` | 41 | `0.014162` | `0.077804` |
| `0.5` | 41 | `0.067746` | `0.328072` |
| `1` | 41 | `0.124122` | `0.555516` |

The monotone scale of disagreement is descriptive fixture evidence, not a rule for selecting the smallest delta. Every delta remains a separate immutable specification.

### Covariance rank and conditioning

Across 767 fixture/pool specification diagnostics:

| Covariance readiness | Rows |
| --- | ---: |
| Ready | 382 |
| Sample/structural rank deficient | 195 |
| Structurally singular raw closure | 91 |
| Structurally singular full CLR | 89 |
| Insufficient eligible rows | 10 |

The pooled dimension-4 result was strongest: all 52 dropped-CLR, 13 ILR, 13 PLR, and 13 Hellinger specifications were full-rank and `READY`; their ready condition numbers ranged `7.19–149.15`. All 13 full-CLR and all 13 raw specifications retained rank 3 in four coordinates and were explicitly structural-singularity controls.

For pooled dimension 2, all eligible one-dimensional dropped-CLR/ILR/PLR configurations were condition number 1. Hellinger was full-rank with condition numbers up to `729.34`. The no-replacement log-ratio/PLR cases had only one strictly positive observation and were reported insufficient rather than regularized.

Fixture-level rank deficiencies are expected for duplicate, very small, or deliberately low-dimensional fixtures. They constrain claims of estimator readiness but do not invalidate a finite reversible transform.

### Valid-specification interpretation

`valid_transform_specification_registry_v1.0.0.yaml` is a validation registry, not a revision of the author-facing registry. It assigns immutable zero/coordinate/basis identities and one of:

- `ACCEPTED_WITH_EXPLICIT_ELIGIBILITY_DOMAIN`; or
- `NOT_EVALUABLE_DECLARED_DOMAIN`.

It selects no paper default. Downstream code must still name a complete S09 specification ID and cannot read registry branch/sentinel text as an executable value.

## Validation

### Canonical validation gates

`validation_summary.json` contains 45 passing checks:

- 28 frozen input identities;
- preregistration ancestry and one-thread environment;
- byte-identical registry and eight-owner preservation;
- exact complete-row coverage;
- all 13 zero treatments and every dropped component;
- eligible finiteness and reasoned ineligibility;
- inverse tolerances and basis validation;
- required isometries;
- schema, checksum, canonical byte round trip;
- all eight failure injections;
- S08 zero-count reconciliation;
- complete diagnostics and nonempty figures; and
- S10 absence.

### Schema and replay

- Draft 2020-12 meta-schema: pass.
- Instance conformance: pass.
- Canonical checksum verification: pass.
- Canonical deserialize/reserialize: byte-exact.
- Isolated replay: `transform_arrays.json`, `transform_outputs.csv`, `numerical_diagnostics.csv`, and `representation_agreement.csv` were byte-identical.
- Canonical/replay transform-array SHA-256: `65cb165e0575e0c03f5aa97f854a71a125ebf946c68059fba5df37d53e7a3ffc`.

### Code tests and static validation

- S09 focused tests: **8 passed**.
- Non-identity-pinned E01 repository tests: **82 passed, 2 deselected**.
- Ruff lint across `src`, `scripts`, and `tests`: pass.
- Ruff format for every S09 Python file: pass.
- JSON schema parse/meta-schema: pass.
- Git whitespace check: pass.

An initial broad repository run selected the deliberately identity-pinned S06 fresh-process test. It produced 82 passes, one expected failure, and one deselection because the old S06 example correctly rejects the later S09 Git commit. The S06 artifact was not rewritten. The final suite excluded both identity-pinned replay tests and passed 82/82. This is recorded in `supplemental_validation.json`.

### Visual validation

All three figures were opened at high detail. Titles, axes, category labels, logarithmic scale, and plotted patterns were legible and consistent with their source tables:

- `zero_frequency.png`;
- `covariance_readiness.png`; and
- `replacement_disagreement.png`.

## Artifacts written

### Shared reusable preprocessing bundle

| Path | Purpose |
| --- | --- |
| `/artifacts/E01_forensic_replication_bundle/preprocessing/compositional_transform_contract_v1.0.0.yaml` | Frozen formulas, domains, diagnostics, acceptance rules, registry boundary, and eight-owner snapshot |
| `/artifacts/E01_forensic_replication_bundle/preprocessing/transform_specifications_v1.0.0.yaml` | Zero, coordinate, complete-specification, and lossless PLR-basis catalog |
| `/artifacts/E01_forensic_replication_bundle/preprocessing/valid_transform_specification_registry_v1.0.0.yaml` | Fixture-level numerical acceptance and eligibility registry without paper-default selection |
| `/artifacts/E01_forensic_replication_bundle/preprocessing/transform_output_schema_v1.0.0.json` | Draft 2020-12 lossless transform-envelope schema |

### S09 handoff directory

| Artifact | Purpose |
| --- | --- |
| `research_step_full_results.md` | This canonical detailed handoff |
| `preregistration.yaml`, `preregistration_record.json` | Outcome-blind contract, commit chronology, and 28 frozen-input checks |
| `transform_arrays.json`, `transform_outputs.csv` | Lossless checksum-protected and convenient row-level transform outputs |
| `zero_frequency_by_observation.csv`, `zero_frequency_by_fixture.csv` | Exact zero-cell, zero-row, and zero-sum audit |
| `eligibility_summary.csv` | Complete retained/ineligible counts and reasons per scope/specification |
| `numerical_diagnostics.csv` | Finiteness, dimensions, covariance rank, condition, and readiness |
| `inverse_transform_validation.csv` | Absolute/relative/closure inverse errors |
| `representation_agreement.csv` | Isometries and distance correlations for every relevant representation |
| `replacement_agreement.csv` | Additive/multiplicative distances and multiplicative ratio-preservation checks |
| `failure_injection.json` | Eight detected failure modes |
| `schema_validation.json` | Schema/checksum/round-trip identities |
| `supplemental_validation.json` | Exact replay, repository tests, expected S06 identity rejection, and visual audit |
| `registry_preservation.json` | Byte-identical v0.3.0 and exact S09-owner snapshot |
| `validation_summary.json` | Canonical 45-check rollup and anchor metrics |
| `zero_frequency.png`, `covariance_readiness.png`, `replacement_disagreement.png` | Compact diagnostic figures |
| `artifact_manifest.json` | Final paths, sizes, SHA-256 hashes, Git identities, and S10-absence evidence |

`RESEARCH_PLAN.md` did not request a distinct S09 `status.json`, so none was created. `validation_summary.json` includes the workflow fields without being relabeled as a status file.

### Git-backed implementation

Repository code remains in Git and was not copied into artifacts:

- `configs/e01/s09_compositional_preregistration.yaml`;
- `configs/e01/s09_transform_output_schema.json`;
- `src/e01_compositional_preprocessing/`;
- `scripts/e01/build_s09_compositional_artifacts.py`; and
- `tests/e01/test_compositional_preprocessing.py`.

## Provenance

- Research date: 2026-08-01 UTC.
- Repository: `/workspace/arrival-of-self-replicators`.
- Branch: `eidosoma/groups/42`.
- Pre-S09 repository identity: `7a8df1dfd596057a8c56a5aa5fdea4d0dbdaebd9`.
- Preregistration commit: `e83b4e9de3e8077cb6aa41b0975adc93e4b6d560`.
- Implementation commit: `8646f4cdf12b2633e0ea1ca1b158c070d9817640`.
- Both commits were pushed to `origin/eidosoma/groups/42` before handoff.
- Registry v0.3.0 SHA-256 remained `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891`.
- Canonical transform envelope SHA-256: `65cb165e0575e0c03f5aa97f854a71a125ebf946c68059fba5df37d53e7a3ffc`.
- Exact code/input/output paths, sizes, hashes, and current Git identity are in `artifact_manifest.json`, which excludes its own hash.

## Caveats, blockers, failed assumptions, and limitations

### Author and paper uncertainty

- The author implementation remains unavailable. Nothing in S09 identifies its zero handling, pseudocount, multiplicative replacement, feature layout, invalid-row policy, or coordinate implementation.
- The paper's final-component removal is represented, but is not promoted to the unique valid branch.
- Natural-log CLR remains a provisional reconstruction supported by the plan and paper's nats language, not direct code identity.
- The original registry remains non-executable with 64 unresolved/conflicting/evidence-deferred parameters and 21 unexpanded branch sets. All eight S09-owner values/statuses are byte-identical.

### Numerical boundaries

- Full CLR is finite and reversible after a valid zero treatment but linearly dependent, so its covariance is structurally singular.
- Raw proportions are also closure constrained and covariance-singular.
- Dropped CLR removes the linear dependence and is invertible, but ordinary Euclidean distances depend on the dropped component. It is not an orthonormal simplex coordinate system.
- ILR and PLR are Aitchison-isometric. PLR is data-adaptive and cannot be silently reused across fit scopes.
- Two small no-replacement PLR scopes could not fit a covariance eigenbasis; one affected the fixture-level valid registry. They remain explicit, not repaired.
- Multiplicative replacement preserves positive ratios but cannot define an all-zero composition. Additive pseudocounts define all-zero rows only through the declared uniformizing rule.

### Evidence limitations

- The five fixtures test zero semantics, geometry, rank handling, and serialization. They are not the paper's 100-run baseline and do not estimate future zero prevalence or estimator stability.
- Pooled-by-dimension conditioning is a validation diagnostic, not an authorized time window, trajectory alignment, estimator fit, or MIB input.
- S09 does not compute self-replicator labels after replacement. The frozen S08 outputs remain unchanged; any future zero-aware label configuration requires a new immutable label specification.
- S09 does not compute \(\Phi^r\), validate a PhiID atom, select redundancy/MIB methods, or adjudicate any paper claim.
- Simulation and transform evidence remains computational proxy evidence, not experimental validation of prebiotic chemistry or biological agency.

### Preserved validation attempt

The broad repository test's expected S06 identity rejection was not hidden. It demonstrates that S06 exact regeneration is source/commit scoped. S09's own exact replay passed in a separate artifact root. No scientific tolerance or outcome rule changed after this test.

## Recommended next action

Stop after S09 and return control to the Chief Scientist. S10 is eligible but was not started. If separately authorized, S10 should consume immutable S09 complete-specification IDs, keep additive/multiplicative and coordinate sensitivities distinct, use ILR or another full-rank coordinate when covariance assumptions require it, and preserve the no-paper-default boundary. It must not optimize delta, dropped component, basis, or exclusions against PhiID benchmark outcomes without a new preregistration.
