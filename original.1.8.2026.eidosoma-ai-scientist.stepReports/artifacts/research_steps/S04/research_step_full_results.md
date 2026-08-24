# S04 full results — Construct a historical-reference GARD engine

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S04** — Construct a historical-reference GARD engine |
| Completion status | **Complete** on 2026-08-01. Only S04 was executed; S05 was not begun. |
| Artifacts written | Git-backed Python engine at commit `1f8e3b0d6fd68b9d62b566b6f237febe64e2afa9`; versioned historical behavior contract and fixture specification; engine pointer; compatibility notes and matrix; source-line traceability table; 15 verified small-case results; registry-preservation audit; validation summary; artifact manifest; and this canonical report. |
| Validation result | **PASS** — 15/15 hand-oracle fixtures, exact hand propensities, strict weighted-event boundaries, all tested event mass changes of exactly ±1, split-size/extinction behavior, even/odd fission, first-child continuation, two-generation lineage, both historical non-drift techniques, nine pinned file hashes, 14 valid source-line mappings, all 19 S04 registry mappings, 11 no-silent-default API checks, 11/11 focused tests, 24/24 repository tests, Ruff, scoped formatting, artifact generation, Git push/remote equality, and S05-absence checks passed. |
| Outcome classification | **Supportive** for S04's bounded hypothesis: the pinned public historical equations and control flow can be translated into a deterministic, source-traceable compatibility engine. Historical-versus-paper differences are important constraining findings, but no paper result was adjudicated. |
| Caveats or blockers | Exact legacy MATLAB `rand`/`randn` stream equality remains unresolved; explicit draw tapes bypass rather than solve it. No MATLAB or GNU Octave executable was present, so original `.m` trajectories were not executed. The pinned historical repository has no detected license. Historical v10 is not the unavailable paper-author implementation. Its fixed-size fission, with-replacement initializer, single-event loop, rate-sum “time,” and absence of `max_steps` differ from or underdetermine paper prose. Registry v0.3.0 remains closed with 64 unresolved/conflict/evidence-deferred items and 21 unexpanded branch sets. |
| Lay summary | A small, testable Python reference now captures what the pinned public 2014 GARD code actually does when given explicit random draws. It deliberately does not blur that behavior with the paper or newer GARD code. The largest discovered difference is fission: the historical code makes equal-size daughters by sampling without replacement, whereas the paper describes binomial splitting. |
| Recommended next action | **Stop after S04 and return control.** If separately authorized, execute S05 as an independently structured engine, keep every historical/paper/modern branch explicit, and compare model-level distributions without copying S04 control flow. |

## Frozen question

Can the public historical GARD equations and control semantics be translated without changing their stochastic behavior?

The operational criterion was that a source-traceable engine reproduce verified small historical cases for catalytic-matrix construction, event weights and selection, stochastic growth, fission, daughter continuation, and non-drift analysis. “Stochastic behavior” is bounded here to the transition law and control flow conditional on explicit random draws. It does not mean that NumPy recreates an unidentified legacy MATLAB random stream.

## Decision

Yes, within that explicit boundary. The engine reproduces all 15 versioned hand/source-derived cases and every preregistered S04 invariant. The supportive result rests on:

- a pinned historical commit and tree;
- hash-verified source files and exact line ranges;
- explicit random-draw inputs for trajectory-equality fixtures;
- independently written expected values rather than output-derived snapshots;
- mandatory kinetic, reservoir, boundary, and random-source API inputs;
- preserved registry sentinels and branch sets; and
- clear separation from both modern GARD and the unavailable author implementation.

The result also constrains later reconstruction. Public historical v10 is not a drop-in implementation of all paper prose: its fission law, initializer, time accumulator, and growth termination semantics materially differ.

## Lay summary

GARD describes a molecular assembly that gains and loses molecules, grows, divides, and sometimes revisits similar compositions. The public historical code is compact but contains consequential conventions that are easy to miss. This step turned those conventions into a transparent Python reference with small examples whose answers can be checked by hand.

For example, the historical divider does not flip an independent fair coin for each molecule. It draws exactly half of the parent molecules without replacement into the first daughter, leaves the complement as the second daughter, and discards one molecule when the parent size is odd. The single-lineage program then follows the first daughter without another random choice. Those details now have explicit tests and provenance, while the paper's different wording remains a separate unresolved reconstruction.

## Inputs

### Governing and uploaded inputs

| Input | Role | Identity or validation |
| --- | --- | --- |
| `/workspace/AGENTS.md` | Execution, artifact, validation, and Git contract | SHA-256 `85041503713d0dd36796acac13e2f8c1d840bbce521e3301da590e422de1195c` |
| `/workspace/FULL_PLAN.md` | Frozen model branches and program guardrails | SHA-256 `6e59a75d2bb23ace8110ebf3da07ddff2f3dc4ae3377cd8d14be8e8bfd22d7ee` |
| `/workspace/RESEARCH_PLAN.md` | S04 frozen question, outputs, validation, and handoff | Refreshed before work and updated after S04 |
| `/workspace/input-attachments/MANIFEST.json` | Attachment route and original paper metadata | SHA-256 `d0f71c606281cf289a4b9e0852e08c1a6b889c9021d37d5d1c32c64b62f1183e` |
| Attachment `_metadata/ATTACHMENT.md` | Required sidecar | SHA-256 `983c410106015858e6a5a2234b1128af3f29d772059775aa8c33785abc0d885c` |
| Attachment `pdf-markdown.md` | Complete accessible paper extraction | SHA-256 `23ca5473759e78be12699655fbdbc143cdd3fd383e3d28485dbb3c042bd1c59a` |
| Official paper PDF | Original equation-bearing arXiv v1 paper | arXiv `2607.28250v1`; 1,117,911 bytes; SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4` |

The attachment manifest was read before the required sidecar and paper-derived content. The official PDF in the pinned S03 cache was independently hash-checked and its GARD Methods pages were re-extracted for S04.

### Prior research-step inputs

- S01 full-results report, 59-row claim ledger, and 12-row source reconciliation.
- S02 full-results report, 105-row ambiguity ledger, 12-row discrepancy taxonomy, claim crosswalk, and 120-parameter registry lineage.
- S03 full-results report, commit verification, source registry audit, clean-environment smoke, dependency/runtime provenance, and artifact manifest.
- Canonical source manifest at `/artifacts/E01_forensic_replication_bundle/provenance/source_manifest.yaml`, SHA-256 `50a569e30826fe893258f5c0935469576008f43134eee94e8bf2654d4ef23ed5`.
- Canonical registry v0.3.0 at `/artifacts/E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml`, SHA-256 `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891`.

### Pinned historical source

| Field | Value |
| --- | --- |
| Repository | `https://github.com/ModelingOriginsofLife/GARD.git` |
| Commit | `86dff6320d5ae91b4e831471079ff46749b14df9` |
| Tree | `a602fc99b494982c04c60405bc6422af9db5a77a` |
| Local reference path | `/cache/e01_s03/sources/gard-historical` |
| Git status | Detached, clean |
| License evidence | `NO-LICENSE-FILE-DETECTED`; reference in place, no source redistribution |

Nine files were required and rehashed: `README.txt`, `tgs_parameters_v10.m`, `tgs_newbeta_v10.m`, `tgs_grow_v10.m`, `tgs_rndpdf.m`, `tgs_split_v10.m`, `tgs_agard_v10.m`, `tgs_H.m`, and `tgs_nondrift.m`.

## Detailed methods

### 1. Historical source contract

The pinned MATLAB files were read with numbered lines. Fourteen behavior-to-line mappings were registered in `s04_historical_contract.yaml`, covering:

1. public v10 parameters and seed slots;
2. catalytic-matrix transformation;
3. propensity equations;
4. weighted event selection;
5. the growth loop and rate-sum accumulator;
6. loss nonnegativity;
7. fixed-size fission;
8. environmental fallback;
9. initializer/global RNG order;
10. daughter selection;
11. generation-endpoint traces;
12. historical H similarity;
13. non-drift technique 1; and
14. non-drift technique 2.

Every mapping includes the source file, inclusive line range, full-file SHA-256, port symbol, and a bounded interpretation. Validation checks the commit, tree, clean checkout, file hash, file length, and line-range bounds.

### 2. Catalytic matrix

The historical transform was ported as

\[
\beta=\exp(\mu+\sigma Z),\qquad Z_{ij}\sim\mathcal N(0,1).
\]

The supplied draw matrix retains its orientation and diagonal. No symmetrization, transpose, or diagonal removal is applied. Exact fixtures call `catalytic_matrix_from_standard_normals`; a separately named NumPy convenience function requires an explicit `numpy.random.Generator` and states that it is not a legacy-MATLAB stream emulator.

### 3. Propensities and event selection

For an integer state \(n\) with mass \(N>0\), the port computes

\[
b=1+\frac{\beta n}{N},
\]

\[
a^+=(k_f\rho N)\odot b,
\qquad
a^-=(k_b n)\odot b.
\]

The event vector is concatenated as all joins followed by all leaves. The historical sampler draws a nonzero uniform \(u\), multiplies it by the total weight, and uses a strict cumulative-boundary comparison. Event records retain the zero-based event index, one-based molecular species, signed historical event convention, pre/post integer states, all weights, total weight, and exact mass change.

All model-defining inputs are mandatory. There is no implicit `k_f`, `k_b`, `rho`, `beta`, `n_max`, or RNG source. A named explicit-draw tape is the trajectory fixture path.

### 4. Historical growth and “time”

The growth function applies one event per loop until mass reaches `n_max` or the assembly becomes empty. From a state below `n_max`, a join reaches the integer boundary exactly; no batch overshoot occurs. The source does not implement the paper's `max_steps=1000` condition.

The source variable named `dt` adds the current **total rate** after each event; it does not sample an exponential waiting time. The historical orchestrator later stores its reciprocal. The port names these fields `legacy_dt_accumulator` and `legacy_inverse_rate_sum` so they cannot be silently mistaken for Gillespie time.

An optional `event_guard` is a validation safety guard. It is a mandatory API argument and raises without returning a historical terminal state if reached. It is never interpreted as the paper's unresolved maximum-step rule.

### 5. Fission and daughter continuation

The source fission algorithm was ported literally at the control-law level:

- draw `floor(N/2)` molecules sequentially without replacement into child A;
- leave the remaining molecules as child B;
- for odd parent mass, draw and discard one additional molecule so both returned children have size `floor(N/2)`; and
- in the single-lineage orchestrator, request only the first output and therefore follow child A with no extra daughter-selection draw.

This is a fixed-sample multivariate-hypergeometric-style allocation, not independent \(\operatorname{Binomial}(n_i,0.5)\) counts. The artifact contract preserves the mismatch with the paper/plan branch rather than changing either definition.

### 6. Initial state and environmental behavior

The historical orchestrator supplies uniform `rho_i=1/N_g` if the supplied reservoir vector has the wrong length and then leaves it constant. Its initializer samples `n_min` molecule types **with replacement** into counts. It also uses order-dependent legacy global `rand`/`randn` state and seed calls; whether initialization is governed by a particular seed depends on how the parameter and beta functions were invoked.

The port provides the with-replacement count-construction helper but requires explicit draws. It does not claim to reconstruct hidden MATLAB global state. The paper's without-replacement statement and author reservoir choice remain separate.

### 7. Historical similarity and non-drift analysis

`historical_h` implements the source's column-wise cosine similarity, `1e-7` norm floor, row-vector reshaping quirk, and clipping to `[0,1]`.

Non-drift technique 1:

- truncates at the first zero-sum generation;
- L2-normalizes active columns;
- computes adjacent-generation cosine similarities;
- gives each interior generation the average of its incoming and outgoing similarity, with endpoint repetition; and
- applies a strict `>` threshold.

The threshold is required because the MATLAB default branch misspells its variable and is not reliable.

Technique 2 preserves the source's consecutive-high-similarity streak and one-index-earlier marking. If a qualifying streak begins at the first element, the source would attempt a MATLAB index zero. The port raises `HistoricalSourceDomainError`; it does not silently repair or reinterpret the edge case.

### 8. Registry preservation

All 19 v0.3.0 parameters owned by S04 were mapped to source findings. The registry itself was not modified because public historical evidence does not establish unavailable author choices.

| Registry action | Count | Meaning |
| --- | ---: | --- |
| `PRESERVED_AUTHOR_SENTINEL` | 12 | Historical value/behavior is recorded separately; author value stays unresolved. |
| `PRESERVED_BRANCH_SET` | 3 | Historical behavior implements one named branch; other branches remain. |
| `PRESERVED` | 2 | Existing fixed/provisional value remains unchanged. |
| `PRESERVED_PAPER_PLAN_VALUE_AND_RECORDED_CONFLICT` | 1 | Paper/plan binomial fission remains while the historical mismatch is recorded. |
| `PRESERVED_PROVISIONAL_PAPER_VALUE_AND_RECORDED_CONFLICT` | 1 | Paper-like initializer remains while historical with-replacement behavior is recorded. |

Registry v0.3.0 remains at 120 parameters, 64 unresolved/conflict/evidence-deferred items, 21 unexpanded branch sets, 85 total blockers, `executable: false`, and `noSilentDefaults: true`.

### 9. Fixture oracle and artifact generation

Expected fixture values were written before execution in `s04_small_cases.yaml`. They are hand calculations or direct source-control consequences, not outputs captured from the port. Explicit uniform tapes remove random-stream assumptions. A deterministic builder executes the cases, performs invariant/source/registry/API audits, and writes compact evidence under `/artifacts`.

Repository code remains in Git, as required. The artifact software directory contains only a Git pointer, compact contract, reports, tables, and fixtures. No historical repository source was copied.

## Parameters and dependencies

| Item | Value |
| --- | --- |
| Experiment / step | E01 / S04 |
| Engine version | `1.0.0` |
| Contract version | `E01-historical-reference-v1.0.0` |
| Fixture version | `E01-S04-fixtures-v1.0.0` |
| Python | 3.13.14 |
| NumPy | 2.4.6 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| Ruff | 0.16.0 |
| CPU workers | 1; deterministic serial fixtures and metadata validation |
| GPU | Not used |
| MATLAB / Octave | Not present |
| New installed dependencies | None |

The engine uses only the pinned/preinstalled Python stack. No package, system dependency, sudo command, network retrieval, nested container, or GPU job was added in S04.

## Commands

Principal implementation and validation commands, run from `/workspace/arrival-of-self-replicators`, were:

```bash
sha256sum tgs_parameters_v10.m tgs_newbeta_v10.m tgs_grow_v10.m \
  tgs_rndpdf.m tgs_split_v10.m tgs_agard_v10.m tgs_H.m \
  tgs_nondrift.m README.txt

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
  -q -p no:cacheprovider tests/e01/test_historical_reference.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
  -q -p no:cacheprovider

ruff check scripts/e01 tests/e01 src/e01_gard_historical
ruff format --check src/e01_gard_historical \
  scripts/e01/build_historical_reference_artifacts.py \
  tests/e01/test_historical_reference.py
git diff --check

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
  scripts/e01/build_historical_reference_artifacts.py \
  --artifacts-dir /artifacts

git commit -m "Add E01 S04 historical GARD reference engine"
git push origin eidosoma/groups/42
git ls-remote origin refs/heads/eidosoma/groups/42
```

Source inspection used `nl -ba`, `diff -u`, `sha256sum`, `git rev-parse`, `git status --porcelain`, and programmatic YAML loading. The official PDF was rechecked with `sha256sum`, `pdfinfo`, and `pdftotext -layout`.

## Results

### Anchor result 1: exact hand propensities

For

\[
n=(2,1),\quad
\beta=\begin{pmatrix}1&2\\0.5&0\end{pmatrix},\quad
\rho=(0.25,0.75),\quad k_f=0.1,\quad k_b=0.2,
\]

the hand and engine results were:

| Quantity | Value |
| --- | --- |
| Catalytic boost | `(7/3, 4/3)` |
| Join weights | `(0.175, 0.300)` |
| Leave weights | `(14/15, 4/15)` |
| Total weight | `1.675` |

Uniform `u=0.2` selected species-2 join and changed `(2,1)` to `(2,2)`. Uniform `u=0.7` selected species-1 leave and changed `(2,1)` to `(1,1)`. The mass changes were exactly `+1` and `-1`.

### Anchor result 2: growth loop and rate accumulator

With zero catalysis, `n=(1,1)`, uniform `rho=(0.5,0.5)`, `k_f=1`, `k_b=0`, `n_max=4`, and draws `(0.25,0.75)`:

- event 1 joined species 1, giving `(2,1)` and total rate 2;
- event 2 joined species 2, giving `(2,2)` and total rate 3;
- growth stopped exactly at mass 4;
- legacy `dt` was `2+3=5`; and
- historical reciprocal was `0.2`.

An independent single-species loss fixture reached exact extinction with mass change `-1`.

### Anchor result 3: historical fission is not binomial fission

For parent `(2,1,1)` and draws `(0.1,0.9)`, historical fission produced:

- child A `(1,0,1)`;
- child B `(1,1,0)`;
- no discarded molecule; and
- followed daughter `(1,0,1)` with no additional draw.

For odd parent `(2,1)` and draws `(0.9,0.2)`, it produced:

- child A `(0,1)`;
- child B `(1,0)`; and
- discarded molecule `(1,0)`.

Both returned children always have `floor(N/2)` molecules. Conservation holds only after explicitly including the odd-parent discard. This source behavior is incompatible with claiming independent binomial daughter counts as the historical implementation.

### Anchor result 4: non-drift alignment

For a four-generation active trace with adjacent similarities `(1.0,0.8,0.6)` and a zero fifth column:

- historical angles were `(1.0,1.0,0.8,0.6,0.0)`;
- local averaged scores were `(1.0,0.9,0.7,0.6,0.0)`; and
- at strict threshold `0.9`, the mask was `(true,false,false,false,false)`.

Technique 2 reproduced its one-index-earlier streak marking on a safe case and raised the declared source-domain error for a qualifying streak beginning at the first element.

### Verified small-case inventory

| Case | Scope | Result |
| --- | --- | --- |
| HC01 | Explicit-draw catalytic matrix; diagonal retained | PASS |
| HC02 | Hand propensities | PASS |
| HC03 | Join event and `+1` mass | PASS |
| HC04 | Leave event and `-1` mass | PASS |
| HC05 | Two-event growth and rate-sum accumulator | PASS |
| HC06 | Loss extinction | PASS |
| HC07 | Even fixed-size fission | PASS |
| HC08 | Odd fission with one discard | PASS |
| HC09 | One generation follows child A without extra draw | PASS |
| HC10 | Historical H similarity | PASS |
| HC11 | Non-drift technique 1, strict threshold, zero tail | PASS |
| HC12 | Non-drift technique 2 index shift | PASS |
| HC13 | Technique-2 index-zero source edge failure | PASS |
| HC14 | Historical with-replacement initializer | PASS |
| HC15 | Two-generation single-lineage trace | PASS |

### Historical, paper, modern, and port boundaries

| Subject | Pinned historical source | Paper/plan | S04 port |
| --- | --- | --- | --- |
| Matrix | `exp(mu+sigma*z)`, source orientation, diagonal retained | Lognormal parameters stated; exact author indexing unstated | Exact transform from supplied draws; explicit NumPy non-MATLAB convenience path |
| Update | One weighted join/leave event | Vector Poisson wording and `max_steps=1000` | Historical loop only; other branches absent by design |
| Time | Sum total rates; orchestrator stores reciprocal | “Molecular steps”; exposure unstated | Fields explicitly named legacy rate sum/reciprocal |
| Growth limit | `n_max` or extinction; no max-step check | `n_max` or `max_steps` | Historical behavior only; validation guard raises |
| Fission | Fixed-size without replacement; odd discard | Binomial `p=0.5` | Historical behavior only; conflict preserved |
| Daughter | First output child A | One daughter, rule unstated | First child, no extra draw |
| Initialization | With replacement; hidden global RNG order | Without replacement | Helper from explicit draws; paper branch unchanged |
| Non-drift | Adjacent H rules; optional source quirks | Historical `H>0.9` mentioned | Both source techniques plus explicit failure boundary |
| Modern GARD | Historical grow SHA differs from modern | Modern is reference only | No modern code substituted; byte-identical non-drift fact retained only for that file |
| Identity | Public 2014 code | Author code unavailable | Never labeled author implementation |

No `EXACT`, `DIRECTIONAL`, `NONREPLICATION`, or `UNDERDETERMINED` verdict was assigned to any paper claim in S04. This step produced a reference engine and source constraints only.

## Validation

### Automated and fixture validation

| Layer | Result |
| --- | --- |
| Focused S04 tests | 11 passed |
| Full repository tests | 24 passed |
| Hand/source-derived fixtures | 15/15 passed |
| Event mass invariant | Every tested event exactly `+1` or `-1` |
| Fission conservation | PASS including explicit odd-parent discard |
| Fixed child-size invariant | PASS |
| Daughter extra-draw audit | PASS; none consumed |
| Pinned source files | 9/9 hashes matched |
| Source-line traceability | 14/14 ranges valid |
| Registry mapping | 19/19 exact; v0.3.0 hash unchanged |
| API no-silent-default checks | 11/11 passed |
| Source manifest historical/modern boundary | PASS |
| Ruff lint | PASS |
| Scoped S04 Ruff format | PASS |
| Git whitespace check | PASS |
| Git commit/push/remote equality | PASS |
| S05 absence | PASS |

The canonical machine-readable result is `/artifacts/research_steps/S04/validation_summary.json`. It includes the status-like fields requested by the workflow (`researchStepId`, `stepNumber`, `success`, `status`, `artifactsWritten`, `validationResult`, `caveatsOrBlockers`, and `recommendedNextAction`) even though `RESEARCH_PLAN.md` does not require a separate `status.json`.

### Independent completeness checks

- Registry and source manifest were loaded from disk and checked against their pre-S04 SHA-256 identities.
- The 19 S04 owner parameters and 19 contract mappings form identical sets.
- Every author sentinel action still points to `UNRESOLVED::...`; every branch action still points to `BRANCH_SET::...`.
- The registry execution gate remains closed with no-silent-default enforcement.
- Required model inputs have no Python defaults; explicit draw sources are mandatory.
- The historical checkout is clean at the expected commit and tree.
- Historical and modern growth files remain distinct; non-drift byte identity is not generalized.
- The engine artifact points to the pushed Git commit and reports a clean worktree at final generation.
- No `/artifacts/research_steps/S05` or independent-engine artifact was created.

### Validation boundary

No MATLAB or GNU Octave executable was available. Therefore validation does not claim byte-for-byte or seed-for-seed equality with a legacy MATLAB runtime. Exact-draw fixtures validate source equations and control flow without an RNG-stream assumption. Distributional goodness-of-fit and cross-engine statistical validation remain assigned to S07 after S05/S06.

## Failed attempts, refinements, and recoveries

- A broad repository-wide `ruff format --check scripts/e01 tests/e01 src/e01_gard_historical` returned nonzero because five pre-existing S03 files would be reformatted by the current formatter. Those unrelated files were not changed. Ruff lint passed repository-wide, and the format check passed for every S04-created Python file.
- During source review, row-vector behavior in `tgs_H.m` was explicitly preserved after noting that the source transposes one-row inputs only after its normalization expression. The final fixture suite and tests exercise the ordinary column-composition behavior; the implementation retains the row-source quirk rather than normalizing it away.
- No attempt was made to install a large MATLAB-compatible runtime solely to create a superficial equality claim. The absence is recorded, and deterministic source-conditioned fixtures are the accepted S04 evidence boundary.

No failed scientific fixture or unresolved implementation test was hidden. Final fixture errors and validation errors are both empty arrays in the machine-readable outputs.

## Artifacts written

### Shared historical-reference artifact directory

| Path | Purpose |
| --- | --- |
| `/artifacts/E01_forensic_replication_bundle/software/historical_reference/engine_pointer.json` | Git commit, branch, engine/config paths, source identity, license/author boundaries |
| `/artifacts/E01_forensic_replication_bundle/software/historical_reference/historical_behavior_contract.yaml` | Versioned source behavior and registry mapping contract |
| `/artifacts/E01_forensic_replication_bundle/software/historical_reference/compatibility_notes.md` | Human-readable historical/paper/modern compatibility handoff with top summary |
| `/artifacts/E01_forensic_replication_bundle/software/historical_reference/compatibility_matrix.csv` | Machine-readable boundary matrix |
| `/artifacts/E01_forensic_replication_bundle/software/historical_reference/source_traceability.csv` | Fourteen behavior-to-source-line mappings and hashes |
| `/artifacts/E01_forensic_replication_bundle/software/historical_reference/verified_small_cases.json` | Fifteen expected/actual cases and invariant results |

### S04 handoff directory

| Path | Purpose |
| --- | --- |
| `/artifacts/research_steps/S04/research_step_full_results.md` | This canonical full-results handoff |
| `/artifacts/research_steps/S04/validation_summary.json` | Machine-readable validation and workflow fields |
| `/artifacts/research_steps/S04/registry_preservation.json` | Exact v0.3.0 S04 mapping/sentinel audit |
| `/artifacts/research_steps/S04/artifact_manifest.json` | Input, code, source, and output sizes and SHA-256 hashes |

`RESEARCH_PLAN.md` does not request a distinct S04 `status.json`, so none was written.

### Repository-backed implementation

| Repository path | Role |
| --- | --- |
| `src/e01_gard_historical/engine.py` | Catalytic matrix, propensity, weighted event, growth, fission, daughter, and lineage implementation |
| `src/e01_gard_historical/nondrift.py` | Historical H and both non-drift techniques |
| `src/e01_gard_historical/__init__.py` | Explicit public API and version |
| `configs/e01/s04_historical_contract.yaml` | Source/compatibility/registry contract |
| `configs/e01/s04_small_cases.yaml` | Versioned hand-oracle fixtures |
| `scripts/e01/build_historical_reference_artifacts.py` | Deterministic validation/artifact builder |
| `tests/e01/test_historical_reference.py` | Focused formula, invariant, traceability, and round-trip tests |

Repository code was not duplicated into artifacts. Historical MATLAB source was not copied into either the repository or artifacts.

## Provenance

- Date: 2026-08-01 UTC.
- Repository: `/workspace/arrival-of-self-replicators`.
- Branch: `eidosoma/groups/42`.
- S04 commit: `1f8e3b0d6fd68b9d62b566b6f237febe64e2afa9`.
- Push: successful; remote branch resolved to the same full commit.
- Historical GARD commit/tree: `86dff6320d5ae91b4e831471079ff46749b14df9` / `a602fc99b494982c04c60405bc6422af9db5a77a`.
- Original official paper: arXiv `2607.28250v1`; SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- Source manifest: SHA-256 `50a569e30826fe893258f5c0935469576008f43134eee94e8bf2654d4ef23ed5`.
- Registry v0.3.0: SHA-256 `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891`, unchanged.
- Exact input/code/output hashes are in `/artifacts/research_steps/S04/artifact_manifest.json`; that manifest excludes its own hash.

## Caveats, blockers, failed assumptions, and limitations

### Caveats and blockers

- The original author code is still `UNAVAILABLE::NO_AUTHOR_CODE_RELEASE_FOUND`. Related GARD code cannot be relabeled as that implementation.
- Legacy MATLAB RNG APIs and cross-version algorithms are not matched. Explicit draw tapes make fixtures exact without pretending to identify the author's streams.
- No MATLAB/Octave source execution was possible in this runtime. The source-conditioned port is validated by line/hash traceability and hand oracles, not original-runtime seed equality.
- No redistribution permission is inferred for the historical repository. The port is independently authored, the source remains in the S03 cache, and the artifact bundle contains no historical source copy.
- The source's `dt` is not physical time. Downstream code must not use `legacy_inverse_rate_sum` as Gillespie time.
- Historical fission is fixed-size without replacement, unlike paper/plan binomial wording.
- Historical initialization is with replacement and uses hidden global-state order, unlike paper wording.
- The historical loop has no `max_steps` branch. S04 does not guess how the paper ends a generation at 1000 updates.
- Technique-2 non-drift contains an index-zero edge failure. It remains explicit rather than silently repaired.
- Registry v0.3.0 remains non-executable. S04 supplies a named historical compatibility profile, not a paper-primary configuration.

### Failed assumptions exposed by S04

- The assumption that public historical fission implements independent `Binomial(n_i,0.5)` daughter counts failed.
- The assumption that the historical initializer matches the paper's without-replacement wording failed.
- The assumption that a historical `max_steps` rule could be ported failed because no such condition exists in `tgs_grow_v10.m` or its parameter defaults.
- The assumption that the source variable `dt` represents a sampled waiting time failed; it accumulates total rates.
- The assumption that “continued from one daughter” implies a random daughter draw failed for the public historical orchestrator, which follows output child A.
- The assumption that a library default can settle an author choice remains rejected. Historical defaults `k_f=1e-2`, `k_b=1e-4`, and uniform `rho` are source-profile facts only.

### Limitations

- S04 did not implement the paper vector-Poisson branch, a modern Gillespie engine, or the independent S05 engine.
- S04 did not perform large-sample stochastic goodness-of-fit; that remains S07 after RNG/schema freezing.
- S04 did not perform compotype clustering beyond the requested historical non-drift analysis; full label reconstruction remains S08.
- No paper claim, trajectory distribution, causal-emergence result, or intervention outcome was reproduced here.
- Simulation remains computational proxy evidence and says nothing directly about real prebiotic chemistry or biological agency.

## Recommended next action

Stop and hand control back to the Chief Scientist. S05 is eligible but unstarted. If it is separately authorized, its implementation should be structurally independent, accept explicit immutable specification IDs, and compare only declared historical/paper/modern branches. It must not turn S04's historical findings into silent paper defaults or claim exact cross-RNG trajectories.
