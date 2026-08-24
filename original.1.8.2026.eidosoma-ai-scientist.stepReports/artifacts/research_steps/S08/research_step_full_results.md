# S08 full results: Reconstruct self-replicator detection

## Top summary

- **Research step ID:** S08
- **Completion status:** Complete on 2026-08-01; S09 was not begun.
- **Artifacts written:** A frozen preregistration and amendment record; a versioned label-family contract and 44 materialized configuration IDs; checksum-protected label arrays and a row-level label table; continuous-recurrence values; binary overlap/Jaccard, binary adjusted Rand, and cluster adjusted Rand outputs with denominators; observation-, run-, temporal-, threshold-, and historical-technique diagnostics; three figures; fixture/seed/schema/registry/validation/provenance artifacts; Git-backed implementation/config/tests; and this canonical full-results report.
- **Validation result:** **PASS.** All 11 frozen input hashes and 40/40 validation checks passed. The artifacts contain 301/301 canonical rows for 43 fixture observations, including 172/172 rows for the four reference families, 170/170 frozen sensitivity evaluations, exact checksum-protected label regeneration, a schema-valid canonical round trip, source-oracle agreement, explicit zero/ineligibility handling, deterministic order/representation checks, 15/15 focused tests, 75/75 non-identity-pinned repository tests, passing Ruff checks, visual inspection of all three figures, pushed Git commits, and verified S09 absence.
- **Outcome classification:** **Supportive** for the bounded S08 hypothesis: all requested families are implementable as explicit, status-bearing configurations and their disagreements are quantifiable without hidden row deletion. The observed fixture disagreement is simultaneously constraining evidence against treating the families as interchangeable.
- **Caveats or blockers:** The historical branch is traced to pinned public GARD v10, not the unavailable author implementation. The cosine-cluster, Euclidean, and Aitchison graph settings are validation branches, not recovered paper defaults. The reference Euclidean distance 0.1, Aitchison distance 0.5, single-linkage connectivity, minimum size 3, and medoid rule have no author-source status. Retrospective configurations use future observations. Nine of 43 fixture observations remain explicitly Aitchison-ineligible because S08 applied no zero replacement; that issue remains S09-owned. Fixture overlap and ARI values are diagnostics, not baseline scientific estimates.
- **Lay summary:** Four reasonable ways to call a molecular composition a self-replicator do not always agree. The historical rule, a cosine cluster rule, a Euclidean cluster rule, and an Aitchison compositional rule were all implemented with their assumptions written down in advance. Every input point is retained, including points that Aitchison cannot process without a future zero-handling decision. On deliberately informative fixtures, the rules sometimes agreed well and sometimes disagreed completely. This means downstream findings must report label sensitivity instead of presenting one reconstruction as uniquely implied by the paper.
- **Recommended next action:** Hand control back to the Chief Scientist. S09 is eligible but must begin only after separate authorization; its task is to evaluate zero-handling transformations without retroactively changing the frozen S08 results.

## Frozen question

Can the source-traceable historical non-drift label and explicit cosine, Euclidean, and Aitchison reconstruction labels be implemented without silently resolving author-facing ambiguity, and can their disagreements be quantified on validated fixtures?

The completion criterion was operational rather than a paper-claim test: every trajectory or fixture had to receive a status-bearing output for every frozen family, with null labels and reasons retained for ineligible observations. No baseline 100-run trajectory set exists yet, so S08 did not estimate real replicator prevalence or adjudicate any paper result.

## Outcome in one sentence

Yes: the four families and three past-only companions passed all implementation gates, while pooled binary ARIs of only 0.062–0.438 and 21 retrospective-to-past-only flips on the fixtures demonstrate that metric and temporal-scope choices are consequential.

## Inputs and evidence boundaries

### Governing and paper inputs

| Input | Role | SHA-256 or identity |
| --- | --- | --- |
| `/workspace/AGENTS.md` | Workspace, artifact, validation, Git, and one-step-only contract | `85041503713d0dd36796acac13e2f8c1d840bbce521e3301da590e422de1195c` |
| `/workspace/FULL_PLAN.md` | Required label families, recurrence alternatives, leakage guardrails, and zero-handling boundary | `6e59a75d2bb23ace8110ebf3da07ddff2f3dc4ae3377cd8d14be8e8bfd22d7ee` |
| `/workspace/RESEARCH_PLAN.md` | Frozen S08 question, outputs, checks, and completion criterion | Pre-S08-update SHA-256 `51d107af11bc9a98d78547d2b9177f6bb58845ce67e7fbdf459fd83bdd1a04ef` |
| `/workspace/input-attachments/MANIFEST.json` | Uploaded-paper identity and extraction route | `d0f71c606281cf289a4b9e0852e08c1a6b889c9021d37d5d1c32c64b62f1183e` |
| Attachment sidecar `_metadata/ATTACHMENT.md` | Original filename, provenance, and extraction limitations | `983c410106015858e6a5a2234b1128af3f29d772059775aa8c33785abc0d885c` |
| Supplied paper extraction `pdf-markdown.md` | Paper Methods, Results, captions, and code-availability statement | `23ca5473759e78be12699655fbdbc143cdd3fd383e3d28485dbb3c042bd1c59a` |
| Official arXiv v1 PDF recovered in S03 | Equation-bearing original paper | `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4` |

The paper says recurring steady compositions are highly similar in Euclidean space and refers to a similarity threshold relative to a recurring composition, but it does not provide the threshold, clustering algorithm, cluster-size/persistence rule, compotype reference/tie rule, temporal information scope, molecular-step alignment, or ineligible-run policy. No author code release was found. Those absences prohibit representing the validation branches below as recovered author behavior.

### Upstream reports refreshed

| Report | SHA-256 | S08 use |
| --- | --- | --- |
| S01 full results | `4e7025fcb2aaa63eb9fad6b0760e5051b857245cfc4fd4c8840d628e44d72a97` | Claim and source-discrepancy boundary |
| S02 full results | `9bcd86ef86c371ba57e991a3bc8295cd92ef2fe05b9edf53327814b0c52f2cfa` | Ambiguity IDs, no-silent-default rule, and S08 ownership |
| S03 full results | `7ef7837bf0b2b65fd011ec5b530b90e07089df637050155b81fc08d3c99992ee` | Paper/source/runtime identities and author-code search |
| S04 full results | `62a1f862e17579769aca3939b69eba3ff725078d593ece6b29ccb58a29b8c59d` | Historical H/non-drift source behavior and HC11 oracle |
| S05 full results | `e83620da619d05687d218186cd2d2789ce26bbac6234de71990948915ea95196` | Independent-engine scope and branch distinctions |
| S06 full results | `206482bce8e8a47d5050e83c4a99370c2bedf7dd4244a066078f9ad5c3233595` | Domain-separated fixture seeds, canonical JSON, and validated trajectory view |
| S07 full results | `9262b881f1dc15392d1d674e3ac15222dc6dfef69d82ca6aeb74f9fe90fd876d` | Stochastic validation gate before scientific analysis |

### Frozen implementation and reproducibility evidence

| Input | SHA-256 | Boundary retained |
| --- | --- | --- |
| Specification registry v0.3.0 | `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891` | All 120 entries retained; 64 unresolved/conflict/deferred and 21 unexpanded branch sets; execution gate remains closed |
| Source manifest | `50a569e30826fe893258f5c0935469576008f43134eee94e8bf2654d4ef23ed5` | Immutable source identities |
| Historical behavior contract | `e6fe49aba2240047d018e5b619ef07d3e48922fb43a963256b6b2233f07d0a43` | Public historical behavior only |
| Historical compatibility notes | `f8cc4fa96d104ce1ddf3ae06da1857a24a7166fa1a2b873358fd82a7bf50c869` | Author/MATLAB/license limitations |
| Independent-engine contract | `a35e313cb0685218691397980d1f5d8020fee8c994359e3227b9b1c1ef8605e8` | Explicit independent branch semantics |
| Independent validation profiles | `959fb3171e19087af06b09d21fd499e776b16de85e0b673f9ccde61e5b23ee0c` | Fixture-only, not paper baseline settings |
| Seed derivation contract | `a4c5586fc6be012afaff21f47fae422c4d6b6c68200c236df4a5b1ea5e736bb1` | Nine PCG64DXSM domain-separated streams |
| Trajectory precision contract | `2c73d7385d7511636cb809cdb1b2b5c0239632faec2f6ff2ffb692a7b3548b4d` | Exact same-environment serialization scope |
| Trajectory JSON Schema | `981807b512bff589a6a693c1da191efad829ccb3294fd4f9297c3ee02a7a5d57` | Lossless event/generation field meanings |
| S06 example trajectory | `020634d20a248ec0516040128e37aed3d7b8f5c1c6b4fde5fca7388b65b55483` | Checksum-valid three-generation growth-final fixture |

No external dataset was required. `DATASET_AVAILABILITY.json` reports 100% coverage with zero requirements. No web source or new package installation was used.

## Preregistration and amendment chronology

Thresholds, fixtures, representation rules, temporal branches, comparison denominators, and validation gates were frozen before canonical outcomes:

1. `E01-S08-label-preregistration-v1.0.0` was committed and pushed at `ef8e3c0b4a3d7429f88d56115abb7c2ba0c096b7`.
2. Before any label outcome was generated, an audit identified that permuting observation order is appropriate for a full-trace retrospective clustering invariance check but changes a past-only estimand. Version `v1.0.1` clarified that retrospective configurations receive seeded order-permutation validation while online configurations receive sequence-preserving exact replay. It was committed and pushed at `972ad87d9b30d5fc17f58938ffdf0010eaa1eaf9`.
3. No family, threshold, minimum size, fixture, metric, or outcome rule changed in that amendment. The artifact copy has SHA-256 `78d8bd9c4cf58e3881ea8e428f2c02a557f8fc533d0c40f27f2c8d000801c9bc`.
4. The implementation was tested, committed, and pushed at `7a8df1dfd596057a8c56a5aa5fdea4d0dbdaebd9` before canonical `/artifacts` generation.

This chronology prevents post-outcome threshold selection and makes the one pre-outcome clarification inspectable.

## Methods

### 1. Four reference label families

| Family | Versioned reference configuration | Representation and metric | Strict rule | Temporal scope | Eligibility/zero rule | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| `Y_H` | `E01-S08-YH-T1-HGT090-v1.0.0` | S04 public-historical columnwise H/cosine after source L2 normalization | Source technique-1 local mean H `> 0.9` | Adjacent local rule; interior labels use the next observation | Source truncates at first zero-sum generation and pads subsequent scores/labels with zero/false; padding is explicitly marked | Source-traceable to pinned public GARD v10 only |
| `Y_C` | `E01-S08-YC-COS-HGT090-MIN3-RETRO-v1.0.0` | Raw nonnegative vectors; pinned historical H/cosine | Edge when H `> 0.9`; connected component size at least 3 | Full-trace retrospective | Zero-sum row retained with null label and reason | Validation-only reconstruction requested by S08 |
| `Y_E` | `E01-S08-YE-EUCLIDEAN-DLT010-MIN3-RETRO-v1.0.0` | L1-closed proportions; Euclidean L2 distance | Edge when distance `< 0.1`; connected component size at least 3 | Full-trace retrospective | Zero-sum row retained with null label and reason | Validation-only reconstruction; threshold is not from the paper |
| `Y_A` | `E01-S08-YA-AITCHISON-DLT050-MIN3-RETRO-v1.0.0` | L1 closure, full CLR, Euclidean distance between CLR vectors | Edge when Aitchison distance `< 0.5`; connected component size at least 3 | Full-trace retrospective | Every row with any zero component is retained with null label and reason; no replacement | Validation-only strict-positive branch; zero support deferred to S09 |

The cluster configurations use a strict-threshold undirected graph and single-linkage transitive closure. Components with at least three members are labeled replicator; smaller eligible components are labeled drift. A replicator component receives a canonical ID from its earliest stable observation ID. Its reference is the observed medoid minimizing within-component distance sum, with the earliest observation ID breaking exact ties. These choices have immutable configuration IDs but do not resolve the author-facing registry sentinels.

### 2. Past-only and continuous diagnostics

Each graph family has a separately identified past-only companion. At observation `t`, the current point is classified using only observations `1..t`; earlier labels are not backfilled after later points arrive. The result object intentionally does not expose a full-trace metric matrix for this branch.

The continuous recurrence diagnostic is

\[
R_g = \max_{h<g} H(\mathbf n_g, \mathbf n_h),
\]

with null values when no eligible past observation exists. It is saved as a continuous diagnostic and is not silently thresholded into another binary label.

### 3. Optional historical technique 2

S04's optional consecutive-H technique was evaluated for drift sizes 2, 3, and 5 at strict `H>0.9`. The pinned source shifts accepted runs backward by one index and has an invalid MATLAB index-zero case. S08 returns `ERROR_SOURCE_DOMAIN` for that case and never repairs it.

### 4. Fixtures and deterministic seeds

| Fixture | Observations | Construction and purpose |
| --- | ---: | --- |
| `TWO-ATTRACTORS` | 20 | S06-derived estimator stream; positive multinomial states around two frozen attractors with four bridge compositions and masses cycling through 80, 100, and 120 |
| `RARE-SPECIES-SWING` | 9 | Literal strictly positive states changing low-abundance components while dominant components remain stable; designed to expose metric disagreement |
| `ZERO-AND-EXTINCTION` | 6 | Literal zeros, scale-equivalent states, one zero-sum state, and later positive states; validates explicit eligibility and source truncation |
| `S04-HC11` | 5 | Exact S04 historical non-drift oracle with adjacent H values 1.0, 0.8, 0.6 and a terminal zero-sum state |
| `S06-GROWTH-FINAL` | 3 | Growth-final states extracted from the checksum-valid S06 example trajectory |

The root seed is the frozen 32-byte hexadecimal value in `fixtureSeedContract`. The `estimator` stream generates the positive multinomial fixture; the `machine_learning` stream generates retrospective order-audit permutations. The clustering algorithm itself consumes no RNG and is deterministic. All nine S06 stream identities remain in the seed manifest even though only those two purposes are consumed in S08.

### 5. Sensitivity design

The frozen grids were evaluated in full:

- historical and cosine H thresholds: 0.85, 0.875, 0.9, 0.925, 0.95;
- Euclidean distance thresholds: 0.025, 0.05, 0.075, 0.1, 0.15, 0.2;
- Aitchison distance thresholds: 0.1, 0.2, 0.35, 0.5, 0.75, 1.0;
- cluster minimum sizes: 2, 3, 4, 5 at each reference metric threshold.

This produced 170 fixture-level sensitivity rows and 44 unique materialized configuration IDs. No result-dependent configuration was promoted to a paper baseline.

### 6. Agreement and disagreement estimands

All binary comparisons use the pairwise common-nonnull denominator. The artifacts report that denominator, each family's null count, positive counts, intersection, union, Jaccard overlap, positive agreement, overall agreement, one-sided disagreements, binary ARI, and any null-metric reason. Ineligible observations remain in the label outputs and appear as explicit denominator exclusions in comparisons.

Cluster ARI is calculated only among `Y_C`, `Y_E`, and `Y_A`. Replicator components retain their component identities; eligible drift components share a `NOISE` class within each fixture. The denominator and ineligible count are saved for every matrix cell.

### 7. Serialization and provenance

The label arrays use the S06 canonical JSON subset and a SHA-256-protected envelope. Binary labels, statuses, component IDs, and reasons therefore round-trip byte-exactly without decimal-float ambiguity. Metric scores remain in the CSV where binary64 decimal rendering is deterministic at 17 significant digits. The canonical label payload SHA-256 is `12dbd2a9f16bea291ee896c0835f06d5ea4184aa8ff228ef2d4408ca3f0a683b`; the complete envelope file SHA-256 is `ee325dbbc34fa429de746cdaad4129d9dc9df17532a0912e1d2895462f8d3646`.

## Results

### Complete label coverage

Five fixtures supplied 43 observations. Seven canonical configurations—the four reference families and three past-only companions—produced exactly 301 unique rows. The four reference families produced exactly 172 rows. No observation was removed.

| Fixture | `Y_H` positive / labeled | `Y_C` positive / labeled | `Y_E` positive / labeled | `Y_A` positive / labeled | Total observations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rare-species swing | 9 / 9 | 9 / 9 | 9 / 9 | 0 / 9 | 9 |
| S04 HC11 | 1 / 5 | 0 / 4 | 0 / 4 | 0 / 1 | 5 |
| S06 growth-final | 0 / 3 | 0 / 3 | 0 / 3 | 0 / 2 | 3 |
| Two attractors | 20 / 20 | 20 / 20 | 16 / 20 | 19 / 20 | 20 |
| Zero and extinction | 2 / 6 | 3 / 5 | 0 / 5 | 0 / 2 | 6 |

`Y_H` retains source-padded false values after the first zero-sum observation; those rows are labeled and explicitly flagged rather than treated as missing. `Y_C` and `Y_E` mark zero-sum compositions ineligible. `Y_A` marks every row with any zero component ineligible. Across all fixtures this yields 9 explicit `Y_A` null labels; all have machine-readable reasons.

### Binary overlap and adjusted Rand agreement

| Pair | Common nonnull | Positive-label Jaccard | Binary ARI | Disagreements / common |
| --- | ---: | ---: | ---: | ---: |
| `Y_H` vs `Y_C` | 41 | 0.829 | 0.438 | 6 / 41 |
| `Y_H` vs `Y_E` | 41 | 0.781 | 0.415 | 7 / 41 |
| `Y_H` vs `Y_A` | 34 | 0.655 | 0.151 | 10 / 34 |
| `Y_C` vs `Y_E` | 41 | 0.781 | 0.415 | 7 / 41 |
| `Y_C` vs `Y_A` | 34 | 0.613 | 0.069 | 12 / 34 |
| `Y_E` vs `Y_A` | 34 | 0.571 | 0.062 | 12 / 34 |

The Jaccards are moderately high because several fixtures contain many positives, while ARI corrects for agreement expected from the marginal label proportions and is much lower. On the rare-species fixture, `Y_H`, `Y_C`, and `Y_E` label all nine observations positive while `Y_A` labels all nine drift at its reference threshold, giving 100% pairwise disagreement with `Y_A`. This is an intended metric stress test, not an estimate of scientific effect size.

### Cluster-partition agreement

| Pair | Common nonnull | Explicitly excluded as ineligible | Cluster ARI |
| --- | ---: | ---: | ---: |
| `Y_C` vs `Y_E` | 41 | 2 | 0.539 |
| `Y_C` vs `Y_A` | 34 | 9 | 0.564 |
| `Y_E` vs `Y_A` | 34 | 9 | 0.865 |

Binary self-replicator status and detailed cluster partition are different estimands. The relatively high `Y_E`/`Y_A` cluster ARI coexists with low binary ARI because their minimum-size classification and eligibility differ.

### Retrospective information dependence

| Family | Common nonnull | Retrospective/past-only flips | Retrospective positive, past-only negative | Reverse flips |
| --- | ---: | ---: | ---: | ---: |
| `Y_C` | 41 | 6 | 6 | 0 |
| `Y_E` | 41 | 7 | 7 | 0 |
| `Y_A` | 34 | 8 | 8 | 0 |
| **Total** | — | **21** | **21** | **0** |

Every flip is retrospective backfilling: a point belongs to a sufficiently large full-run component only after future members arrive. This is direct fixture evidence for the caveat that paper-like retrospective labels cannot be treated as prospective targets without a separate information-scope verdict.

### Threshold sensitivity

The complete 170-row table shows large sensitivity in the deliberately informative fixtures:

- two-attractor `Y_E` positive fraction ranges from 0 to 1 across the frozen Euclidean grid;
- two-attractor and rare-species `Y_A` positive fractions each range from 0 to 1 across the Aitchison grid;
- rare-species `Y_E` ranges from 0.667 to 1;
- historical `Y_H` is stable on the rare-species, S06, and zero fixtures but changes from 0.4 to 0.2 on HC11 and from 1.0 to 0.9 on the two-attractor fixture;
- cosine clustering is comparatively stable on these fixtures, with the two-attractor positive fraction changing only from 1.0 to 0.95.

These patterns validate the sensitivity machinery and constrain downstream interpretation. They do not identify an optimal author threshold.

### Historical behavior checks

The S04 HC11 trace reproduced exactly:

- incoming/source-aligned H values: `[1.0, 1.0, 0.8, 0.6, 0.0]`;
- local technique-1 scores: `[1.0, 0.9, 0.7, 0.6, 0.0]`;
- strict `>0.9` labels: `[true, false, false, false, false]`;
- the terminal zero-sum observation is source-padded and explicitly marked.

Historical technique 2 produced 7 valid diagnostic records and 8 explicit source-domain errors across the five fixtures and three drift sizes. No source repair was applied.

### Anchor interpretation

The implementation hypothesis passed, but the scientific takeaway is a constraint: self-replicator status is not metric- or information-scope-invariant on the available fixtures. Downstream association, prediction, and intervention work must keep family and temporal-scope IDs in every result and must not collapse ineligible Aitchison rows into drift or silently adopt a pseudocount.

## Validation

### Preregistered checks

All 40 checks passed. They cover:

| Validation group | Result |
| --- | --- |
| Frozen evidence | 11/11 hashes matched exactly |
| Preregistration | Freeze commit is an ancestor; artifact/source bytes and hashes match; outcomes were recorded absent at freeze |
| Upstream fixtures | S04 HC11 oracle and checksum-valid S06 growth-final view match |
| Determinism | Seeded fixture/permutation replay exact; label-array payload and serialized bytes exact |
| Representation | Historical and all three graph metrics invariant to positive row scaling and a common component permutation |
| Clustering order | All retrospective graph branches invariant after seeded row permutation and restoration by observation identity |
| Past-only replay | All three online branches replay exactly with the sequence preserved; no full-trace metric matrix is exposed |
| Strict comparisons | Similarity exactly 0.9 and distance exactly 0.1 are rejected, confirming `>` and `<` rather than inclusive rules |
| Eligibility | Aitchison zeros remain null with reasons; negative and nonfinite states fail closed |
| Historical edge behavior | HC11 exact; technique-2 index-zero failures retained; zero repairs |
| Coverage and schema | 301/301 canonical rows, 172/172 reference rows, unique identities, no unexplained nulls, Draft 2020-12 conformance, checksum and canonical round trip |
| Sensitivity and comparisons | 170/170 sensitivity rows, 96 overlap records, 30 unordered run-pair diagnostics, all cluster-ARI denominators present |
| Registry and scope | Registry byte-identical with all nine S08-owned values/statuses unchanged; execution gate closed; S09 absent |
| Figures | Three nonempty plots written and visually inspected for legibility and consistency with tables |

### Repository tests

The focused suite passed 15/15 tests. The repository-wide command collected 76 tests and passed all 75 selected non-identity-pinned tests; the one deselected S06 test intentionally asserts the old exact repository/source identity and is expected to reject later commits. The separate S06 identity-change rejection test remained selected and passed.

Ruff format and lint checks passed. `git diff --cached --check` passed before commit. The working tree was clean and synchronized with `origin/eidosoma/groups/42` after the implementation push.

### Visual validation

`label_disagreement_map.png` shows expected blue/white/gray patterns for replicator/drift/ineligible states, including complete rare-species disagreement for `Y_A`, explicit Aitchison gray cells at zeros, and metric-specific bridge behavior. `threshold_sensitivity.png` agrees with the tabulated ranges. `binary_label_overlap.png` agrees with the pooled Jaccard matrix to two displayed decimal places. No clipped labels or unreadable panels were observed.

## Commands

The principal commands were:

```bash
# Freeze and validate the pre-outcome configuration.
python - <<'PY'
# Loaded the YAML, checked all frozen hashes and all nine S08 registry owners.
PY
git commit -m "Preregister E01 S08 label reconstruction"
git push origin eidosoma/groups/42
git commit -m "Clarify S08 temporal determinism audit"
git push origin eidosoma/groups/42

# Format, lint, and validate implementation.
ruff format scripts/e01/build_s08_label_artifacts.py src/e01_replicator_labels tests/e01/test_replicator_labels.py
ruff check scripts/e01/build_s08_label_artifacts.py src/e01_replicator_labels tests/e01/test_replicator_labels.py
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/e01/test_replicator_labels.py
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/e01 -k 'not checksum_tamper_detection_and_exact_same_engine_regeneration'

# Commit implementation before canonical outcome generation.
git commit -m "Implement E01 S08 replicator label audit"
git push origin eidosoma/groups/42

# Canonical result generation.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/e01/build_s08_label_artifacts.py --artifacts-dir /artifacts
```

The builder was also run twice in an isolated temporary artifact root. The label-array file hash and preregistration-record bytes were identical across the two runs.

## Dependencies and compute

| Dependency | Version |
| --- | --- |
| Python | 3.13.14 |
| NumPy | 2.4.6 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |
| Matplotlib | 3.11.1 |
| PyYAML | 6.0.3 |
| jsonschema | 4.26.0 |
| pytest | 9.1.1 |
| Ruff | 0.16.0 |

No dependency was installed or updated. Execution was serial and CPU-only; no process pool, GPU, or model training was used. The container exposed 24 logical CPUs, but the compact deterministic fixture work did not benefit from parallelism. No explicit numeric thread environment override was set for S08; operations were small pairwise matrices and deterministic graph traversals rather than parallel reductions.

## Artifacts written

### Canonical S08 directory

| Artifact | Purpose |
| --- | --- |
| `research_step_full_results.md` | Canonical detailed handoff report |
| `preregistration.yaml`, `preregistration_record.json` | Byte-identical frozen configuration, commit chronology, and pre-outcome absence record |
| `label_outputs.csv` | 301 row-level status-bearing labels, scores, component/reference identities, and reasons |
| `label_arrays.json` | Checksum-protected canonical arrays for seven canonical configurations |
| `continuous_recurrence.csv` | Past-only continuous `R_g` values without binary thresholding |
| `label_overlap_long.csv` | Fixture and pooled overlap/Jaccard/binary-ARI records with explicit denominators |
| `binary_jaccard_matrix.csv`, `binary_ari_matrix.csv` | Pooled four-family matrices |
| `cluster_ari_matrix.csv`, `cluster_ari_denominators.csv` | Three-family cluster partitions and exact denominators |
| `run_level_disagreement.csv`, `disagreement_diagnostics.csv` | Pairwise fixture and per-observation disagreement diagnostics |
| `temporal_scope_diagnostics.csv` | Retrospective versus past-only flips |
| `threshold_sensitivity.csv` | Complete 170-row threshold and minimum-size audit |
| `historical_technique2_diagnostics.json` | Optional source behavior and unrepaired source errors |
| `edge_case_validation.json` | Strict boundaries, zeros, invalid-state injections, and source-edge checks |
| `fixture_catalog.json`, `seed_manifest.json` | Exact fixture states, hashes, seed identities, and order permutations |
| `label_disagreement_map.png`, `threshold_sensitivity.png`, `binary_label_overlap.png` | Visual diagnostics |
| `registry_preservation.json` | Before/after registry identity and nine-owner snapshot |
| `validation_summary.json` | Machine-readable 40-check result and anchor metrics |
| `artifact_manifest.json` | Paths, sizes, hashes, repository files, commits, and S09 absence |

### Shared reusable label bundle

| Artifact | Path | Purpose |
| --- | --- | --- |
| Label-family contract | `/artifacts/E01_forensic_replication_bundle/labels/label_family_contract_v1.0.1.yaml` | Source boundaries, family rules, temporal/zero policy, and registry sentinels |
| Materialized configurations | `/artifacts/E01_forensic_replication_bundle/labels/clustering_configurations_v1.0.1.yaml` | All 44 immutable reference, companion, sensitivity, and diagnostic IDs |
| Label-array JSON Schema | `/artifacts/E01_forensic_replication_bundle/labels/label_arrays_schema_v1.0.0.json` | Draft 2020-12 contract for checksum-protected arrays |

`RESEARCH_PLAN.md` did not request `status.json`, so none was created. `validation_summary.json` is a scientific validation artifact, not a workflow-status substitute.

## Provenance

- Repository: `/workspace/arrival-of-self-replicators`
- Branch: `eidosoma/groups/42`
- Pre-S08 repository state: `08033e3dfe1c5b63389ff91021f24ec5c07ea194`
- Initial preregistration commit: `ef8e3c0b4a3d7429f88d56115abb7c2ba0c096b7`
- Pre-outcome clarification commit: `972ad87d9b30d5fc17f58938ffdf0010eaa1eaf9`
- Implementation commit: `7a8df1dfd596057a8c56a5aa5fdea4d0dbdaebd9`
- All three commits pushed to `origin/eidosoma/groups/42`
- Label-array envelope file SHA-256: `ee325dbbc34fa429de746cdaad4129d9dc9df17532a0912e1d2895462f8d3646`
- Label-output CSV SHA-256: `24fd8ed34ab49e9fbde511d78bb653445ce37872990a8fe676827ed0d800e08b`
- Label-family contract SHA-256: `4d1bf4d1f57fbf43a49bfd95f072f33e4f78782e2afb1cded9dc1ed8a3a689d5`
- Materialized-configurations SHA-256: `535db35cb5e477f871c059a66daa4ed4ac6b6255c70942248b136f99a645cc07`
- Full output hashes and sizes: `/artifacts/research_steps/S08/artifact_manifest.json` (self-hash excluded)

Repository source remains in Git and was not copied into `/artifacts`, as required by the repository workspace contract.

## Caveats, blockers, failed assumptions, and limitations

### Evidence limitations

- The paper's author implementation remains unavailable. Public historical GARD v10 is a source-traceable comparison layer, not author-code identity.
- The paper mentions Euclidean similarity but omits a threshold and clustering procedure. S08 therefore cannot establish an exact Euclidean author label.
- `Y_C` is a separately named cosine-cluster audit branch required by the S08 instruction; it is not silently inserted into the registry's existing family set as an author family.
- The graph algorithm can chain points through transitive single-linkage connections. This is explicit and sensitivity-audited, but other legitimate algorithms may differ.
- The medoid reference and earliest-ID tie break are validation choices. They do not resolve `labels.compotype.reference_definition`.

### Retrospective-label limitation

- Full-trace cluster labels use future observations. The 21 observed backfills demonstrate this limitation rather than removing it.
- The past-only companions are prefix graph classifiers, not fixed training-derived compotypes. The registry's `training_derived_reference` branch remains unimplemented and unresolved.
- The historical technique-1 interior score itself uses the next adjacent observation, so it is not a strictly online target.

### Zero and representation limitation

- S08 deliberately did not apply pseudocounts, multiplicative replacement, dropped CLR, or ILR. Doing so would have begun S09 and silently changed its frozen question.
- Aitchison results are therefore limited to strictly positive fixture rows. All zero-containing rows remain in the output as null with reasons.
- The pooled Aitchison pairwise denominators are 34 rather than 43. Every matrix includes denominator/exclusion records to prevent accidental comparison against the wrong total.

### Fixture limitation

- The fixtures validate geometry, status handling, source compatibility, and disagreement accounting. They do not represent the paper's 100-run baseline distribution.
- The S06 upstream trajectory has only three growth-final states and is useful for schema/alignment validation, not cluster discovery power.
- The high or low overlap values cannot be interpreted as replication, nonreplication, biological effect size, or expected future prevalence.

### Preserved source failure

- Historical technique 2 failed in 8/15 fixture/drift-size combinations because a qualifying run began at the source's invalid index-zero boundary. Repairing it would create a modernized branch, so S08 preserved and reported the error.

### Pre-outcome amendment

- The initial preregistration's order-audit wording was too broad for a past-only estimator. It was corrected before outcomes and retained in Git history. This was a specification clarification, not a result-driven change; thresholds and outcome rules remained identical.

### Remaining downstream blockers

- Registry v0.3.0 remains non-executable and unchanged. Its S08-owned unresolved sentinels and branch sets remain evidence boundaries rather than being overwritten by fixture settings.
- S09 must establish defensible zero transforms before Aitchison can be evaluated on general GARD trajectories.
- Molecular-step versus generation alignment remains unresolved. S08 used explicitly named fixture views and did not broadcast labels to event-level records.
- No S09 file, code, directory, or analysis was created.

There is no blocker to declaring S08 complete. The blockers above constrain downstream interpretation and configuration eligibility.
