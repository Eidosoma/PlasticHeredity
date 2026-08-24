# S02 full results: Create the ambiguity and discrepancy ledger

## Top summary

- **Research step ID:** S02
- **Completion status:** Complete on 2026-08-01; S03 was not begun.
- **Artifacts written:** Versioned ambiguity ledger (CSV and Markdown), discrepancy taxonomy, specification registry, claim-to-ambiguity crosswalk, validation summary, artifact/provenance manifest, this full-results report, and repository-backed schema, inventory, generator, and tests.
- **Validation result:** PASS. The generator produced 105 unique ambiguity items, 120 total registry parameters, direct coverage of all 59 S01 claims and all 12 S01 discrepancies, exact sentinels for every unresolved item, no blank primary values, no silent defaults, and 6/6 passing tests. Ruff and independent structural audits also passed with no errors or warnings.
- **Outcome classification:** **Supportive** for the frozen S02 hypothesis within the supplied source corpus: every material undocumented choice found in the paper, S01 ledgers, and governing plans is represented by a versioned parameter and an explicit value, branch set, conflict, or unresolved sentinel.
- **Caveats or blockers:** The registry is deliberately non-executable. Sixty-seven items remain unresolved, conflicting, or dependent on later evidence, including 39 critical items; another 21 are unexpanded frozen branch sets. The supplied paper extraction omits the central equations, the author code is absent, and source/version recovery belongs to S03. Completeness is relative to the available paper extraction, S01 evidence, and plans; later source inspection may require versioned additions rather than silent edits.
- **Lay summary:** The paper does not say exactly how several important simulation and analysis decisions were made. This step converted those gaps into 105 named decisions. Known choices are written down, disagreements are preserved as separate alternatives, and unknown choices are marked with identifiers that software must reject. This prevents later code from quietly choosing convenient defaults.
- **Recommended next action:** Hand control back to the Chief Scientist. If separately authorized, execute S03 to pin source and environment identities and resolve only the evidence-owned S03 sentinels; do not treat branch choices as outcome-tunable defaults.

## Frozen question

Can every material undocumented choice be enumerated before final results are inspected and mapped to a versioned specification parameter or an explicit unresolved sentinel?

The operational success criterion was that no silent default remain in the primary reconstruction. “Every” is bounded to choices discoverable in the supplied paper extraction, its attachment metadata, S01 claim and reconciliation artifacts, `FULL_PLAN.md`, and `RESEARCH_PLAN.md`. No downstream result was inspected because no reconstruction result exists yet.

## Inputs

| Input | Role in S02 | Validation or provenance |
| --- | --- | --- |
| `/workspace/AGENTS.md` | Workspace execution, artifact, validation, and handoff contract | Refreshed before action; pre-execution SHA-256 `85041503713d0dd36796acac13e2f8c1d840bbce521e3301da590e422de1195c` |
| `/workspace/FULL_PLAN.md` | Frozen parameter values, required sensitivity branches, prospective-analysis constraints, and intervention alternatives | Refreshed in full; SHA-256 `6e59a75d2bb23ace8110ebf3da07ddff2f3dc4ae3377cd8d14be8e8bfd22d7ee` |
| `/workspace/RESEARCH_PLAN.md` | S02 frozen question, named choices, outputs, and completion criterion | Refreshed before action; pre-S02-update SHA-256 `231200e008ba54cdf498d2aa1ab167ce9ddbc5db953e0aab3149d9c574a1f9f5` |
| `/workspace/input-attachments/MANIFEST.json` | Attachment identity and extraction route | SHA-256 `d0f71c606281cf289a4b9e0852e08c1a6b889c9021d37d5d1c32c64b62f1183e` |
| `/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/_metadata/ATTACHMENT.md` | Attachment provenance, limitations, and original filename | SHA-256 `983c410106015858e6a5a2234b1128af3f29d772059775aa8c33785abc0d885c` |
| `/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md` | Supplied representation of the original paper, including Methods, Results, tables, and captions | SHA-256 `23ca5473759e78be12699655fbdbc143cdd3fd383e3d28485dbb3c042bd1c59a`; two central equations are marked `formula-not-decoded` |
| `/artifacts/research_steps/S01/research_step_full_results.md` | S01 findings, limitations, and handoff | SHA-256 `4e7025fcb2aaa63eb9fad6b0760e5051b857245cfc4fd4c8840d628e44d72a97` |
| `/artifacts/E01_forensic_replication_bundle/ledgers/claim_ledger.csv` | All 59 preregistered claim targets and their discrepancy links | 59 unique claims; SHA-256 `e9cd4148dd47fa538fde9a683ce60d17bfc9c781a7cb0f7438e199d68801eafc` |
| `/artifacts/research_steps/S01/source_reconciliation.csv` | Twelve independent Results-versus-caption/table reconciliations | 12 records (`R01`–`R12`); SHA-256 `4027bd137643efd554004bd0e6980aca194af181c8ca96b98246bdc0a6e45d3d` |

No dataset, previous-artifact mount, web source, external repository, or downstream result was used. `DATASET_AVAILABILITY.json` reports no required dataset, and no new software was installed.

## Methods

### 1. Three-pass ambiguity extraction

The inventory was assembled before any simulation or estimator result using three evidence passes:

1. The supplied paper representation was read across Methods, Results, Table 1, figure captions, code availability, and formula placeholders. Explicit constants were separated from missing algorithmic semantics.
2. Every S01 discrepancy (`D01`–`D12`) and every S01 claim row was revisited. Each discrepancy received at least one ambiguity item, and every claim carrying a discrepancy received a direct link to an item carrying the same discrepancy ID.
3. `FULL_PLAN.md` and the S02 queue were audited for prospective constraints and branch grids, including the exact user-named choices. Plan-specified alternatives were retained as branch sets; historical values named only as references were not promoted to author defaults.

The resulting inventory spans source provenance, GARD dynamics, compositional preprocessing, replicator labels, Phi estimation, descriptive statistics, prediction, and intervention outcomes.

### 2. Traceable row and parameter contract

Each ambiguity row has a unique ID (`E01-A001`–`E01-A105`) and a unique dotted specification parameter. It records materiality, source evidence, the ambiguity itself, admissible alternatives, an explicit primary value or sentinel, resolution status and basis, downstream owner step, affected S01 claim IDs, S01 discrepancy IDs, silent-default risk, a validation rule, and registry version.

The version identifiers are:

- ambiguity ledger: `E01-S02-v1.0.0`
- specification registry: `E01-specification-registry-v0.2.0`
- ambiguity schema: `1.0.0`

The registry version is pre-execution (`v0.2.0`) because evidence blockers and unexpanded branch sets intentionally remain.

### 3. Resolution semantics

| Status | Count | Operational meaning |
| --- | ---: | --- |
| `PAPER_FIXED` | 4 | Explicit in the supplied paper and recorded literally. |
| `PLAN_FIXED` | 7 | Prospectively fixed by `FULL_PLAN.md`. |
| `PROVISIONAL_PRIMARY` | 5 | Explicit provisional choice with a stated basis and later sensitivity obligation. |
| `RECONCILED` | 1 | Apparently inconsistent statements can coexist after preserving their scopes or denominators. |
| `FROZEN_BRANCH_SET` | 21 | All listed alternatives are prospective; each must later receive a distinct specification ID. |
| `CONFLICT_PRESERVED` | 8 | Incompatible source-supported interpretations remain separate; an unqualified value is prohibited. |
| `UNRESOLVED_REQUIRED` | 45 | A required method value is absent and must be resolved before its owner step runs. |
| `DEFERRED_EVIDENCE` | 14 | Resolution needs source or implementation evidence assigned to a later step. |

`UNRESOLVED_REQUIRED` and `DEFERRED_EVIDENCE` values must equal `UNRESOLVED::<ambiguity-id>` exactly. Conflicts must begin `CONFLICT::` and enumerate at least two alternatives. Branch sets must begin `BRANCH_SET::` and enumerate at least two alternatives. Fixed, provisional, and reconciled statuses may not use any of these sentinels.

### 4. Registry execution gate

The registry combines 15 unambiguous paper parameters from S01/plan context with the 105 S02 parameters, for 120 total records. It is a decision registry, not yet a runnable primary configuration:

- 67 parameters are unresolved, conflicting, or evidence-deferred;
- 21 parameters are explicit but still require branch expansion;
- therefore 88 parameters block direct execution;
- `executionGate.executable` is `false`;
- validation requires `noSilentDefaults: true`.

This distinction permits a complete S02 handoff without pretending that unknown kinetics, estimator identities, or source conflicts have been solved.

### 5. Deterministic generation

The repository-backed YAML inventory is the single editable source. The Python generator expands `ALL`, individual claim IDs, and inclusive claim-ID ranges; validates the full contract; then writes the CSV/Markdown ledger, discrepancy taxonomy, claim crosswalk, registry, validation JSON, and manifest. Generated research artifacts were written only under `/artifacts`; repository code remained in Git.

## Results

### Inventory coverage

| Category | Items |
| --- | ---: |
| Source provenance | 4 |
| GARD dynamics | 20 |
| Preprocessing | 9 |
| Replicator labels | 9 |
| Phi estimation | 18 |
| Descriptive statistics | 17 |
| Prediction | 11 |
| Intervention outcomes | 17 |
| **Total** | **105** |

Materiality was 60 critical, 40 high, and 5 medium. Among the 67 unresolved/conflict/evidence-deferred items, 39 are critical, 26 high, and 2 medium. These counts are decision counts, not evidence weights or replication scores.

All 59 claims are mapped. Shared upstream assumptions deliberately link broadly: the minimum, median, and maximum ambiguity counts per claim are 53, 69, and 72. This high coverage reflects common source, simulation, preprocessing, and estimator dependencies, not 53 independent claim-specific discrepancies.

### Required S02 choices

| Named choice | Ambiguity ID | Status | Explicit registry value or sentinel |
| --- | --- | --- | --- |
| `k_f` | `E01-A009` | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A009` |
| `k_b` | `E01-A010` | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A010` |
| `rho_i` | `E01-A011` | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A011` |
| Initial state | `E01-A019` | `PROVISIONAL_PRIMARY` | 40 distinct uniformly sampled types, one molecule per selected type |
| Stochastic update semantics | `E01-A013` | `FROZEN_BRANCH_SET` | paper vector-Poisson, historical loop, or direct Gillespie |
| Zero replacement | `E01-A027` | `FROZEN_BRANCH_SET` | pseudocount grid or multiplicative replacement |
| Estimator | `E01-A045` | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A045` |
| Window | `E01-A048` | `FROZEN_BRANCH_SET` | 32, 64, 128, 256, or whole-trajectory descriptive |
| Lag | `E01-A049` | `FROZEN_BRANCH_SET` | 1, 2, 4, or 8 |
| MIB normalization | `E01-A055` | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A055` |
| Partition search | `E01-A056` | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A056` |
| Spike definition/scope | `E01-A064`–`E01-A066` | unresolved plus frozen branches | Threshold scope remains explicit unresolved; signed/absolute SD and MAD branches are preserved |
| Label threshold | `E01-A035` | `PLAN_FIXED` | historical label uses `H > 0.9`; sensitivity grid must be versioned before S08 results |
| Consistency formula | `E01-A098` | `FROZEN_BRANCH_SET` | `C_Y`, `C_H`, or `C_Z` |
| Machine-learning layout | `E01-A080` | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A080` |
| Intervention scoring | `E01-A092` | `FROZEN_BRANCH_SET` | append-only primary, one-step expected, or generation rollout |

No historical `k_f`, `k_b`, or `rho_i` value was silently adopted. Values mentioned in `FULL_PLAN.md` as historical references remain candidates only until source evidence is pinned and evaluated in the owning step.

### S01 discrepancy preservation

| S01 ID | Ambiguity mapping | Classification and handling |
| --- | --- | --- |
| `D01` | `E01-A070` | Scope-preserving reconciliation: retain raw count 54 and report both 54/100 and 54/73. |
| `D02` | `E01-A068`–`E01-A069` | Preserve Phi level versus first-difference estimands as separate branches; alignment remains unresolved. |
| `D03` | `E01-A039`, `E01-A050`–`E01-A053`, `E01-A078`–`E01-A081`, `E01-A083`–`E01-A088` | Preserve full state-trajectory versus strict first-event endpoints and prospective versus retrospective information access. |
| `D04` | `E01-A100`–`E01-A101` | Preserve percent-cell versus molecular-step conflict; censoring and equivalence remain unresolved. |
| `D05` | `E01-A098`–`E01-A099` | Preserve the “min worsened” text versus Table 1’s higher min consistency, and run all three consistency formulas. |
| `D06` | `E01-A097` | Preserve overall occupancy, per-generation trajectory, and final-generation probability scopes. |
| `D07` | `E01-A059`, `E01-A064`–`E01-A067`, `E01-A076` | Preserve signed Phi values and explicit positive, negative, and absolute spike branches; threshold scope/count remain unresolved. |
| `D08` | `E01-A001`, `E01-A004`, `E01-A029`, `E01-A043`–`E01-A045`, `E01-A048`–`E01-A049`, `E01-A054`–`E01-A056`, `E01-A092`–`E01-A093` | Keep formula, atom, estimator, MIB, partition, and intervention-score dependencies unresolved or branched pending authoritative evidence. |
| `D09` | `E01-A104`–`E01-A105` | Preserve regression-slope versus correlation-coefficient interpretations; aggregation model remains unresolved. |
| `D10` | `E01-A102`–`E01-A103` | Do not infer SD, SE, or CI half-width; keep inference branches distinct. |
| `D11` | `E01-A041`, `E01-A071`, `E01-A073`–`E01-A074` | Preserve pooled-step, runwise, and run-summary test scopes and leave Fisher-combination details explicit unresolved. |
| `D12` | `E01-A071`, `E01-A077` | Preserve the reported nonsignificance while marking the missing coefficient, p-value, and equivalence criterion unresolved. |

The machine-readable taxonomy contains source summaries, affected claim IDs, specification parameters, statuses, preservation class, handling rule, silent-default prohibition, and downstream owner for all 12 records.

### Downstream ownership

Every unresolved decision has an owner step rather than a generic future placeholder. Item counts by owner are: S03 6, S04 19, S06 2, S08 9, S09 8, S10 8, S11 8, S14 7, S15 8, S16 13, and S17 17. Ownership is not authorization: S02 did not execute any of these steps.

## Commands and dependencies

The final implementation and validation commands were:

```bash
ruff format scripts/e01/build_ambiguity_ledger.py tests/e01/test_ambiguity_ledger.py
ruff check scripts/e01/build_ambiguity_ledger.py tests/e01/test_ambiguity_ledger.py
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/e01/test_claim_ledger.py tests/e01/test_ambiguity_ledger.py
./scripts/e01/build_ambiguity_ledger.py --artifacts-dir /artifacts
git diff --cached --check
git commit -m "Add E01 S02 ambiguity registry"
git push origin eidosoma/groups/42
```

Runtime dependencies were Python 3.13.14, PyYAML 6.0.3, pytest 9.1.1, and Ruff 0.16.0. No dependency was installed or changed. The step was deterministic and serial; no CPU/GPU research computation was needed.

## Validation

### Automated tests

The combined S01/S02 suite passed `6/6` tests in the final pre-commit run. The S02 tests independently rebuild an S01 claim ledger in a temporary artifact root, then verify:

- exactly 105 ambiguity rows and 105 unique ambiguity parameters;
- complete mapping of 59 claims and 12 discrepancies;
- direct links for every discrepancy-bearing claim;
- coverage of all 16 choices named in S02;
- correct exact unresolved sentinels and representative branch/conflict anchors;
- 120 registry parameters and a closed execution gate;
- presence and nonzero size of every generated S02 artifact;
- absence of an S03 output directory.

Ruff lint and format checks passed. `git diff --cached --check` found no whitespace errors.

### Independent completeness checks

A separate post-generation audit read the CSV/YAML artifacts rather than the in-memory generator objects. It confirmed:

- ambiguity IDs are the uninterrupted sequence `E01-A001` through `E01-A105`;
- all 18 required ledger columns are present and every required cell is nonblank;
- every unresolved/evidence-deferred value exactly matches its row ID;
- every conflict and frozen branch set contains multiple explicit alternatives;
- S01 reconciliation records `R01`–`R12` and discrepancy IDs `D01`–`D12` are all represented;
- the registry has 120 unique parameters, `noSilentDefaults: true`, and `executable: false`;
- the validation output contains no errors or warnings.

The canonical validation result is `/artifacts/research_steps/S02/validation_summary.json`.

## Artifacts written

| Artifact | Path | Purpose |
| --- | --- | --- |
| Ambiguity ledger CSV | `/artifacts/E01_forensic_replication_bundle/ledgers/ambiguity_ledger.csv` | Authoritative 105-row machine-readable ambiguity inventory |
| Ambiguity ledger Markdown | `/artifacts/E01_forensic_replication_bundle/ledgers/ambiguity_ledger.md` | Human-readable status summary and complete item index |
| Discrepancy taxonomy | `/artifacts/E01_forensic_replication_bundle/ledgers/discrepancy_taxonomy.csv` | Traceable handling of `D01`–`D12` |
| Specification registry | `/artifacts/E01_forensic_replication_bundle/specifications/specification_registry.yaml` | Versioned known values, branches, conflicts, unresolved sentinels, and execution gate |
| Claim crosswalk | `/artifacts/research_steps/S02/claim_ambiguity_map.csv` | All 59 S01 claims mapped to ambiguity and discrepancy items |
| Validation summary | `/artifacts/research_steps/S02/validation_summary.json` | Machine-readable counts, coverage, gate state, errors, and warnings |
| Provenance manifest | `/artifacts/research_steps/S02/artifact_manifest.json` | Input/code/output paths, sizes, SHA-256 hashes, and Git commit |
| Full-results report | `/artifacts/research_steps/S02/research_step_full_results.md` | Canonical S02 handoff |
| Repository schema | `/workspace/arrival-of-self-replicators/configs/e01/ambiguity_schema.yaml` | Schema, enums, known parameters, and validation contract |
| Repository inventory | `/workspace/arrival-of-self-replicators/configs/e01/ambiguity_targets.yaml` | Single source for the 105 curated items |
| Repository generator | `/workspace/arrival-of-self-replicators/scripts/e01/build_ambiguity_ledger.py` | Deterministic artifact builder and validator |
| Repository tests | `/workspace/arrival-of-self-replicators/tests/e01/test_ambiguity_ledger.py` | Focused schema, anchor, crosswalk, and output tests |

`RESEARCH_PLAN.md` did not require a compact `status.json`, so none was created. The structured validation summary is not a workflow status file.

## Provenance

- Repository: `/workspace/arrival-of-self-replicators`
- Branch: `eidosoma/groups/42`
- Committed and pushed implementation: `5b8812fabe1fe548ee58a7e3aa6f08d1709d6d55`
- Source branch state before S02 implementation: `a884967dbd2d0187cdd16df0a8d0f28e54fd4060`
- Artifact date: 2026-08-01 UTC
- Input, code, and output file hashes: `/artifacts/research_steps/S02/artifact_manifest.json` (the manifest is excluded from its own hash list).

The original PDF binary was not materialized in the workspace. The supplied attachment manifest and sidecar identify it, while the Docling Markdown extraction is the accessible paper representation. S02 did not retrieve replacement equations or source code because immutable source collection is the frozen purpose of S03.

## Caveats, blockers, failed assumptions, and limitations

- The assumption that the paper prose alone determines a runnable GARD model fails: `k_f`, `k_b`, `rho_i`, update exposure, molecular-step semantics, nonnegativity, growth boundary, daughter choice, and continuation semantics remain absent.
- The assumption that the supplied extraction can identify the Phi-r atom and equation fails because central displayed equations are undecoded. Estimator, redundancy, MIB objective/normalization, and partition search therefore remain evidence-deferred.
- Source contradictions cannot be resolved by majority wording. D01 is reconcilable only by retaining two denominators; D02–D06, D09, and D11 require separate scopes or interpretations; D07, D08, D10, and D12 retain explicit unresolved fields.
- A `PROVISIONAL_PRIMARY` value is an explicit reconstruction proposal, not evidence of author intent. It remains subject to the frozen sensitivity branches and owner-step validation.
- The no-silent-default result is a structural completeness claim, not proof that no future source inspection will reveal another choice. New evidence must add or amend a versioned item with provenance.
- No paper claim received `EXACT`, `DIRECTIONAL`, `NONREPLICATION`, or `UNDERDETERMINED` evidence in S02; this step created prerequisites only.
- The registry must not be passed directly to a simulation runner. Its 67 unresolved/conflict sentinels must be resolved by evidence or explicit pre-result specifications, and its 21 branch sets must be expanded into separate immutable specification IDs.

## Outcome and handoff

S02 is **supportive**: the scoped methodological and source ambiguities were enumerated, versioned, linked to all claims and discrepancies, and guarded against silent execution. The main constraining result is that the reconstruction is not yet executable: 88 parameters require either resolution or branch expansion. Control is returned to the Chief Scientist. S03 remains queued and unstarted.
