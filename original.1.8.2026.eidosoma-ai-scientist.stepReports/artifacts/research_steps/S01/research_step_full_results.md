# S01 — Create the E01 claim ledger: full results

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S01** |
| Completion status | **Complete** — the frozen S01 question was answered; S02 was not started. |
| Artifacts written | `/artifacts/E01_forensic_replication_bundle/ledgers/claim_ledger.csv`; `/artifacts/E01_forensic_replication_bundle/ledgers/claim_ledger.md`; `/artifacts/research_steps/S01/source_reconciliation.csv`; `/artifacts/research_steps/S01/validation_summary.json`; `/artifacts/research_steps/S01/artifact_manifest.json`; this report. Repository code/config/tests were committed at `a884967dbd2d0187cdd16df0a8d0f28e54fd4060`. |
| Validation result | **PASS** — 59 unique claim IDs, 59/59 rows complete for every required field, all 11 required claim families present, all 12 Table 1 cells represented separately, all required numerical anchors found, all 12 discrepancy IDs resolved to claim rows, 12 cross-source reconciliation records covering 33 claims, 3/3 automated tests passed, Ruff passed, and the repository diff check passed. |
| Outcome classification | **Supportive** — the paper's E01-targeted results can be decomposed into independent test targets with explicit estimands and claim-level reproduction criteria. This classification concerns ledger construction, not whether the scientific claims reproduce. |
| Caveats or blockers | The exact causal-emergence equations are missing from the supplied PDF extraction; the paper has no supplementary material or available code; 54/59 targets remain underdetermined pending method adjudication; source conflicts affect correlation wording, prediction endpoint, time-to-first units, min-condition consistency, and probability scope. |
| Lay summary | The paper's findings have been turned into a checklist of 59 questions that can be tested one at a time. The checklist preserves the numbers the paper reports and flags contradictions instead of averaging them into a single score. |
| Recommended next action | Hand control to the Chief Scientist. If authorized, execute **S02** to convert the 12 source discrepancies and all unresolved method choices into a versioned ambiguity ledger. Do not begin simulation or S02 implicitly. |

## Frozen question

Can every reported claim be represented as a separate, testable target with an explicit statistic, unit of analysis, expected direction, and replication criterion?

## Decision

Yes, at the claim-ledger level. Fifty-nine targets were registered. Each has a unique ID, family, evidence layer, paraphrased claim, primary source, reported statistic and target, expected direction, unit of analysis, sample scope, reproduction estimand, claim-specific criterion, inferential test, specification status, downstream step, and linked discrepancy notes.

This does not mean all targets are presently executable. Only one row is fully specified by the paper, four are testable with declared ambiguity, and 54 require S02 adjudication. The supportive outcome therefore means that S01's decomposition criterion was met while the underidentification was made explicit.

No composite pass score was created. Later `EXACT`, `DIRECTIONAL`, `NONREPLICATION`, or `UNDERDETERMINED` decisions must be made per claim.

## Inputs

### Governing documents

- `/workspace/AGENTS.md`
- `/workspace/FULL_PLAN.md`
- `/workspace/RESEARCH_PLAN.md`
- `/workspace/DATASETS.md`, `DATASET_CATALOG.json`, and `DATASET_AVAILABILITY.json` (confirmed that no dataset input is required)
- `/workspace/CAPABILITIES.md` and `/workspace/CAPABILITY_AVAILABILITY.json` (zero registered capabilities; no new software was needed)

### Uploaded paper materials

- Attachment manifest: `/workspace/input-attachments/MANIFEST.json`
- Required sidecar: `/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/_metadata/ATTACHMENT.md`
- Docling Markdown extraction of the 18-page paper: `/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md`
- Eight extracted figures: `/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/figures/figure-01.png` through `figure-08.png`

The original PDF was not materialized for the Researcher. The attachment sidecar directs use of the derived Markdown and figure files. The paper states that supplementary materials are absent and that code will be released upon publication; no author code accompanied the input.

## Methods

### Independent two-pass extraction

1. **Results/Methods pass:** Claims and numerical values were extracted from Results paragraphs, Materials and Methods, the Table 1 text, and the Discussion. Source locations use paper sections plus line numbers in the supplied extracted Markdown.
2. **Caption/figure/table pass:** Figure captions and all eight extracted figure images were inspected independently. Values visible only in figures were retained, including Figure 3's mean Spearman coefficient of 0.139 and Figure 6C's displayed coefficients/p-values of 0.041 (`p<0.001`), 0.008 (`p=0.4659`), and -0.03 (`p=0.0034`). All 12 Table 1 cells were transcribed as separate targets.
3. **Reconciliation:** The two passes were compared in 12 source-reconciliation records. Matches, partial matches, discrepancies, contradictions, and blocked method links were retained explicitly in `source_reconciliation.csv`.

### Claim decomposition rules

- Each named network or dynamical comparator is a separate metric-distinctiveness target.
- The 73 positive runs, 54 positive/significant runs, mean correlation, and one-sample significance are separate claims.
- Run-count, Mann-Whitney, and Fisher-combined state-comparison findings are separate claims.
- Raw temporal rejection count, median p-value, and differenced rejection count are separate claims.
- Phi-r prediction is compared separately against change in composition, raw composition, flux, and the class-prior baseline.
- Spike time, inter-spike distance, and spike height are separate targets.
- Each of the 3 treatments by 4 Table 1 outcomes is a separate absolute target.
- Absolute intervention values, pairwise contrasts, and generation trends are separate targets.
- Broad causal or predictive prose was mapped to explicit downstream estimands; it was not used as a substitute for the numerical targets.

### Schema and generation

The versioned schema is `E01-S01-v1.0.0`. Repository configuration records the source contract, 18 ledger columns, allowed enumerations, required claim families, numerical anchors, 12 discrepancy definitions, and validation rules. A deterministic Python builder expands the source targets, validates them, writes CSV and Markdown ledgers, writes the reconciliation and validation summaries, and hashes inputs/code/outputs into the artifact manifest.

Repository files:

- `/workspace/arrival-of-self-replicators/configs/e01/claim_schema.yaml`
- `/workspace/arrival-of-self-replicators/configs/e01/claim_targets.yaml`
- `/workspace/arrival-of-self-replicators/scripts/e01/build_claim_ledger.py`
- `/workspace/arrival-of-self-replicators/tests/e01/test_claim_ledger.py`

No simulation, statistical fitting, external repository retrieval, package installation, or S02 ambiguity-ledger construction was performed.

## Parameters and dependencies

| Item | Value |
| --- | --- |
| Experiment | E01 — Forensic Replication of the Paper |
| Research step | S01 — Create the E01 claim ledger |
| Ledger/schema version | `E01-S01-v1.0.0` / `1.0.0` |
| Extraction date | 2026-08-01 |
| Paper extraction | Docling; 18 pages; 8 figures |
| Python | 3.13.14 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| Ruff | 0.16.0 |
| CPU/GPU use | Serial metadata processing; no substantive CPU parallelism and no GPU use |
| New dependencies | None |

## Commands

Primary inspection and build commands were:

```bash
nl -ba input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md | sed -n '29,135p'
nl -ba input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md | sed -n '228,292p'
python scripts/e01/build_claim_ledger.py --artifacts-dir "$ARTIFACTS_DIR"
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/e01/test_claim_ledger.py
ruff check scripts/e01/build_claim_ledger.py tests/e01/test_claim_ledger.py
ruff format --check scripts/e01/build_claim_ledger.py tests/e01/test_claim_ledger.py
git diff --check
git push origin eidosoma/groups/42
```

The extracted figures were also opened at original detail and inspected visually.

## Results

### Claim inventory

| Claim family | Count | Principal downstream step |
| --- | ---: | --- |
| Metric distinctiveness | 12 | S15 |
| Aggregate trend | 1 | S14 |
| Spiking | 1 | S14 |
| Run-level association | 4 | S15 |
| Replicator-versus-drift state comparison | 3 | S15 |
| Temporal structure | 3 | S15 |
| Prediction | 6 | S16 |
| Spike timing/height | 3 | S16 |
| Table 1 absolute intervention cells | 12 | S17 |
| Intervention contrasts | 11 | S17 |
| Intervention time trends | 3 | S17 |
| **Total** | **59** | — |

### Anchor numerical targets preserved

The ledger retains, among other targets:

- aggregate trend `p=0.1995`;
- 73/100 positive runwise correlations;
- 54/100 positive/significant correlations, equivalently 54/73 among the positive runs;
- Figure 3 mean Spearman coefficient 0.139;
- 57/100 runs with higher mean Phi-r during replication;
- Fisher-combined `p<0.001`;
- 86/100 raw trajectories rejecting temporal independence, median Ljung-Box `p=2.07e-51`, and 100/100 after differencing;
- four separate Phi-r-versus-baseline prediction claims, each reported at `p<0.01`;
- spike-time `rho=0.66, p<0.001`, inter-spike-distance `rho=0.71, p<0.001`, and nonsignificant spike height;
- every Table 1 point and plus/minus value;
- Figure 6C displayed treatment coefficients and p-values.

### Reconciliation of 54/100 versus 54/73

The raw count is 54 runs. Figure 3 expresses that count over all 100 runs: 54/100, or 54%. Results conditions on the 73 positive-coefficient runs: 54/73, or about 74%. These statements are arithmetically compatible but answer different denominator questions. Claim `E01-C016` therefore requires later analyses to report both denominators and retain the raw count.

### Source discrepancies and escalation findings

| ID | Finding | Consequence |
| --- | --- | --- |
| D01 | 54/100 versus 54/73 denominator wording | Preserve both denominators; do not call them the same percentage. |
| D02 | Correlation uses a Phi-r trajectory in Results but “changes in Phi-r” in Figure 3 | Level and first-difference estimands must not be conflated. |
| D03 | Full future state-trajectory prediction is described as “initial appearance” | S16 must report paper-like and strict first-event tasks separately. |
| D04 | Time-to-first Table 1 cells contain `%`, while the note says molecular steps | Absolute reproduction is underdetermined pending unit adjudication. |
| D05 | Min consistency is said to worsen, but Table 1 gives 0.42 versus control 0.38 and defines higher as better | Source-specific targets conflict; no favorable direction may be selected post hoc. |
| D06 | Overall max/control probability is nonsignificant, while Discussion says max is higher | Separate overall occupancy from final-generation contrast. |
| D07 | “Most” runs spike above overall mean + 3 SD, but no count/scope is given and examples have large negative excursions | Exact spike prevalence is underdetermined. |
| D08 | Two causal-emergence equations are undecoded in the supplied extraction | Exact Phi-r method linkage is blocked until a source/formula is recovered. |
| D09 | Figure 6C uses rho-like notation next to regression lines | Preserve values but defer slope-versus-correlation identity. |
| D10 | Table 1 does not define the plus/minus dispersion | Do not assume SD, SE, or CI. |
| D11 | Scope of the reported Mann-Whitney `p<0.001` is unclear | Pooled versus runwise reconstruction must be adjudicated. |
| D12 | Spike height is nonsignificant but has no coefficient or p-value | Only significance-class matching is possible from the paper. |

The S01 escalation triggers were therefore encountered. They do not prevent completion of claim extraction, but they do prevent treating most targets as fully specified. They are handed off for explicit S02 adjudication.

## Validation

### Automated validation

`validation_summary.json` reports:

- `valid: true`;
- 59 claims and 59 unique IDs;
- 11/11 required claim families;
- 12/12 Table 1 absolute targets;
- 12 defined and claim-linked discrepancies;
- 12 reconciliation records explicitly covering 33 claims;
- 0 schema errors and 0 warnings;
- 1 fully specified target, 4 targets with declared ambiguity, and 54 targets underdetermined pending S02.

Focused tests validated schema completeness and enums, ID uniqueness, required numerical anchors, discrepancy linkage, 54/100-versus-54/73 retention, Table 1 cell count, deterministic output creation, CSV round-trip, Markdown claim coverage, validation JSON, and provenance manifest shape.

Test result:

```text
3 passed in 0.67s
All checks passed!
2 files already formatted
```

### Manual validation

- Results/Methods and captions/figures/Table 1 were read as separate passes.
- All eight extracted figures were viewed at original detail.
- Figure-only values were checked against their plotted legends.
- The generated Markdown ledger was spot-checked at the aggregate trend, correlation, prediction, Table 1, and intervention endpoints.
- Artifact paths, sizes, and hashes were checked after generation.

### Artifact hashes before adding this report to the manifest

| Artifact | SHA-256 |
| --- | --- |
| `claim_ledger.csv` | `e9cd4148dd47fa538fde9a683ce60d17bfc9c781a7cb0f7438e199d68801eafc` |
| `claim_ledger.md` | `cf65341a576133f0d7e523d069df9400d1fab44f6dc0fae597d4fcf6202dcfc7` |
| `source_reconciliation.csv` | `4027bd137643efd554004bd0e6980aca194af181c8ca96b98246bdc0a6e45d3d` |
| `validation_summary.json` | `a00de0b1704812eda12322292500c89e13d48fe8cef8491fc2cdd44c78f5a7` |

The final `artifact_manifest.json` is regenerated after this report is written and is the canonical source for report-inclusive hashes.

## Artifacts written

| Path | Role |
| --- | --- |
| `/artifacts/E01_forensic_replication_bundle/ledgers/claim_ledger.csv` | Canonical machine-readable 59-row claim ledger |
| `/artifacts/E01_forensic_replication_bundle/ledgers/claim_ledger.md` | Human-readable ledger with claim-level criteria and discrepancy registry |
| `/artifacts/research_steps/S01/source_reconciliation.csv` | Independent Results-versus-caption/table extraction reconciliation |
| `/artifacts/research_steps/S01/validation_summary.json` | Compact machine-readable schema/coverage validation |
| `/artifacts/research_steps/S01/artifact_manifest.json` | SHA-256 provenance for inputs, figures, code/config, and outputs |
| `/artifacts/research_steps/S01/research_step_full_results.md` | This required S01 handoff report |

`RESEARCH_PLAN.md` does not require an S01 `status.json`, so none was created.

## Provenance

- Repository: `/workspace/arrival-of-self-replicators`
- Branch: `eidosoma/groups/42`
- Commit: `a884967dbd2d0187cdd16df0a8d0f28e54fd4060`
- Push: successful to `origin/eidosoma/groups/42`
- Source attachment provenance: artifact and figure IDs are preserved in `/workspace/input-attachments/MANIFEST.json`.
- Exact input, figure, configuration, script, and output hashes are recorded in `/artifacts/research_steps/S01/artifact_manifest.json`.
- The generated ledger is deterministic for the versioned schema/targets; only the recorded repository commit changes when the code commit changes.

## Caveats, blockers, failed assumptions, and limitations

### Caveats and blockers

- The original PDF is unavailable in the Researcher mount; only Docling Markdown and extracted figures are present.
- Two central Phi-r formulas were not decoded. Equation-level identification of the information atom, redundancy function, and cut statistic is therefore blocked from this attachment alone.
- The paper supplies no supplementary material, code, raw data, estimator settings, multiple-testing policy, or several analysis details.
- Fifty-four of 59 claim rows are intentionally marked `underdetermined_pending_S02`; this is a finding, not a schema failure.
- Exact numeric tolerances in the claim criteria are preregistration proposals. S02 may version them only for a documented methodological reason, before reconstruction results are inspected.

### Failed assumptions exposed by S01

- The assumption that “54/73” and “54%” expressed the same proportion failed; they share a numerator but not a denominator.
- The assumption that Figure 3 and Results name the same correlation estimand failed because one says levels and the other says changes.
- The assumption that “initial appearance” describes the stated machine-learning label failed; the text predicts the entire remaining 75% state trajectory.
- The assumption that all Table 1 outcome units are internally consistent failed for time to first replicator.
- The assumption that min-Phi-r worsens all four Table 1 properties failed against the table's own consistency direction.

### Limitations

- S01 did not evaluate whether any paper claim is scientifically correct or reproducible.
- No values were digitized from plot geometry. Figure-only legend values were transcribed, while unprinted boxplot medians and dispersions were left unreported.
- No external paper version or author repository was searched because S01's frozen inputs were the uploaded paper and extracted figures; source retrieval/pinning belongs to later planned steps.
- The source uses association, prediction, and intervention language at different evidential strengths. The ledger keeps these layers separate and does not treat prediction as causation.
- Simulation evidence, when eventually generated, will remain a computational proxy and will not validate real prebiotic chemistry or biological agency.

## Recommended next action

Stop after S01 and return control to the Chief Scientist. If the workflow authorizes the queued next step, execute S02 as a fresh research step and map every unresolved choice plus D01–D12 into the ambiguity/discrepancy ledger. Highest-priority S02 resolutions are the exact Phi-r equation/atom, level-versus-change correlation, prediction label layout, time-to-first unit, Table 1 dispersion, consistency contradiction, intervention probability scope, and Figure 6C coefficient identity.
