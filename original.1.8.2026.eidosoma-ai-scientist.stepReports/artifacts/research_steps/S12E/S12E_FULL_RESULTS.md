# S12E full results — Paper-pipeline detective reconstruction

## Top summary

- **Research step ID:** `E01-S12E-PAPER-PIPELINE-DETECTIVE-RECONSTRUCTION-v1.0.0`.
- **Completion status:** COMPLETED_FAIL_CLOSED_AFTER_PHASE1; stopped after `Phase 1` under the preregistered firewall.
- **Artifacts written:** all required status-bearing tables, ledgers, registries, seed/provenance records, nine figures, validation/manifests, a documented schema-only reporting amendment, and equivalent named/canonical reports under `/artifacts/research_steps/S12E/`.
- **Validation result:** Phase sequencing, hashes, exact replay, seed firewall, immutability, scope, runtime, and storage checks were run; terminal scientific gate classification is TIME_BASE_MISMATCH_CONFIRMED.
- **Outcome classification:** `TIME_BASE_MISMATCH_CONFIRMED` (constraining/contradictory).
- **Caveats or blockers:** None of the five source-grounded engine candidates simultaneously met the frozen 500–1,500-step paper-time interval and all completion, extinction, growth/fission, and replay gates.
- **Lay summary:** The audit first asked whether the paper-described simulation produces the same basic clock and replication behavior before looking at causal-emergence values. The first failed dependency layer is reported explicitly; downstream analyses were stopped rather than tuned to compensate.
- **Recommended next action:** Return for human review with S13 blocked. The first failed layer is the paper time base/GARD dynamics; do not tune a sixth engine inside S12E.

## Frozen question and evidentiary boundary

S12E asked whether one source-grounded dependency chain could explain the paper's molecular-step scale, 100 growth–fission generations, control replication fingerprints, punctuated source-defined causal emergence, positive emergence–replication association, and max/control/min direction. It is a `SOURCE_AND_PAPER_INFORMED_FORENSIC_RECONSTRUCTION`, not author-code identity. No method was eligible for selection merely because it produced a favorable emergence association. S13 remained `BLOCKED_PENDING_S12E_HUMAN_REVIEW` throughout.

## Lay summary

The first failed dependency layer is reported explicitly; downstream analyses were stopped rather than tuned to compensate. Later phases were never used to compensate for an upstream mismatch. This preserves the difference between a forensic reconstruction and an exact replication of unavailable code.

## Inputs and provenance

- Governing plans, the original arXiv v1 paper, its PDF-only source endpoint response, figure rasters, the S01–S12D evidence chain, and the S12B safe lattice were refreshed before execution.
- Public snapshots were pinned to historical GARD `86dff6320d5ae91b4e831471079ff46749b14df9`, IIGR `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`, PhiRL `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`, and BreakingGRNMemories `afe44231ad3ce915172cdb53a6b234bd76fcb6a5`.
- The arXiv source response was PDF-only (SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`); no TeX comments or original filenames were exposed.
- No dataset mount or upstream previous-artifact mount was present. No authors were contacted.
- Prior S01–S12D evidence was hashed before outcomes and checked again afterward: `PASS (326 files; 0 changed/missing)`.

## Detailed methods

### Phase 0 — archaeology and method lock

The paper fingerprint, implementation ambiguity, source clue, source snapshot, and figure-measurement ledgers were frozen before development simulation. Exactly five engine candidates, four labels, four metric branches, and three intervention semantics were preregistered. Development, confirmation, and intervention used three disjoint 256-bit seed roots. The design was committed and pushed before development outcomes were opened.

### Phase 1 — paper-prose GARD time base

For Poisson candidates, each batch update drew all joins and losses simultaneously from the frozen rates, clipped losses at the current count, retained overshoot, and used complementary binomial fission. K0 isolated the historical categorical event kernel while sharing the paper-style distinct initialization, max-step boundary, and binomial fission. The same 24 catalytic matrices and initial states were used across candidates, while dynamics streams were engine-specific. Each trajectory was independently regenerated exactly.

| engineId                         |   completed100Fissions |   extinctionCount |   medianTotalBatchSteps |   medianPostFissionMass |   meanOvershoot |   fractionGenerationsReachingNMax | exactReplayPassed   | phase1Eligible   | selectedForPhase2   |
|:---------------------------------|-----------------------:|------------------:|------------------------:|------------------------:|----------------:|----------------------------------:|:--------------------|:-----------------|:--------------------|
| K1_PAPER_POISSON_RANDOM_NONEMPTY |                     24 |                 0 |                   440.5 |                   43.25 |         16.1229 |                                 1 | True                | False            | False               |
| K3_PAPER_POISSON_RANDOM_LITERAL  |                     24 |                 0 |                   443.5 |                   44    |         16.2521 |                                 1 | True                | False            | False               |
| K2_PAPER_POISSON_FIRST_DAUGHTER  |                     24 |                 0 |                   434.5 |                   44    |         17.0229 |                                 1 | True                | False            | False               |
| K4_PAPER_POISSON_RHO_ONE         |                     24 |                 0 |                    23   |                  189.75 |       1496.09   |                                 1 | True                | False            | False               |
| K0_HISTORICAL_EVENTWISE          |                     24 |                 0 |                  5095.5 |                   40    |          0      |                                 1 | True                | False            | False               |

### Phase 2 — replicator labels

Emergence was prohibited until the engine/time-base gate passed. The four frozen labels operated on post-fission relative compositions and were compared to occupancy, persistence, consecutive-label consistency, and onset fingerprints. Up to two development pipelines could be locked for an untouched 24-matrix confirmation.

Phase 2 was not reached. `label_development_results.parquet`, `label_fingerprint_summary.csv`, `confirmation_trajectories.parquet`, `confirmation_labels.parquet`, and `confirmation_pipeline_results.csv` each retain one explicit placeholder with `status=NOT_REACHED` and `reason=phase1_time_base_gate_failed`; zero label or confirmation outcome rows were generated.

### Phase 3 — causal emergence and past-only audit

Phase 3 was conditional on a confirmed engine–label pipeline. The four frozen branches kept `corr(E_t,Y_t)` separate from the Figure-3-caption `corr(delta E_t,Y_t)` diagnostic. Full values were retrospective. Prefix values, when authorized, were complete past-only refits at eligible post-fission endpoints.

### Phase 4 — intervention-semantics pilot

Phase 4 was conditional on a confirmed retrospective Phase-3 candidate. The three literal scoring semantics remained distinct, with no no-op in the max/min search. No intervention finding could automatically establish causality.

## Results and first failed layer

The first terminal layer was **Phase 1**. Classification: `TIME_BASE_MISMATCH_CONFIRMED`.

Failure/status ledger:

| failureId   | phase   | severity      | status        | reason                                                                                             | consequence                                                     |
|:------------|:--------|:--------------|:--------------|:---------------------------------------------------------------------------------------------------|:----------------------------------------------------------------|
| S12E-F001   | Phase 1 | TERMINAL_GATE | FAILED_CLOSED | No frozen engine candidate met all predeclared Phase-1 time-base/growth/fission eligibility gates. | No labels, emergence, prefixes, or interventions were computed. |

## Commands and dependencies

Design freeze and validation:

```bash
python scripts/e01/freeze_s12e_preregistration.py
python -m pytest -q tests/e01/test_s12e_paper_pipeline_detective.py
git commit ... && git push origin eidosoma/groups/42
python scripts/e01/freeze_s12e_preregistration.py --record-commit
```

Execution:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/e01/run_s12e_paper_pipeline_detective.py --require-pushed-preregistration
```

Python 3.13 float64 used NumPy, SciPy, pandas/PyArrow, scikit-learn, statsmodels, NetworkX, Matplotlib, and the already confirmed local source wrappers. Six process workers were used for simulation; numerical-library threads were pinned to one. Exact dependency versions and environment details are recorded in `runtime_manifest.json` and the prior S03 environment lock.

## Validation

- Preregistration/source/registry checks: `PASS: validated and pushed before development access`.
- Exact trajectory replay: `PASS`.
- Cross-phase seed-material intersection: `PASS`.
- Confirmation firewall: `PASS: phase sequencing honored; conditional outputs not opened early`.
- Prior immutability: `PASS (326 files; 0 changed/missing)`.
- Required artifact and schema checks: `all required status-bearing paths are created and hash-checked`.
- Post-outcome reporting amendment: `PASS`; 15 downstream placeholder schemas were normalized to explicit `NOT_REACHED`/`phase1_time_base_gate_failed` fields without changing code, simulations, numeric outcomes, gates, scope, or classification.
- Runtime/storage ceiling: `PASS`.
- Report equivalence is checked after both reports are written.

## Caveats, blockers, and interpretation

None of the five source-grounded engine candidates simultaneously met the frozen 500–1,500-step paper-time interval and all completion, extinction, growth/fission, and replay gates. Full-trajectory source values, if any, are retrospective and cannot support early warning or online causal control. Public source lineage is not the unavailable GARD implementation. Negative, missing, and stopped branches remain status-bearing and were not replaced. S12, S12C, and S12D classifications remain unchanged.

## Artifact provenance

The run began at `2026-08-02T22:58:33.445871+00:00`. `artifact_manifest.json` records SHA-256 and size for every retained output except itself, `source_snapshot_manifest.json` records source commits/blobs/files, and `regeneration_validation.json` records immutability, seed, replay, scope, and completeness gates. `postoutcome_reporting_amendment.json` documents the schema-only normalization discovered during the final completeness audit. Large disposable trajectory caches remained under `/cache/e01_s12e/` and were not promoted into artifacts.

## Recommended next action

Return for human review with S13 blocked. The first failed layer is the paper time base/GARD dynamics; do not tune a sixth engine inside S12E. Do not begin S13, E02, E03, intervention scale-up, or another estimator repair automatically.
