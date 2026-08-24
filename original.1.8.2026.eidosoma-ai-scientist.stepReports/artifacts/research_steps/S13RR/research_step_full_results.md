# S13RR Full Results: Downstream Schema Canonicalization

## Top summary

- **Research step ID:** `E01-S13RR-DOWNSTREAM-SCHEMA-CANONICALIZATION-v1.0.0` (S13RR).
- **Completion status:** `PERMANENTLY_STOPPED_NO_FURTHER_REPAIR`.
- **Artifacts written:** 48 status-bearing artifacts under `/artifacts/research_steps/S13RR/`, including the frozen contract, 1,600-view task audit, field audit, eight-family schemas, replay gate, complete suppression or scientific tables, provenance, manifests, and this report.
- **Validation result:** canonical derived views 1600/1600; strict family schemas 8/8; frozen source/replay gate `False`.
- **Outcome classification:** `S13RR_REPAIR_PATH_PERMANENTLY_STOPPED`; scientific held-out status `NOT_EVALUATED`.
- **Caveats or blockers:** The unchanged executed-suffix gate failed (3,552 observed versus 3,600 required). A final ordered-schema audit also retained mismatches for `prefix_endpoint_values.parquet` and the empty `ensemble_adjudication.csv`; neither was repaired.
- **Lay summary:** The final override can standardize only how known all-null and empty files are represented; it cannot change a simulation or scientific value. The analysis proceeds only if the original replay contract still passes exactly.
- **Recommended next action:** Mandatory human review. Keep S14–S18, prediction, interventions, E02, report-bundle progression, and further scale-up blocked.

## Frozen question

Can exact, value-preserving views remove the diagnosed physical-schema variants and allow the unchanged held-out S13 analysis to run twice identically?

## Inputs

Only the 200 frozen S13 task bundles (2,000 hash-locked files) and prior S13/S13R evidence were used. No trajectory, source fit, partial concatenation, subset, candidate 1 result, or S12G scientific cache was used.

## Detailed methods

The pushed adapter reproduced the exact S13R label typing, typed seven all-null prefix fields and one all-null seed endpoint field in the two matrix-72 tasks, and assigned the canonical schema to their two 0-row/0-column suffix tables. It then required zero invented/omitted rows, unchanged source hashes, logical values, null masks, order, keys, task identities, and physical identity for every non-adapter field. All eight families were strictly concatenated without promotion. The original S13 source/replay gate—including its exact executed-suffix cardinality—was evaluated before any statistic. Conditional statistics were the unmodified twice-run S13 procedures with original 4,096-replicate seeds and gates.

## Commands

```bash
PYTHONPATH=src python -m pytest -q tests/e01/test_s13rr_downstream_schema_canonicalization.py tests/e01/test_s13r_schema_normalization_confirmation.py tests/e01/test_s13_confirmed_timebase_scaleup.py
python -m ruff check src/e01_s13rr_downstream_schema_canonicalization scripts/e01/freeze_s13rr_preregistration.py scripts/e01/run_s13rr_downstream_schema_canonicalization.py tests/e01/test_s13rr_downstream_schema_canonicalization.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s13rr_preregistration.py --record-commit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13rr_downstream_schema_canonicalization.py
```

## Results

- Canonical views: 1600/1600 tasks and 27000/27000 fields passed.
- Invented rows: 0; omitted rows: 0; source files mutated: False.
- Strict table-family schemas: 8/8; strict concatenations: 8/8.
- Frozen executed-suffix count: observed 3,552 versus required 3,600. Candidate 2 matrix 68 contributed only six executed sentinels because it had two eligible endpoints; candidate 2 and candidate 3 matrix 72 contributed none because they had zero eligible endpoints. All other 197 tasks contributed the frozen 18.
- Candidate statistics and two-candidate adjudication executed: False.
- Final output-schema audit: 16/18 artifacts had the declared ordered columns. `prefix_endpoint_values.parquet` retained the source-canonical position of the three label columns rather than the separate S12G reporting order, and the schema-bearing empty `ensemble_adjudication.csv` retained the frozen S13R result columns rather than the older S12G declaration. No reorder or reporting adapter was applied after the stop.

## Validation

The method was committed and pushed before candidate statistics. All prior artifacts and S13 cache inputs were hash-checked before and after view construction. The original S13 and S13R classifications remain unchanged. The scientific result tables are schema-bearing and empty because statistics were suppressed. The separate final ordered-schema audit failed 2/18 rows and is retained as additional fail-closed evidence, not silently corrected.

## Runtime and storage

Wall time was 202.103 seconds and process CPU time 202.884 seconds; no simulator, source worker, or GPU was used. Derived cache bytes: 55094456.

## Caveats and limitations

- This second override follows a prior promise of permanent termination and substantially weakens confirmatory credibility.
- Typed nulls and schemas do not create scientific observations or establish source validity.
- The replay-cardinality shortfall is not limited to the two schema-affected matrix-72 tasks: candidate 2 matrix 68 has only two eligible prefix endpoints. Relaxing the exact 3,600 gate or defining a low-eligibility exception would be a new replay rule.
- Two final ordered reporting schemas also disagree with the existing declaration; changing either after the terminal gate would be another schema operation.
- Full fits remain retrospective and future-dependent; pre-256, fixed-window, prediction, intervention, and author-identity questions remain unresolved.
- No further repair is authorized.

## Provenance

The pushed commit, complete prior/cache baselines, task/field audits, source and view hashes, strict collation schemas, replay gate, scope ledger, runtime/storage records, status, and artifact manifest preserve the full chain. Bulky derived views remain under `/cache/e01_s13rr/`.

## Recommended next action

Return for mandatory human review and begin no later step automatically.
