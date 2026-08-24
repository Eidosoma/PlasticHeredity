# S13R Full Results: Schema Normalization Confirmation

## Top summary

- **Research step ID:** `E01-S13R-SCHEMA-NORMALIZATION-CONFIRMATION-v1.0.0` (S13R).
- **Completion status:** `PERMANENTLY_STOPPED_UNDER_ONE_REPAIR_RULE`; no candidate statistic or two-candidate adjudication was computed.
- **Artifacts written:** 48 status-bearing artifacts under `/artifacts/research_steps/S13R/`, including the preregistration/method lock, 200-task and field-level adapter audits, normalized-view hashes, downstream schema diagnostics, suppressed scientific schemas, validation/provenance/failure records, and this report.
- **Validation result:** The authorized label adapter passed 200/200 task views, but the frozen downstream aggregation contract exposed another incompatible schema and stopped: `ADDITIONAL_UNAUTHORIZED_SOURCE_TABLE_SCHEMA_NORMALIZATION_REQUIRED:prefix.parquet,suffix.parquet,seeds.parquet`.
- **Outcome classification:** `S13R_REPAIR_PATH_PERMANENTLY_STOPPED` (constraining/contradictory operational evidence); held-out scientific association remains `NOT_EVALUATED`.
- **Caveats or blockers:** Incompatible downstream tables: prefix.parquet, suffix.parquet, seeds.parquet. The authorization permits no second schema adapter, data-loading workaround, subset analysis, source rerun, or method change. S13 remains byte-for-byte unchanged and classified `S13_VALIDATION_FAILED_CLOSED`.
- **Lay summary:** The five label files were safely normalized without changing a row, value, or null. The next untouched table family then revealed a separate all-ineligible-task physical-schema mismatch. Because the human authorized exactly one narrow repair, the analysis stopped before correlations rather than silently adding a second repair.
- **Recommended next action:** Close this repair path and return for mandatory human review. Keep S14–S18, prediction, interventions, E02, report-bundle progression, and further scale-up blocked.

## Frozen question

S13R asked whether the five all-null label tables could be represented with the 195-task canonical physical schema and, only after every frozen gate passed, whether the original S13 statistics could adjudicate the held-out two-candidate result. The first question passed; the conditional scientific question was not reached.

## Inputs

Only the 200 frozen per-task bundles under `/cache/e01_s13/source_results/` and their S13 manifests were read. The partial S13 concatenation was not read. No trajectory was generated, no source fit was rerun, no candidate was added or removed, and candidate 1 remained excluded.

## Detailed methods

Before any candidate statistic, the exact three-field adapter, all validations, the inherited S13 statistics entry points, and the one-repair stop rule were committed and pushed. For each label task, Arrow `null` was changed only to `string` for `clusterId` and `referenceObservationId`, and only to `double` for `metricToReference`. Logical value hashes, null-mask hashes, row/key order, task identity, column order, non-adapter physical-field hashes, and original source-file hashes were checked. The 195 canonical inputs were required to remain value-identical; exactly the five preregistered tasks were allowed a physical type change.

After the adapter passed, S13R performed a read-only schema compatibility audit for the exact frozen strict concatenation over all 200 tasks. It did not coerce, omit, or normalize another table. Any second schema variant was a terminal condition before scientific access.

## Commands

```bash
PYTHONPATH=src python -m pytest -q tests/e01/test_s13r_schema_normalization_confirmation.py tests/e01/test_s13_confirmed_timebase_scaleup.py
python -m ruff check src/e01_s13r_schema_normalization scripts/e01/freeze_s13r_preregistration.py scripts/e01/run_s13r_schema_normalization_confirmation.py tests/e01/test_s13r_schema_normalization_confirmation.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s13r_preregistration.py --record-commit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13r_schema_normalization_confirmation.py
```

## Results

- Adapter task views: 200/200 passed.
- Canonical typed tasks: 195; physically adapted tasks: 5.
- Field checks: 3000/3000 passed.
- All normalized label views share one schema: True.
- Downstream strict schema-compatible tables: 5/8.
- Newly exposed incompatible tables: `prefix.parquet, suffix.parquet, seeds.parquet`.
- Association, drift, temporal, spike, metric-identity, future-dependence, resampling, paired comparison, statistics replay, and two-candidate adjudication: `NOT_REACHED_ONE_REPAIR_RULE`.

## Validation

All prior artifact hashes and all 2,000 S13 cache-file hashes were checked before adaptation. Every original source file remained unchanged. S13's failure classification and lack of scientific adjudication were retained. The adapter and derived-view hashes are complete. Scientific result artifacts are schema-bearing and empty because the downstream gate fired.

## Runtime and storage

S13R used 13.866 wall-seconds and 13.938 orchestration CPU-seconds, no simulation/source worker and no GPU time. Retained derived-cache bytes: 2748452. Retained artifact bytes before the final manifest: 1215062.

## Caveats and limitations

- This is a post-failure human override and further weakens the confirmatory standing of the branch.
- The successful label normalization does not validate any source-emergence association.
- The additional mismatch occurs in task bundles with no eligible prefix endpoint; treating those rows or empty suffix records specially would itself be another schema/data-interface decision and was not authorized.
- Public-source behavior is not the unavailable author implementation; retrospective full fits remain future-dependent, and fixed-window/early-time claims remain unresolved.

## Provenance

The pushed method lock, complete S01–S13 artifact baseline, 2,000-file S13 cache manifest, per-field logical/physical hashes, normalized-view manifest, source/postcheck hashes, scope ledger, runtime/storage records, failure ledger, and artifact manifest preserve the entire decision path. Derived views live under `/cache/e01_s13r/`; compact audit evidence is under `/artifacts/research_steps/S13R/`.

## Recommended next action

Mandatory human review. Under the one-repair rule, no additional S13/S13R schema adapter or analysis repair is authorized. Do not begin later work automatically.
