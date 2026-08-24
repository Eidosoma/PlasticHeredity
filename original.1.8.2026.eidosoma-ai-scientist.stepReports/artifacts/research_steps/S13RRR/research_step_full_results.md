# S13RRR Full Results: Eligibility-Aware Replay Finalization

## Top summary

- **Research step ID:** `E01-S13RRR-ELIGIBILITY-AWARE-REPLAY-FINALIZATION-v1.0.0` (S13RRR).
- **Completion status:** `COMPLETED_AT_MANDATORY_S13RRR_HUMAN_REVIEW_BOUNDARY`.
- **Artifacts written:** 54 status-bearing files under `/artifacts/research_steps/S13RRR/`, including the 200-task availability ledger, 48-slot non-applicability ledger, 3,552 executed comparisons, complete twice-run statistics, adjudication, validation, provenance, status, and this report.
- **Validation result:** `PASS`; 3552/3552 executable suffix sentinels passed, 48 frozen slots were exactly unavailable, and the unchanged source/replay, reporting, schema, twice-run statistics, provenance, and immutability gates passed.
- **Outcome classification:** `HELD_OUT_TWO_CANDIDATE_SCALEUP_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE` (constraining/contradictory).
- **Caveats or blockers:** This is an explicitly post-outcome third override of two earlier permanent-stop decisions. It changes the replay cardinality gate after endpoint availability was known, so the result is substantially less confirmatory than a clean preregistered analysis. Full-trajectory fits remain retrospective/future-dependent and this cannot identify the unpublished author implementation.
- **Lay summary:** Every replay comparison that could exist in the frozen data was exact. Running the original held-out analysis twice then gave the same answer both times; the two confirmed simulator candidates did not jointly pass the required retrospective and prospective evidence gates.
- **Recommended next action:** Mandatory human review. Keep all later work blocked; do not authorize another repair or automatic continuation.

## Frozen question

Can the exact 48 unavailable suffix slots be treated as not applicable—without changing any other gate—and thereby allow a deterministic answer to the frozen two-candidate held-out S13 question?

## Inputs and provenance

Only the 200 frozen S13 task bundles and the 1,600 value-preserving S13RR canonical views were read. Candidate 1 was excluded. No simulation, source fit, task omission, subset, resampling-seed change, candidate change, or downstream step occurred. S13, S13R, and S13RR remain byte-for-byte immutable with their failure classifications unchanged.

## Detailed methods

Before statistics, the pushed method reconstructed nominal first/middle/last suffix slots from each implementation's frozen eligible endpoint generations. Duplicate nominal generations were treated according to the already frozen source precedence; zero eligible endpoints generated no executable comparison. The exact ledger required 3,552 executable and 48 not-applicable slots. Candidate 2 matrix 68 contributed six executable comparisons and twelve duplicate nominal slots; candidate 2 matrix 72 and candidate 3 matrix 72 each contributed eighteen unavailable slots. Every other source gate remained unchanged.

The only table-interface operations reordered existing fields in `prefix_endpoint_values.parquet` and `ensemble_adjudication.csv`. Field sets, rows, values, null masks, and keys were unchanged. The source statistics were then executed twice with the original 4,096-replicate bootstrap, circular-shift, and block-aware seeds and the original two-candidate unanimity rule.

## Commands

```bash
PYTHONPATH=src python -m pytest -q tests/e01/test_s13rrr_eligibility_aware_replay.py tests/e01/test_s13rr_downstream_schema_canonicalization.py tests/e01/test_s13_confirmed_timebase_scaleup.py
python -m ruff check src/e01_s13rrr_eligibility_aware_replay scripts/e01/freeze_s13rrr_preregistration.py scripts/e01/run_s13rrr_eligibility_aware_replay_finalization.py tests/e01/test_s13rrr_eligibility_aware_replay.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s13rrr_preregistration.py --record-commit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13rrr_eligibility_aware_replay_finalization.py
```

## Results

The scientific adjudication was `HELD_OUT_TWO_CANDIDATE_SCALEUP_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE`. Candidate-specific results remained primary:

- `S12F-CANDIDATE-02`: retrospective IIGR median rho -0.00615106 (45/96 positive; 95% trajectory-bootstrap [-0.0200496, 0.00545063]; association gate `False`); replicator-minus-drift median mean difference 2.3721e-05 (48 positive; gate `False`); prospective median rho -0.00299977 (41/90 positive; 95% interval [-0.0359245, 0.0318142]; circular-shift p=0.575543; gate `False`); combined gate `False`; classification `CANDIDATE_NOT_SUPPORTED`.
- `S12F-CANDIDATE-03`: retrospective IIGR median rho -0.008407 (41/95 positive; 95% trajectory-bootstrap [-0.0127349, 0.00125201]; association gate `False`); replicator-minus-drift median mean difference 8.97742e-05 (49 positive; gate `False`); prospective median rho -0.0308806 (42/89 positive; 95% interval [-0.0624167, 0.0373812]; circular-shift p=0.970954; gate `False`); combined gate `False`; classification `CANDIDATE_NOT_SUPPORTED`.

This result is labeled exactly as a third-override, post-outcome eligibility-exception analysis. It does not retroactively validate S13, S13R, or S13RR and cannot support fixed-window, early-warning, prediction, intervention, causal-control, or author-identity claims.

## Validation

- Availability: 200/200 task ledgers passed; 3,552/3,552 applicable identities existed exactly once and passed structural/result replay; 48/48 unavailable identities were confined to the three declared tasks.
- Source gates: full and eligible-prefix replay, structural suffix, >=0.80 finite coverage, zero worker failures, and <=1e-12 component identity error all passed (`True`).
- Complete execution validation: `True`. Both statistics executions were bit-exact at the serialized DataFrame level.
- Source bundles, canonical views, and all earlier artifact files passed pre- and post-run SHA-256 validation.

## Runtime and storage

Wall time was 963.750 seconds and process CPU time was 963.775 seconds. No simulation, source-fit worker, or GPU was used. S13RRR retained 19336108 artifact bytes and 0 disposable derived-cache bytes.

## Caveats, blockers, and limitations

- The 3,552 rule was authorized after the three low-availability tasks were known, and two reporting-order corrections were also post-outcome. This is not clean confirmation.
- Repeated waivers and repairs materially reduce procedural credibility even though this operation was value-preserving and exactly replayed.
- The public-source information pipeline remains source-informed only; the paper's unpublished GARD and fixed-window implementation are unavailable.
- Retrospective full-trajectory local values use future observations. Prospective prefix results start only after 256 locked-clock transitions.
- No additional repair or downstream continuation is authorized.

## Artifact provenance

The pushed method commit, complete immutable-prior baseline, 2,000 source-cache hashes, 1,600 canonical-view hashes, exact task/slot ledgers, input and reporting audits, twice-run result hashes, schema checks, status, runtime/storage records, and final artifact manifest provide the audit chain. Bulky immutable inputs remain in their existing cache roots.

## Recommended next action

Return for mandatory human review and begin no later work automatically.
