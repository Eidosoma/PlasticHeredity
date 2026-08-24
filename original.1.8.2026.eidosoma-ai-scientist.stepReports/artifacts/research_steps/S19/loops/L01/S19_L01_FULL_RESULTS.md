# E01/S19 Loop 1 — Unevaluated-claim recovery (failed closed)

## Concise top summary

- **Research step ID:** S19-L01
- **Completion status:** `LOOP_FAILED_CLOSED`; mandatory human-review boundary active
- **Artifacts written:** every required S19 root/L01 status, ledger, empty-not-eligible result table, additive claim overlay, validation, runtime, and artifact manifest; no invalidated scientific value is reported
- **Validation result:** `FAIL_CLOSED_S16_25_75_EXACT_REPLAY`; immutable S01–S18/V1/V2 baseline passed across 1432 files
- **Outcome classification:** `LOOP_FAILED_CLOSED` for all sixteen additive claims; zero S20 promotions
- **Caveats or blockers:** Bundle A and B computations were invalidated before serialization; Bundle C never started; likely cutoff-source seed transcription issue remains unverified because replay failure is a global stop
- **Recommended next action:** human review should consider `CONTINUE_S19` with the narrow replicator-definition/temporal-fingerprint 88%-versus-98% theme; no next loop is active

## Lay summary

Loop 1 did not produce an eligible scientific result. The first attempt stopped because a helper did not reproduce the frozen first H value. A separately recorded amendment restored exact H across all 200 trajectories without changing the declared method. The restarted attempt then failed the decisive check that its 25/75 prediction results exactly reproduce S16. The protocol requires stopping at that point. Rather than expose unvalidated in-memory numbers or patch toward a preferred result, this report preserves the failures and returns control.

The strongest next scientific question is independent of this failure: why the frozen adjacent-similarity label yields about 98% occupancy when the paper reports about 88% and a much later onset. The proposed next loop would reconstruct the replicator state and its full temporal fingerprint, not tune a threshold to match occupancy.

## Frozen question and intended scope

L01 was prospectively locked to evaluate C001–C012, C029, and C031–C033 through three bundles: same-lineage network/dynamical metrics; five prediction proportions under the exact S16 model contract; and two bounded spike definitions. Both simulator candidates were mandatory, pooling secondary, with no new GARD trajectories.

## Inputs

- Frozen S13Y trajectories, completed-fit and past-only PhiRL values, and `Y=I(H>0.9)` labels.
- Frozen S16 matrix splits, seeds, tensor/model rules, and 25/75 results as the replay oracle.
- Original paper, S18 claim matrix, and public same-author source lineages.
- The immutable baseline validated 1432 files (969751349 bytes) with aggregate SHA-256 `34a4e43e437a98e42874698961df77561d157e8292f99d2fe0f7e73044336fd3`.

## Methods and commands

The preregistration, method lock, candidate ranking, seed manifest, input manifest, and source snapshot were committed and pushed before claim-level outcome access. The locked scientific command was:

```text
PYTHONPATH=src:/cache/e01_s19_l01/python_deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s19_l01.py --workers 8
```

Focused pre-outcome and amendment validation used 13 passing S16/S19 tests. The amendment audit independently replayed all 200 H vectors with maximum absolute error exactly zero and verified `Y=I(H>0.9)` with zero failures.

## Execution chronology

### Attempt 1

Bundle A ran in memory. Bundle B stopped while loading frozen inputs because its independent H helper set the initial value to 1 rather than duplicating the first adjacent similarity. No Bundle B outcome was calculated. Amendment `S19-L01-VPA-001` changed no declared method: it restored the exact S13Y/S16 convention and required use of serialized frozen H.

### Attempt 2

The amended commit was tested, pushed, and matched the clean remote before restart. Bundle A and the five-proportion Bundle B calculation ran in memory. The 25/75 comparison then found at least one metric unequal to frozen S16 and triggered the global exact-replay stop. No result table had been serialized or inspected, and Bundle C had not begun.

A static post-stop audit identifies a likely implementation cause: the new cutoff-source task appears to use a different seed domain token and width than S16's frozen `cutoff_source` 32-bit seed helper. This is a diagnosis, not a verified repair. Testing or repairing it would require another post-failure scientific run and is therefore deferred to human review.

## Results

There are no eligible L01 scientific estimates. Required result tables are present with explicit empty schemas and status ledgers; they are not missing and do not contain fabricated values. The additive overlay retains S18's original `NOT_EVALUATED` field and marks all sixteen S19 entries `LOOP_FAILED_CLOSED`.

| Additive claim family | Claims | L01 status | S20 promotion |
|---|---:|---|---|
| Network/dynamical distinctiveness | C001–C012 | LOOP_FAILED_CLOSED | No |
| Alternative prediction proportions | C029 | LOOP_FAILED_CLOSED | No |
| Spike timing/spacing/height | C031–C033 | LOOP_FAILED_CLOSED | No |

No retrospective, prospective, causal, exact, or directional paper match is inferred from the invalidated computations.

## Validation

- Immutable prior: PASS (1432 files; zero mismatches).
- Pre-outcome clean pushed lock: PASS at commit `3950b84060b4fc45f6108126f67c2973625c78c0`.
- Value-preserving H amendment: PASS at commit `ebeeb528cb8b3c95804462635c730b305485a10e`.
- Exact 25/75 S16 replay: **FAIL**.
- Loop-level scientific eligibility: **FAIL CLOSED**.
- New GARD trajectories: zero.
- S18 artifacts/status totals: unchanged.

## Self-improvement analysis

- **Belief before:** public lineages could resolve metric and spike ambiguities while frozen S16 could anchor proportion sensitivity.
- **What the loop attempted:** complete sixteen unevaluated claims without method search.
- **What was learned:** the execution harness itself was not yet a trustworthy extension of S16. Exact-H identity was repaired, but prediction replay still failed. No claim result survives that failure.
- **Hypotheses weakened:** confidence that the new L01 prediction extension exactly instantiates S16.
- **What remains plausible:** the paper's replicator state may differ structurally from adjacent `H>0.9`; this is motivated by existing S18 evidence, not by invalid L01 values.
- **Why a next loop could add information:** a label-focused loop attacks one upstream dependency shared by Figures 3–6 and Table 1, using fingerprints independent of emergence. It is not an opportunity to add thresholds until a positive result appears.

## Proposed inactive next-loop theme: replicator definition and temporal fingerprint

The human-proposed next loop should compare a small, source-grounded family:

1. adjacent `H>0.9` as the frozen comparator;
2. dominant recurring-composition/centroid membership;
3. recurring Euclidean composition-cluster membership as described in the paper;
4. historical source-traceable GARD compotype/non-drift machinery.

Labels should be ranked without emergence using the joint fingerprint: occupancy near 88%, persistence, time to first onset, consistency, entry/exit counts, episode duration/structure, actual cluster recurrence, and candidate-2/candidate-3 agreement. The already observed `H>0.97` occupancy resemblance is not an eligible solution because it was outcome-guided and did not reproduce onset or consistency. Any chosen fixed definition would require untouched S20 confirmation. This theme is proposed only; it is not authorized or executed.

## Caveats and provenance

The two attempts consumed several hours but no authoritative child-process CPU total survived abort. Both remained structurally below the 100-CPU-hour ceiling, used eight one-thread CPU workers, no GPU, and no trajectory generation. Failure messages, commits, source identities, hashes, seeds, failure statuses, and the reporting-only finalizer are retained. Public source without a compatible license remains cache-only.

## Mandatory human-review boundary

Choose exactly one: `CONTINUE_S19`, `ACTIVATE_S20_CONFIRMATION`, `ACTIVATE_S20_CLOSEOUT_ONLY`, or `PAUSE_PROGRAM`. Given the zero eligible L01 leads and the human's stated priority, the scientific recommendation is `CONTINUE_S19` only if the reviewer explicitly authorizes the narrow replicator-definition/temporal-fingerprint loop and a fresh compute ceiling. No L02, S20, E02, or report bundle has started.
