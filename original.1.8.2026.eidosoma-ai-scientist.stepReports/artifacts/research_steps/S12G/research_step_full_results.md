# S12G Full Results: Frozen Time-Base Ensemble

## Top summary

- **Research step ID:** `E01-S12G-FROZEN-TIMEBASE-ENSEMBLE-v1.0.0` (S12G)
- **Completion status:** `STOPPED_FAIL_CLOSED_AT_C0_ENDPOINT_MAPPING_GATE`; no ensemble statistics or scientific classification was performed.
- **Artifacts written:** Complete preregistration and method locks, 96-input/shared-identity/source audits, benchmark, 95-task partial-cache provenance, C0 structural audit, schema/immutability/runtime/replay/scope/failure/status/hash manifests, schema-bearing suppressed scientific tables, six stop-state figures, and this canonical report.
- **Validation result:** Upstream and completed-task checks passed, but the global endpoint-eligibility contract failed: candidate 1 matrix 13 has a zero-update generation and therefore no distinct C0 state at generation 2. Global S12G validation is `FAIL_CLOSED`.
- **Outcome classification:** `S12G_VALIDATION_FAILED_CLOSED` (constraining/contradictory); all label/emergence association questions are `NOT_EVALUATED`.
- **Caveats or blockers:** C0 excludes daughter states. Inserting the daughter, duplicating the prior state, skipping the generation, or reusing the last state would change or silently complete the locked clock. The preregistered stop rule forbids that repair after outcomes opened.
- **Recommended next action:** Keep S13 blocked and return for human review. No S12G repair, candidate deletion, candidate reweighting, favorable-candidate analysis, or statistical use of the 95 partial task caches is authorized.

## Lay summary

The three time-base candidates were supposed to be analyzed under one identical set of rules. One retained-overshoot C0 trajectory entered nine generations already at or above the fission threshold, so those generations had no molecular update. Because C0 records only molecular updates, there is no new C0 composition at those fission boundaries. The analysis would have to invent, duplicate, skip, or substitute a state to continue. The method was frozen to forbid exactly that kind of silent choice, so S12G stopped without calculating ensemble associations. The other 95 task results remain uninspected cache material and are not evidence.

## Frozen question

S12G asked whether historical/past-only replicator labels and S12C-confirmed source-defined emergence agree across all three S12FR-confirmed time bases. An ensemble positive required the same frozen gate on all three; no candidate could be selected, eliminated, or reweighted from downstream results.

## Inputs and provenance

- Exactly 96 S12FR confirmation trajectories were mounted, 32 per candidate; zero GARD trajectories were generated.
- All 96 cache hashes and S12FR replay flags passed, and all 32 catalytic-matrix/initial-state identities were shared across candidates.
- Pinned IIGR commit: `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`.
- Pinned PhiRL commit: `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`.
- Safe lattice SHA-256: `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`; no pickle was loaded for scientific execution.
- S12C source equivalence and S12D's 40/40 source-emergence identity evidence passed before the firewall opened.
- Pre-outcome design commit: `0118892d035eef932274b0f44bd1ecc024268fa2`; locked implementation commit: `29a8ac5`.

## Detailed methods

The frozen C0 sequence is initial state plus molecular batch-update states only. The frozen C1 sequence additionally records the selected daughter. A post-fission endpoint maps to the final molecular update in C0 and to the selected daughter in C1. Prefix evaluation begins at 256 prior locked-clock transitions; each source pipeline is independently refit and replayed, with structural suffix checks at every endpoint and executed deletion/shuffle/replacement sentinels at the first, middle, and last eligible endpoint.

Labels were frozen as `HISTORICAL_H090_REPLICATOR` (primary) and `PAST_ONLY_COSINE_REPLICATOR` (secondary). Preprocessing was additive 0.5 closure, full CLR, and removal of original component 100. IIGR synergy plus two downward-causation atoms was primary; PhiRL was robustness; corrected local Phi-r was comparator-only. These calculations were launched only after the complete design, source identities, input hashes, statistical gates, schemas, and stop rules were committed and pushed.

## Results

The three-trajectory benchmark completed with exact replay and suffix checks. It projected 20.416 CPU-hours and 3.406 wall-hours, below the hard ceilings. During full execution, 95/96 tasks completed in cache with zero task failure rows and unanimous task-level full replay, prefix replay, and suffix flags. The sole incomplete task was candidate 1 matrix 13.

That trajectory's first fission followed a large retained overshoot: pre-fission mass 197 and selected-daughter mass 90. Generation 2 therefore began above `n_max=80` and performed zero batch updates before fission. The same trajectory has nine zero-update generations: `[2, 4, 12, 15, 26, 32, 57, 62, 91]`. It is the only affected candidate-1 trajectory. At generation 2, the frozen C0 clock has no distinct state that can serve as a new endpoint; processing stopped before source fits for that task.

No cached label or emergence value was collated into a scientific artifact. `candidate_associations.csv`, drift/temporal/spike/identity/future-dependence/cross-candidate/adjudication tables are schema-bearing and empty. The six figures are explicit stop-state figures. No candidate-specific or ensemble association result exists.

## Validation

- Prior immutability: PASS across 465 S01–S12FR artifact files and 96 locked trajectory caches; changed count 0.
- Source/metric identity: PASS before execution.
- Shared identities: PASS, 32/32 paired matrix/initial units.
- Input hashes and S12FR replay evidence: PASS, 96/96.
- Completed source tasks: 95/95 full replay, prefix replay, and suffix task flags passed; zero task-level failure rows.
- Required table schemas: PASS; all scientific tables are suppressed/empty after the global stop.
- Scope: zero new GARD trajectories, zero predictions, zero MLP fits, zero interventions, zero candidate selection/reweighting, zero S13 work.
- Runtime: approximately 4.567 wall-hours and 26.034 summed completed-worker CPU-hours; no GPU use and no hard ceiling exceeded.
- Partial-cache provenance: 950 files, 18023415 bytes, all hash-recorded in `partial_execution_manifest.json`; none promoted as scientific evidence.

## Commands

```bash
PYTHONPATH=src pytest -q tests/e01/test_s12g_frozen_timebase_ensemble.py
ruff check src/e01_frozen_timebase_ensemble scripts/e01/freeze_s12g_preregistration.py scripts/e01/run_s12g_frozen_timebase_ensemble.py tests/e01/test_s12g_frozen_timebase_ensemble.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s12g_preregistration.py --design-commit 0118892d035eef932274b0f44bd1ecc024268fa2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s12g_frozen_timebase_ensemble.py --workers 6
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/finalize_s12g_fail_closed.py
```

## Caveats, blockers, and interpretation

- This is a structural endpoint-definition failure, not evidence for or against an emergence/replication association.
- The 95 completed caches cannot be analyzed as a favorable subset or promoted as confirmation evidence.
- A rule for zero-update C0 generations would be a new methodological choice. None was inferred or added after failure.
- Full fits would be retrospective; only prefix fits could have addressed prospective behavior.
- Public source code remains source-informed, not author-code, paper-primary, or exact GARD identity.
- S12F remains `SIMULATOR_IDENTIFICATION_FAILED`; S12FR remains `NONIDENTIFIABLE_TIMEBASE_ENSEMBLE`; all prior negative and failed evidence remains intact.

## Recommended next action

Return for mandatory human review with S13 `BLOCKED_PENDING_S12G_HUMAN_REVIEW`. No repair or continuation is authorized. A future human decision would have to explicitly preregister how zero-update C0 generations are represented and whether doing so preserves the meaning of the locked clock; it must not reuse the 95 cached results for method selection.
