# Pre-print review and recovery

## Decision

**Review attempt:** 1  
**Review command:** `chief-command:21e60b2fafd63e6398dd710bcaeb73a6`  
**Decision:** `ok`

The pre-print bundle is scientifically acceptable after a bounded prose, scope, and provenance recovery. I verified the central claims directly against the canonical L44, L53, L54, S18, and S20 artifacts and machine-readable results. No simulation, branching, model fitting, metric recomputation, or scientific-result change was needed.

## What was reviewed

### Code and computational workflow

- Reviewed the L44 plastic-heredity, L53 regime-capacity discovery, L54 untouched-confirmation, and S20-B closeout runners and their supporting source modules.
- Confirmed that the L53 primary graph/state view is target-blind and that L54 uses the same frozen representation and model without refitting.
- Confirmed the L54 gate is conjunctive, keeps simulator candidates separate, and requires exact replay.
- Ran the focused test suite with the documented module path: 22 tests passed.
- Ran Ruff over the scientific L44/L53/L54 implementation and tests: all checks passed.
- Ran the deterministic S20-B preflight. The paper, V1/V2, S01-S17, S19, L53, L54, S17 waiver, classification, and regeneration validations all passed.

An initial test invocation omitted `PYTHONPATH=src` and failed during collection because the repository module was not on the import path. Repeating the same suite with the repository's documented environment passed all 22 tests; this was an invocation issue, not a code or result failure.

### Results, provenance, and figures

- Cross-checked the canonical L44, L53, and L54 reports and machine-readable tables rather than relying on bundle-level excerpts.
- Verified L54's 40 untouched matrices, 80 trajectories, 400 states, 64 branches per state, two exact campaigns, frozen F12 target, candidate-separated gates, reliability metrics, rank correlations, proper-score improvements, and permutation results.
- Verified the reported minimum full-versus-direct log-loss bootstrap lower bounds: 0.0259224254 and 0.0355115313.
- Verified the immutable S18 totals and all retained negative boundaries.
- Verified all 14 copied images against their source hashes and visually checked the nine pre-print figure-caption pairings.
- Verified that no mounted dataset was used.
- Counted 57 S19 loop directories: L01-L54 plus the separately versioned L06R, L11R, and L49R repairs.
- Ran an assertion-only evidence crosswalk covering fixtures, regeneration, metrics, gates, paper structure, image provenance, and loop accounting: all assertions passed.

### Draft and bundle integrity

- Parsed every bundle Markdown file with Pandoc.
- Verified all 27 local links resolve.
- Verified the required section structure and ordering, including **New Hypotheses** after **Conclusion & Future Directions** and before **References**.
- Confirmed `RESEARCH_PLAN.md` and `FULL_PLAN.md` retained their baseline hashes.
- Confirmed the repository remained clean and no code or plan file was changed during review.

## Planner findings and resolution

| Finding | Resolution |
|---|---|
| Canonical L44/L53/L54 evidence was not present in the planner excerpt | Located and directly verified the canonical reports, tables, locks, and manifests in the workspace. |
| Exact lower bounds 0.0259 and 0.0355 needed support | Confirmed exact values 0.0259224254 and 0.0355115313 in the registered L54 full-versus-direct log-loss comparison; qualified the draft wording. |
| L53 lock language was too categorical | Reframed L53 as adaptive exploratory discovery and L54 as the untouched test of the frozen final contract. |
| “54 additive loops” omitted repair units | Replaced it with “running through L54, including L06R, L11R, and L49R.” |
| Title generalized to GARD broadly | Scoped the title and major conclusions to two reconstructed GARD simulator candidates. |
| Figure 2 needed alignment and lag caveats | Added available-case alignment, unequal tail support, alternate-alignment robustness, and unresolved author alignment/Ljung–Box lag semantics beside the result and caption. |
| Authorship and affiliations unconfirmed | Kept explicit human-completion placeholders; no identities or affiliations were invented. This is a submission-metadata task, not a scientific blocker. |

## Additional correction

One bundle summary conflated the prevalence of the old-neighbourhood event with the conditional probability of exact old-composition recovery after a break. The report now distinguishes the event prevalence (approximately 0.0026–0.0069 in the registered reliability summaries) from conditional recovery after a break (approximately 0.0024–0.0051 in the plasticity decomposition).

## Files changed

- `PRE_PRINT_PAPER_DRAFT.md`
- `STATUS_CLUSTER_PAPERRECON_FULL_REPORT.md`
- `STATUS_REPORT_OVERVIEW.md`
- `STATUS_CLUSTER_PROCESSRISK_NEW_DISCOVERIES_REPORT.md`
- `PRE_PRINT_REVIEW.json`
- `PRE_PRINT_REVIEW.md`

No underlying scientific artifact, result table, source code, research plan, or classification was changed.

## Remaining nonblocking editorial work

Before journal submission, humans must supply and approve authors, affiliations, author contributions, acknowledgements, funding, conflicts of interest, and corresponding-author details. The draft marks these fields explicitly. No unresolved scientific problem requires human judgment for this review decision.
