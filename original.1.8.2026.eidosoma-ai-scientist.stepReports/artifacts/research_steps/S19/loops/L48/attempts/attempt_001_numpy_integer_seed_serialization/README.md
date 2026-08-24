# L48 Attempt 001 — Preserved technical failure

- **Stage:** first matrix-bootstrap seed construction, before any scientific table, figure, report, classification, or root-ledger update was written.
- **Failure:** pandas supplied the frozen `averageBranchBudget` group key as `numpy.int64`; Python `json.dumps` rejected that scalar while deriving the registered bootstrap seed.
- **Scientific status:** no L48 cohort result was persisted or released. The target, input rows, branch halves, budgets, allocation, statistics, controls, and gates were unchanged.
- **Authorized repair:** TA01 converts NumPy scalar seed components with `.item()`. The mandatory fixture requires `derived_seed("fixture", numpy.int64(32)) == derived_seed("fixture", 32)`, proving the repair uses the already registered native-integer seed material.
- **Disposition:** retry only from a fresh cache after committing and pushing TA01. The original implementation and failure remain recoverable from the pre-outcome lock and repository commit.

The failure occurred on 2026-08-12 UTC and raised `TypeError: Object of type int64 is not JSON serializable` in `seed_material` before the first bootstrap result was constructed.
