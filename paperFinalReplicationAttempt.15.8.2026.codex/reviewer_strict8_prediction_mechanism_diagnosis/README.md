# Strict-8 prediction-mechanism diagnosis

This self-contained subfolder implements the frozen internal analysis requested after the strict-eight predictor review. It does not modify the manuscript or any completed audit.

The analysis has four linked parts:

1. Decompose the rare strict-eight endpoint into four conditional hurdles: a break, a renewed run of eight after that break, mutual coherence of the eight daughters, and separation from the old anchor.
2. Compare development-fitted, confirmation-scored models using the retained `h10` history block (H), six exact concentration descriptors (C), and the 26-variable state block (S): H, H+C, H+S, and H+C+S.
3. Exactly replay every selected strict-event window and its frozen same-state control, inspect all 28 daughter pairs, and decompose Bray–Curtis distance into the leading type, ranks 2–5, and the remaining tail.
4. Run a fresh common-random-stream intervention on all 2,000 retained confirmation states. Eleven arms move zero, one, or four molecules along separate evenness and occupied-richness axes, with 64 futures per state and arm.

## Reproducibility contract

- `core.py` contains the pure endpoint-transition, concentration, edit, and Bray-decomposition rules.
- `run_analysis.py` contains the frozen protocol, development-only model sealing, exact replay, intervention, inference, reporting, and verification.
- `test_core.py` checks mass preservation, nested doses, axis direction, transition denominators, and paired power accounting.
- Long work is checkpointed per state under `artifacts/work/`. Re-running the same command resumes completed work.
- The future seed deliberately excludes the intervention arm, so competing arms use common random streams.
- Final output checksums and a result manifest are created only after deterministic replay verification succeeds.

## Commands

Use the replication environment:

```bash
/home/robert/Projects/replications/PlasticHeredity/replicators.13.8.2026.codex/.venv/bin/python -m pytest -q test_core.py
/home/robert/Projects/replications/PlasticHeredity/replicators.13.8.2026.codex/.venv/bin/python run_analysis.py all --workers 14
```

The `all` command freezes the protocol before reading any new confirmation gate or intervention result, then runs every stage and verifies the result. Individual resumable commands are available through `python run_analysis.py --help`. Use `python run_analysis.py status` for checkpoint counts.

## Deliverables

After a successful run, the main human-readable files are:

- `DIAGNOSTIC_REPORT.md` — technical internal interpretation.
- `LAY_FINDINGS.md` — plain-language summary.
- `artifacts/output/result_classification.json` — machine-readable decision readout.
- `artifacts/output/` — prediction, reliability, geometry, intervention, and validation tables and figures.
- `artifacts/output/verification_audit.json` and `result_manifest.json` — exact audit and file identities.

All results are post-hoc diagnostic. The intervention supports causal claims only for the specified editing policies on retained surviving/observable selected-lineage states.
