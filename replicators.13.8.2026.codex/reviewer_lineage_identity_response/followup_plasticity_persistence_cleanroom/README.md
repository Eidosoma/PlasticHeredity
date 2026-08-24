# Plasticity–persistence and lineage-identity campaign

This clean-room follow-up prospectively tests whether weak break-and-recovery
plasticity trades against strict persistent form, whether high-F12 states carry
lineage-specific information, and which unresolved GARD simulator choices
control the result.

The surface and factorial cohorts use fresh matrices and seed domains. Earlier
results generate the fixed hypotheses only and are never pooled with the new
outcomes. This is exploratory work for a possible later preprint, not evidence
for the current preprint.

The production pipeline is checkpointed and detached:

```bash
./run_detached_pipeline.sh 16
```

Read `artifacts/STATUS.json` for status. Restarts share the same cumulative
eight-hour watchdog ledger.

