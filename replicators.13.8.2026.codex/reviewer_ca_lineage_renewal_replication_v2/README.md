# Corrected CA lineage-renewal replication (v2)

This is the standalone corrected confirmation of the final positive Rule-31649
Stage-3R hypothesis. It preserves the sealed v1 package and outcomes unchanged.
The v1 negative result is quarantined as non-comparable because its reader,
reset, and within-sweep ordering differed materially from the intended model.

The v2 implementation uses the one-time recovered specification in
`SOURCE_SPEC.md`, but the source implementation and all retained source
outcomes have no evidential role. The scientific estimate comes only from 92
locally untouched, donor-disjoint, same-launch A/B pairs selected before any v2
confirmation trajectory is generated.

The confirmation retains all 11 interventions, 64 futures per history, 16
visibly reset generations, checkpoints 1, 2, 4, 8, and 16, strict original-form
gates, and the complete carrier and 41-feature visible held-out decoder panel.
Pair-cluster intervals use 10,000 draws at alpha 0.0125.

## Workflow

```bash
.venv/bin/python -m reviewer_ca_lineage_renewal_replication_v2.run prepare
.venv/bin/python -m reviewer_ca_lineage_renewal_replication_v2.run validate
.venv/bin/python -m reviewer_ca_lineage_renewal_replication_v2.run audit-tests
.venv/bin/python -m reviewer_ca_lineage_renewal_replication_v2.run register
.venv/bin/python -m reviewer_ca_lineage_renewal_replication_v2.run quarantine --workers 1
.venv/bin/python -m reviewer_ca_lineage_renewal_replication_v2.run confirm --workers 8
.venv/bin/python -m reviewer_ca_lineage_renewal_replication_v2.run report
.venv/bin/python -m reviewer_ca_lineage_renewal_replication_v2.run verify
```

`status` is read-only and can be called while confirmation is running. Pair
checkpoints are atomic, hash-bound, resumable, and independent of worker count.
No additional scientific experiment is included.
