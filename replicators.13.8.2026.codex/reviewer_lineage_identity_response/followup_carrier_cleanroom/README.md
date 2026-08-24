# Clean-room GARD lineage-carrier follow-up

This isolated campaign asks whether an explicitly inherited molecule-indexed
register can make a post-break GARD form persist, install that form in an
abundance-matched stranger, or select between two pre-existing forms. It follows
the frozen protocol in `PROTOCOL.md` and the interpretation limits in
`REPORTING_BOUNDARY.md`.

The scientific inputs are only the verified 50-rule cohort, strict-B bank,
frozen GARD simulator, and its seed/configuration code. Selected NewIdeas GRN
documents and result summaries are hashed separately as hypothesis-only inputs.
No Wagner implementation is used.

Commands (from `replicators.13.8.2026.codex`):

```bash
python -m reviewer_lineage_identity_response.followup_carrier_cleanroom.run_campaign prepare
python -m reviewer_lineage_identity_response.followup_carrier_cleanroom.run_campaign test
python -m reviewer_lineage_identity_response.followup_carrier_cleanroom.run_campaign benchmark
python -m reviewer_lineage_identity_response.followup_carrier_cleanroom.run_campaign calibrate --workers 16
python -m reviewer_lineage_identity_response.followup_carrier_cleanroom.run_campaign confirm --workers 16
python -m reviewer_lineage_identity_response.followup_carrier_cleanroom.run_campaign analyze
python -m reviewer_lineage_identity_response.followup_carrier_cleanroom.run_campaign report
python -m reviewer_lineage_identity_response.followup_carrier_cleanroom.run_campaign verify --workers 16 --full-replay
python -m reviewer_lineage_identity_response.followup_carrier_cleanroom.run_campaign status
```

`run_detached_pipeline.sh 16` launches the complete resumable campaign in a
detached process group. A persistent cumulative ledger applies a 27,000-second
soft stop and a 28,800-second hard stop across restarts. Long stages checkpoint
atomically by candidate/rule/setting.

