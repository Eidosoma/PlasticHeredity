# Reviewer motif-channel replication

This is an independent, reviewer-facing replication of the latest positive
Rule-31649 cellular-automaton result. It asks whether local motif statistics
written by A-like and B-like parents causally steer bitwise-identical daughters
toward the matching form, and whether the fixed reader generalizes. It does not
test multigenerational carrier renewal and cannot establish Plastic Heredity.

The package has two strictly separated layers:

1. `parity` checks retained data non-evidentially: state encoding, toroidal
   observers, Rule-31649 parent evolution, calibration tables, writer values,
   and the recorded headline decisions.
2. Stage 1 and, only after an explicit gate, Stage 2 generate fresh daughter
   trajectories using frozen acquisition donors that never appeared in any
   retained historical outcome pair.

Only `snapshot.py` may read the historical directory. It accepts an explicit
root and opens a fixed allowlist of JSON and Markdown files. It extracts a
minimal immutable bundle under `artifacts/input`; all subsequent scientific
commands use that local bundle only. No historical implementation source is
read, copied, imported, hashed, or executed.

## Workflow

Run these commands from the repository root:

```bash
.venv/bin/python -m reviewer_motif_channel_replication.run snapshot --source /path/to/codex.reconstructionsAndStressTesting
.venv/bin/python -m reviewer_motif_channel_replication.run validate
.venv/bin/python -m reviewer_motif_channel_replication.run parity
.venv/bin/python -m reviewer_motif_channel_replication.run register-stage1
.venv/bin/python -m reviewer_motif_channel_replication.run stage1 --workers 8
.venv/bin/python -m reviewer_motif_channel_replication.run stage1-report
```

Stage 2 never starts automatically. If and only if the fixed primary passes the
registered robust Stage-1 gate, review the report and then explicitly run:

```bash
.venv/bin/python -m reviewer_motif_channel_replication.run register-stage2
.venv/bin/python -m reviewer_motif_channel_replication.run stage2 --workers 8
.venv/bin/python -m reviewer_motif_channel_replication.run stage2-report
```

Use `status` while a campaign is running and `verify` to recheck every local
hash binding. Checkpoints are per pair, content-addressed, atomic, and safe to
resume. Semantic seeds do not depend on worker count or completion order.

No Stage 3, new stress panel, or automatic continuation is implemented.
