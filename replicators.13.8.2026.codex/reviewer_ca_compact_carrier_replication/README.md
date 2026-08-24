# Clean-room compact-carrier replication

This package prospectively replicates the latest positive cellular-automaton
compression result with fresh Rule-31649 founders.  It tests the full 512-value
float32 anchor, the frozen rank-8 PCA 4-bit codec, and the frozen rank-16 Walsh
4-bit codec under ordinary and registered moderate-copying stress.

The NewIdeas campaign is an outcome-known hypothesis source only.  The input
adapter allow-lists data and documents; it never opens, imports, hashes, or
executes NewIdeas source code, and it does not copy NewIdeas results or
checkpoints into this evidence bundle.

The prospective workflow is:

```bash
.venv/bin/python -m reviewer_ca_compact_carrier_replication.run prepare
.venv/bin/python -m reviewer_ca_compact_carrier_replication.run acquire
.venv/bin/python -m reviewer_ca_compact_carrier_replication.run validate
.venv/bin/python -m reviewer_ca_compact_carrier_replication.run audit-tests
.venv/bin/python -m reviewer_ca_compact_carrier_replication.run smoke
.venv/bin/python -m reviewer_ca_compact_carrier_replication.run register
.venv/bin/python -m reviewer_ca_compact_carrier_replication.run confirm \
  --resume --authorize-confirmation --workers 20
.venv/bin/python -m reviewer_ca_compact_carrier_replication.run verify
```

Confirmation has 128 independent founder pairs, 64 futures per history, 16
generations, three codecs, two environments, and twelve causal conditions.
The atomic unit of work is pair × codec × environment (768 checkpoints).

The targeted replication succeeds only when the identity anchor passes under
ordinary conditions and the frozen Walsh carrier passes the complete causal
ladder under both ordinary and moderate stress.  PCA is reported but cannot
substitute for Walsh.  Walsh-minus-identity contrasts at generations 8 and 16
are registered secondary, non-gating outcomes.

The scientific boundary is deliberately narrow: a positive result supports a
compact causal carrier in this synthetic CA lineage.  It is not evidence of
cross-substrate transfer, metabolism, agency, biological life, or the full
Ruliad.
