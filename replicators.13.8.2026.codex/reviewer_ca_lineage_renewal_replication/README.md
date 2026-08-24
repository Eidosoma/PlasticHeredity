# Independent CA lineage-renewal replication

This package is a clean-room replication of the final positive Stage-3R
cellular-automaton result suggested by the NewIdeas data and documentation. No
NewIdeas implementation code was read, copied, hashed, imported, or executed.
The historical result supplies a hypothesis and fixed protocol, never evidence
for the local verdict.

The test asks whether daughters of Rule-31649 founders can repeatedly recover
an A-like or B-like form from a hidden motif carrier, then rewrite that carrier
from their own late texture after reading has stopped. It tests the strict
original-form endpoints at generations 4, 8, and 16 and the complete causal
ladder. It is not limited to the earlier strict-8 result.

## Fixed confirmation

- Reader: `motif_energy512-w32-s025-d32`, unchanged from the sealed local
  Stage-1/2 replication.
- Daughter writer: motif counts from sweeps 49--64, after reading ends at sweep
  32; its raw carrier is multiplied by the universal gain 0.5.
- Observer: accumulated 2x2 texture from sweeps 57--64, plus the terminal 2x2
  gate observer.
- Cohort: 96 matched pairs and 64 futures per history, all donor-disjoint from
  local Stage 1/2 and every pair exposed in the NewIdeas Stage-3/3R data.
- Causal panel: intact, zero, shuffle, reader-off, founder-writer-off,
  no-rewrite, generation-2 ablation, same/opposite generation-4 rescue,
  opposite founder, and repeated 1% sign corruption.

The two additional fully fresh pairs are an engineering quarantine only and
never enter inference. Every visible daughter starts from the same pair-specific
neutral board on every generation. Dead and unresolved futures remain in all
denominators. Pair-cluster inference uses 10,000 bootstrap draws at alpha
0.0125.

The source's secondary drift decoder is not replicated: the data/docs do not
operationally define its “independent texture descriptor.” It is non-gating for
the strict original-form verdict, and inventing a substitute would no longer be
a direct replication.

## Workflow

Preparation freezes a local allowlisted snapshot and seals the design without
creating lineage outcomes:

```bash
.venv/bin/python -m reviewer_ca_lineage_renewal_replication.run snapshot --source /path/to/codex.reconstructionsAndStressTesting
.venv/bin/python -m reviewer_ca_lineage_renewal_replication.run validate
.venv/bin/python -m reviewer_ca_lineage_renewal_replication.run parity
.venv/bin/python -m reviewer_ca_lineage_renewal_replication.run register
```

Registration never launches confirmation. After reviewing the registration,
the explicit runner performs the engineering quarantine, resumable
confirmation, reporting, and verification:

```bash
bash reviewer_ca_lineage_renewal_replication/run_confirmation.sh 1 8
```

Use `status` while it runs. Per-pair checkpoints are atomic, hash-bound, and
worker-count invariant.
