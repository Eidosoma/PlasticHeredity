# Reviewer lineage-identity response

This isolated package implements reviewer tests 2--4: a matched
sibling-versus-stranger baseline, independent post-break forks, and an
attractor census over 50 strict-capable catalytic rules. It writes only below
this directory. Existing Codex results and simulator sources are verified and
read read-only; draft-only directories and the manuscript are not analysis
inputs.

The primary object is the final daughter of a strict coherent-eight episode.
Ordinary F12 run-three episodes are repeated as descriptive, non-rescuing
controls because F12 usually does not define one coherent form.

## Frozen design

- Source cohort: the 134 REGCONF matrices with at least one archived strict
  event in both candidates.
- Rule selection: 50 shared matrices selected by a frozen SHA-256 order, not by
  event rate.
- Fresh simulation: both candidates, 128 independent random starts per rule,
  256 fissions, with the first 32 discarded as burn-in.
- Strict-B bank: first 20 qualifying lineages in seed order; indices 128--255
  are generated only when the fixed census cohort does not fill the bank.
- Census: always restricted to the fixed 128 random starts.
- Inference: candidate-separated, 10,000 whole-rule bootstrap replicates, with
  no pooling or candidate rescue.
- Readouts: statistically corrected primary gates plus the reviewer's literal
  overlap and `H=0.90` gates.

`REVIEW_AND_PLAN.md` contains the complete rationale, endpoint contracts,
decision rules, source boundary, output schemas, and claim limitations.

## Reproduce

Run from `replicators.13.8.2026.codex/` with an environment satisfying
`requirements-lock.txt`:

```bash
python -m reviewer_lineage_identity_response.run_analysis prepare
python -m reviewer_lineage_identity_response.run_analysis simulate --mode all --workers 14
python -m reviewer_lineage_identity_response.run_analysis fork --workers 14
python -m reviewer_lineage_identity_response.run_analysis analyze
python -m reviewer_lineage_identity_response.run_analysis report
python -m reviewer_lineage_identity_response.run_analysis verify --full-replay
python -m pytest -q reviewer_lineage_identity_response/test_lineage_identity.py
```

`simulate` and `fork` are checkpointed per candidate/rule cell and may be
restarted safely. `status` is read-only:

```bash
python -m reviewer_lineage_identity_response.run_analysis status
```

The detached wrapper accepts a worker count as its first argument:

```bash
bash reviewer_lineage_identity_response/run_detached_pipeline.sh 14
```

## Artifacts

- `artifacts/protocol/`: immutable protocol, selected rules, source hashes,
  seed registry, registration, and checksums.
- `artifacts/work/lineages/`: compressed fixed and conditional-extension
  trajectory checkpoints.
- `artifacts/work/forks/`: strict-B and F12-control fork checkpoints.
- `artifacts/output/`: complete tables, figures, scientific report, appendix,
  proposed reviewer/manuscript text, manifest, and checksums.
- `artifacts/verification/`: checkpoint audit and full deterministic replay
  verdict.

The campaign is complete only when `verification_audit.json` reports
`"complete": true`. A verification without `--full-replay` is an integrity
inspection, not the registered completion audit.

