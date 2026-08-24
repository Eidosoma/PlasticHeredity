# Exploratory GARD dynamic-regime clean room

This directory contains a preregistered follow-up asking whether the frozen
reconstructed GARD dynamics cross a perturbation-contraction/expansion
boundary, whether their original operating point lies near it, and whether
operational PH or the previously frozen lineage carrier are enriched there.

The primary regime map uses fresh catalytic matrices that were not selected
for prior PH capability.  The previous 47-rule strict-capable bank appears only
in the separately labelled frozen-carrier overlay.

Long production work is checkpointed and run detached:

```bash
./run_detached_pipeline.sh 16
```

Status is recorded in `artifacts/STATUS.json`; the cumulative ledger prevents a
restart from exceeding eight hours.

