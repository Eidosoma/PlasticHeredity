# Wagner memory clean-room replication

Independent, GPU-native implementation of the registered Wagner expression-
state and multigeneration-carrier tests. The package has no runtime dependency
on NewIdeas or on the existing GRN F12 project.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/wagner-memory validate
.venv/bin/wagner-memory benchmark --run runs/wagner-memory-v1 --profile full
scripts/run-campaign-detached.sh runs/wagner-memory-v1 full
scripts/campaign-status.sh runs/wagner-memory-v1
```

`campaign` verifies both GPUs, runs the discarded admission benchmark, creates
fresh sealed cohorts, runs each stage with one deterministic worker per GPU,
performs independent regeneration and replay, then writes the verdict, report,
lay summary, manifests, and checksums. `smoke` and `quick` profiles exercise the
same pathway but are explicitly non-scientific and cannot receive a claim.

