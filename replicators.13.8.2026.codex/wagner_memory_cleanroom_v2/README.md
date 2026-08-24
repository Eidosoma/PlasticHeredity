# Corrected Wagner memory clean-room v2

V2 preserves the v1 run and corrects the review blockers: source landscapes and
two midpoint starts, trajectory-written carriers, paired arm randomness, the
complete mark control matrix, direct A/B crossover reliability, genuine held-out
history decoding, simultaneous source bootstrap bounds, exact count guards, and
future-ID replay. The frozen protocol also gives smoke, full, and admission
benchmark sources disjoint deterministic namespaces, so no diagnostic source is
later reused as confirmatory evidence.

The existing v1 environment contains the frozen dependencies:

```bash
PYTHONPATH=src ../wagner_memory_cleanroom/.venv/bin/python -m wagner_memory_cleanroom_v2 validate
PYTHONPATH=src ../wagner_memory_cleanroom/.venv/bin/python -m pytest -q
scripts/run-campaign-detached.sh runs/wagner-memory-v2 full
scripts/campaign-status.sh runs/wagner-memory-v2
```

Smoke and quick profiles never receive scientific verdicts. A full launch is
admitted only after the full preflight validation passes and exactly two GPU
workers project below the registered limit. The validation output is retained
inside the sealed run.
