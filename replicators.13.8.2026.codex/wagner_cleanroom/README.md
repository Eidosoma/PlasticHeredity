# Wagner clean-room replication

This standalone project implements two Wagner-only studies:

1. A fresh 240-rulebook replication of the exact-state prediction channel.
2. A separately sealed PH-style predictor extension with 96 development and 128
   untouched evaluation rulebooks.

The model has ten binary genes, dense standard-normal regulatory weights, zero
bias, sequential in-place updates, deterministic (`theta=0`) landscapes, a 5%
expression-flip rate, and exact enumeration of all 1,024 states. The independent
inferential unit is the sampled rulebook.

## Commands

From this folder, use the parent environment without importing its package:

```bash
PYTHONPATH=src ../.venv/bin/python -m wagner_cleanroom validate
PYTHONPATH=src ../.venv/bin/python -m wagner_cleanroom benchmark --output runs/benchmark
PYTHONPATH=src ../.venv/bin/python -m wagner_cleanroom campaign --output runs/campaign --workers 12
PYTHONPATH=src ../.venv/bin/python -m wagner_cleanroom verify runs/campaign
PYTHONPATH=src ../.venv/bin/python -m wagner_cleanroom replay runs/campaign FUTURE_ID
```

Long campaigns should be detached so their process does not depend on an
interactive agent session:

```bash
scripts/run-campaign-detached.sh runs/campaign 12 full
scripts/campaign-status.sh runs/campaign
```

The launcher writes its PID, log, and launcher metadata to a sibling
`runs/campaign.launcher/` directory, outside the scientific output tree. The
scientific runner continues to write atomic `STATUS.json` heartbeats and
per-rulebook checkpoints inside the campaign directory.

`campaign` measures wall time from its discarded benchmark, refuses to begin the
registered cohorts when the projected runtime is unsafe, checkpoints by
rulebook, stops submitting work after 11.5 hours, and hard-stops at 12 hours.
`--profile smoke` is available only for engineering tests and is prominently
marked non-scientific.


## Frozen scientific conventions

- A state integer uses bit `i` for gene `i`; set bits encode `+1` and clear bits
  encode `-1`.
- A sequential sweep updates rows `0..9` in place. An exactly zero field retains
  the previous bit.
- Eligible rulebooks contain a pair of point attractors at normalized Hamming
  distance at least 0.40, each with a basin of at least 5%, and a one-bit forced
  break for each form.
- A/B selection maximizes the smaller basin, total basin size, separation, then
  uses state IDs as an ascending tie-break.
- The two midpoint recipients split A/B-disagreeing coordinates in a frozen
  permutation and are complementary on those coordinates.
- A destination is the first active point attractor occupied for three
  consecutive descendant boundaries. Cycles and nonconvergence are explicit
  outcome classes.

All large future data are compact, per-rulebook `.npz` shards. JSON manifests,
reports, replay audits, and SHA-256 checksums make completion independently
auditable.
