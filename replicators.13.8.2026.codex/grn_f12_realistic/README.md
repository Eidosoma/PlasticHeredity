# Realistic GRN F12 replication

This is a clean-room, GPU-native test of the Wagner PH prediction result in two
more realistic gene-regulatory settings:

1. a powered 32-gene continuous noisy regulatory network; and
2. a 16-gene stochastic mRNA/protein tau-leap bridge.

The primary event, `JOINT_BREAK_RUN3/F12`, is a low parent-daughter phenotype
similarity within 12 generations followed later by three consecutive high
similarities. The prediction question is fixed before confirmation: can the
present molecular/regulatory state forecast the probability of that event more
accurately than history alone?

The continuous tier carries the primary replication verdict. The molecular tier
can upgrade the result to a cross-realism replication; it cannot retroactively
invalidate the continuous result. Mechanistic controls are reported separately.

## Commands

From this directory, after creating the supplied environment:

```bash
.venv/bin/pip install -e .
.venv/bin/grn-f12 validate
.venv/bin/grn-f12 campaign --run runs/grn-f12-v1 --profile full
scripts/run-campaign-detached.sh runs/grn-f12-v1 full
scripts/campaign-status.sh runs/grn-f12-v1
```

`campaign` performs registration, an eight-network-per-tier admission benchmark,
calibration, development, frozen model fitting, untouched confirmation,
independent regeneration, controls, analysis, verification, and sealing. Full
campaigns refuse CPU fallback and use one deterministic worker per visible L4.
The launcher uses `tmux`; polling the status script does not consume or disturb
the scientific process.

## Verdict hierarchy

- `CONTINUOUS_CONFIRMED`: every registered continuous prediction gate passes.
- `CROSS_REALISM_CONFIRMED`: continuous passes and the independently trained
  molecular tier passes the same gates.
- `MECHANISTIC_SUPPORT`: the separate intervention controls pass.
- `INCOMPLETE` or `NOT_CONFIRMED`: missing/failed gates remain visible; no gate
  is substituted after outcomes are observed.

