# CA motif-lineage Stage 1

State: **complete**. Profile: `reference`.
Verdict: **ROBUST_LOCAL_MOTIF_CONTROLLABILITY**.
Elapsed: `0.040` wall hours.

## Validation nominees

- `contextual256-w32-s025-d08` at sweep 8: intact crossover `-0.002685546875`, CI `[-0.005126953125, -0.000732421875]`; controllability `False`; robust `False`.
- `contextual256-w32-s025-d16` at sweep 8: intact crossover `-0.002685546875`, CI `[-0.005126953125, -0.000732421875]`; controllability `False`; robust `False`.
- `motif_energy512-w32-s025-d32` at sweep 64: intact crossover `0.920166015625`, CI `[0.8828125, 0.95263671875]`; controllability `True`; robust `True`.
- `motif_energy512-w16-s025-d32` at sweep 64: intact crossover `0.8740234375`, CI `[0.82470703125, 0.916748046875]`; controllability `True`; robust `True`.

## Interpretation boundary

This stage tests whether a motif-indexed hidden channel can steer one visibly reset daughter. It does not test daughter rewriting or persistence across generations and therefore cannot, by itself, establish plastic heredity.

Stages 2--5 remain blocked until this result and its frozen decision artifact are reviewed.
