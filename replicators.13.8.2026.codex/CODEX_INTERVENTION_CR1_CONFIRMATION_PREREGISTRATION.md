# Codex full CR1 model-guided molecular confirmation

This is the full prospective confirmation of the frozen Codex
`JOINT_BREAK_RUN3` molecular intervention predictor. It is independent of P4
and cannot use P4 outcomes. It preserves the sealed 40-matrix P1 pilot as
developmental evidence only.

## Frozen design

- 200 entirely fresh catalytic matrices shared across candidates 02 and 03;
- untreated landmarks 20, 35, 50, 65, and 80;
- 2,000 restored states;
- arms `MODEL_UP`, `MODEL_DOWN`, uniformly random legal substitution, and
  `NOOP`;
- exhaustive scoring of every legal mass-preserving one-molecule substitution
  by the immutable candidate-separated 5x-development predictor;
- deterministic lexicographic resolution of extreme-score ties;
- 64 F12 futures per arm/state, halves A=0--31 and B=32--63;
- 512,000 primary futures and a complete 512,000-future replay;
- common future random streams across arms, with arm identity absent from the
  seed; and
- no refitting, recalibration, threshold change, matrix replacement, future
  retry, or candidate pooling.

## Primary gate

Candidate 02 half A, candidate 02 half B, candidate 03 half A, and candidate 03
half B must each satisfy:

1. `MODEL_UP - MODEL_DOWN > 0`;
2. positive 95% whole-matrix bootstrap lower bound;
3. Holm-adjusted whole-matrix sign-randomization `p < 0.05`;
4. `MODEL_UP - NOOP` has a positive 95% lower bound;
5. `NOOP - MODEL_DOWN` has a positive 95% lower bound;
6. the random arm is TOST-equivalent to NOOP within `+/-0.025`; and
7. the absolute random-minus-NOOP point difference is no greater than 25% of
   the `MODEL_UP - MODEL_DOWN` effect.

Complete exact replay and artifact readback are additional integrity gates.
The catalytic matrix is the inference unit; all states, candidates, arms, and
branch halves belonging to one matrix travel together in every one of 4,096
bootstrap and 4,096 randomization draws.

## Operational boundary

The run may start only after P4 has a checksum-sealed terminal result and an
operational phase-boundary estimate leaves at least 17 CPU-hours available
inside the current 30 CPU-hour round. A started run completes, replays, seals,
and reports even if an estimate was imperfect.

The result stops before CR2. Exact restored states are retained so a passing
CR1 may later support the separately registered dose-response experiment.

## Claim boundary

A pass may support prospective model-guided molecular control of the
operational Codex break-and-renewal event. It does not establish strict-eight
control, agency, biological memory, life, autonomous organization, real
prebiotic chemistry, or a universal origin-of-life mechanism.

