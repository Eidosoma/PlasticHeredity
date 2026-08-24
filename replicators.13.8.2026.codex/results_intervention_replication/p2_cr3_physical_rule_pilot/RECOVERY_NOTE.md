# P2 readback recovery

Amendment: `1cad7cda20f852e33c21e2ac1aee4e96c925c2c170b3f2714dcd7642ef3699ea`.

The original run completed every primary and replay future but stopped because readback omitted the derived `pilot_eligibility` field. This source-additive recovery loaded all 800 completed state checkpoints, generated zero futures, recomputed that field from readback inference and exact replay, verified unchanged checkpoint aggregates, and sealed the result.

No simulator, endpoint, state, intervention, seed, model, branch, inference, margin, gate, or claim boundary changed.
