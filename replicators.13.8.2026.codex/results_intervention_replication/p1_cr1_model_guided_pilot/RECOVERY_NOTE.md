# P1 readback recovery

Amendment: `8ac3df3717880fad899187dcfaaec5c56cda897838e0704511792f883adad620`.

The original run completed all primary and replay futures but stopped because the readback dictionary omitted the derived `pilot_eligibility` field. This source-additive recovery loaded the 800 completed state checkpoints, generated zero futures, recomputed that field from the readback inference and exact-replay flag, verified all checkpoint aggregate hashes unchanged, and sealed the result.

No simulator, endpoint, state, edit, seed, model, branch, inference, margin, gate, or claim boundary changed.
