# Review item and frozen implementation plan

## Reviewer comment

> The direct-history comparator may be too weak a baseline for the headline
> predictor claim. The composite's advantage is measured against a 9-feature
> ridge logistic on history (line 169). But Appendix C shows post-break
> sequences carry first-order Markov and (in test 1) semi-Markov structure worth
> 0.03–0.06 bits/transition. A referee will ask: does the frozen composite beat
> a sequence-model history baseline, not just the 9-scalar summary? Given the
> composite's gain is itself 0.03–0.05 nats, this is not a rhetorical question.
> If the retained artefacts allow scoring a Markov/semi-Markov history baseline
> on the confirmation cohorts without new simulation, I'd add it; if not,
> acknowledge the gap explicitly in limitation 7 or the predictor discussion
> (line 646).

## Scientific distinction

Appendix C predicts the next inheritance symbol after observing part of a
future trajectory.  The headline task predicts, at launch, whether an entire
F12 future will contain a break followed strictly later by three inherited
fissions.  Bits per transition and nats per future are therefore not directly
comparable.  A fair control must use development data only and integrate the
sequence law forward before any confirmation future is observed.

The registered direct comparator already contains current inheritance state,
current regime duration, recent and prefix inheritance fractions, trailing run,
latest continuous H, phase, and mass.  The new models nevertheless test the
remaining possibility that nonlinear duration effects or ordered prefix
information explain the composite advantage.

## Frozen execution contract

1. Freeze hashes, cohorts, grids, inference, and gates before result readout.
2. Replay development and confirmation natural main paths only; save launch
   histories separately from retained confirmation targets.
3. Fit every new model by clean room and candidate using development data only.
4. Score retained outcomes without refitting, recalibration, pooling candidates,
   or pooling deterministic halves.
5. Use whole-matrix bootstrap and paired sign randomization.  Apply Holm across
   the eight primary composite-versus-lagged cells.
6. Report all cells.  Secondary 40-matrix results cannot rescue a failed
   primary gate.
7. Generate proposed manuscript/reviewer-response wording but do not edit the
   preprint.

The broad conclusion passes only if all eight primary cells have a positive
gain, a positive 95% matrix-bootstrap lower bound, and Holm-adjusted one-sided
`p < 0.05`.  Otherwise the report retains the narrower claim against the
registered direct ridge and states the sequence-baseline result exactly.

