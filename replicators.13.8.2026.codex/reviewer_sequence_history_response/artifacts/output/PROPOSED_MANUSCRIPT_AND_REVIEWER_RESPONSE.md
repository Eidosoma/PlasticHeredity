# Proposed manuscript and reviewer-response language

## Methods addition

As a reviewer-prompted post-hoc robustness analysis, we compared the frozen F12
composite with development-fitted sequence-history models.  Candidate-specific
first-order and duration-aware transition laws were integrated over the F12
endpoint before observing each retained confirmation future.  A stronger
history comparator augmented the registered direct variables with ordered
pre-launch continuous H values, strict-H indicators, and padding masks; lag
length and ridge strength were selected by development-matrix-grouped
cross-validation.  All models were then scored without recalibration on the
already-observed confirmation branches, separately by candidate and frozen
half.

## Results addition

The frozen composite improved branch log loss over the selected ordered-history
model in all eight implementation-by-candidate-by-half cells; every
whole-matrix 95% interval excluded zero and every Holm-adjusted paired
randomization p value was below 0.05.  Thus, within the tested model family, the
composite advantage was not explained solely by first-order, duration-aware, or
ordered recent inheritance history.  Because this analysis was prompted after
the confirmation results existed, it is supportive post-hoc evidence rather
than a new prospective confirmation.

## Reviewer response

We agree that Appendix C motivated a stronger launch-time history baseline.  We
added both generative Markov/semi-Markov controls and an ordered-history ridge,
fit exclusively on development matrices.  These were converted to or directly
estimated as F12 launch probabilities and rescored on the existing confirmation
outcomes; no new futures were generated.  The frozen composite retained its
advantage in every primary clean-room cell.  We also clarified that Appendix C
uses bits per transition after part of the future is observed, whereas the
headline task uses nats per complete future at launch.
