# Self-replicator label contract: unresolved gate

Status: unresolved as of 2026-08-21.

The original clean-room reconstruction uses the standard GARD cosine/H
similarity to a dominant recurring-composition medoid, with a fixed threshold
of 0.95 and at least three recurrences. That detector produced a mean control
probability of 16.7% in the original 100-run reconstruction, while the paper
reported 88%.

The separately registered 12-seed formulation pilot made the discrepancy more
severe: its mean molecular-observation label rate was 5.58%, with one
constant-zero trajectory. This is why a training-only majority classifier was
94.62% accurate while all four information histories were near 50%.

No local fixture or released author implementation currently establishes that
the reconstruction's medoid, reference-state set, similarity function,
threshold, recurrence rule, or molecular-time alignment matches the authors'
detector. Matching the reported 88% by lowering the threshold would be outcome
calibration, not resolution, and is prohibited.

Accordingly:

- the original negative replication remains unchanged;
- the completed formulation pilot remains explicitly a
  reconstruction-label result;
- the covariance-support phase may proceed because it is forbidden from
  importing, computing, or reading replicator labels; and
- a new replicator-outcome pilot is locked until either author code/fixtures
  establish label parity or a human explicitly authorizes a provisional-label
  study after reviewing a numerically stable estimator.

When author material arrives, the label gate requires shared fixtures covering
the reference composition, reference candidate states, support/recurrence
count, similarity at and around the decision boundary, and the exact label
sequence for at least one complete trajectory. Aggregate agreement alone is
not sufficient.
