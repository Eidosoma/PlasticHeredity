# CR9M launch-state moderation

CR9 remains sealed and unchanged. CR9M primary launch-moderation gate: **True**.
Protocol-robust nascent-hysteresis classification: **True**.
Complete registered gate including integrity: **True**.

## Primary launch-moderation contrasts

### Candidate 02

- Fresh-minus-mature mean matrix Spearman: +0.347906.
- 95% whole-matrix bootstrap CI: [+0.199997, +0.494559].
- 90% whole-matrix bootstrap CI: [+0.222984, +0.469741].
- Holm-adjusted one-sided matrix-randomization p: 0.000488162.
- Candidate primary gate: **True**.

### Candidate 03

- Fresh-minus-mature mean matrix Spearman: +0.361605.
- 95% whole-matrix bootstrap CI: [+0.197673, +0.523461].
- 90% whole-matrix bootstrap CI: [+0.223549, +0.495623].
- Holm-adjusted one-sided matrix-randomization p: 0.000488162.
- Candidate primary gate: **True**.

## Factorial cell correlations

- Candidate 02 / NASCENT / RELAXED: rho +0.507587, CI95 [+0.407757, +0.599962], P60-P1 +4.736.
- Candidate 02 / NASCENT / POST_EDIT: rho +0.462079, CI95 [+0.347091, +0.566183], P60-P1 +4.910.
- Candidate 02 / MATURE / RELAXED: rho +0.078735, CI95 [-0.055055, +0.216169], P60-P1 -0.340.
- Candidate 02 / MATURE / POST_EDIT: rho +0.195119, CI95 [+0.057329, +0.328862], P60-P1 +0.299.
- Candidate 03 / NASCENT / RELAXED: rho +0.514576, CI95 [+0.415606, +0.604804], P60-P1 +5.674.
- Candidate 03 / NASCENT / POST_EDIT: rho +0.549981, CI95 [+0.438905, +0.649444], P60-P1 +5.743.
- Candidate 03 / MATURE / RELAXED: rho +0.185970, CI95 [+0.028868, +0.334423], P60-P1 -1.097.
- Candidate 03 / MATURE / POST_EDIT: rho +0.155377, CI95 [+0.033138, +0.278204], P60-P1 -0.007.

## Convention and interaction diagnostics

- Candidate 02 relaxed-minus-post-edit: -0.035438 [-0.120106, +0.047338].
- Candidate 02 launch×convention interaction: +0.161892 [-0.005771, +0.326250].
- Candidate 03 relaxed-minus-post-edit: -0.002406 [-0.080654, +0.076455].
- Candidate 03 launch×convention interaction: -0.065997 [-0.217947, +0.098155].

## Integrity and claim boundary

- Exact replay: **True**.
- Release interventions exactly zero: **True**.
- Artifact readback exact: **True**.
- Registered packing diagnostics are included in `primary_metrics.json`; they cannot rescue the primary gate.
- CR9M tests transient consolidation, not an autonomous restoring basin or installed biological memory.
