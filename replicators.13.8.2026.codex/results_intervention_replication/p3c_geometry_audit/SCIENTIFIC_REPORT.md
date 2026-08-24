# P3c post-hoc geometry audit

This is an exploratory analysis of the sealed P3b data, not a new simulation and not a repair of P3b's failed specificity gate.

## Main finding

- c02 half A: the high-dose balanced-log random arm changed mean log throughput by +0.01063 and JOINT_BREAK_RUN3 by -0.01250; the state-centred throughput slope was -0.14530 (95% matrix bootstrap (-0.17412897464953248, -0.1177064069471482)).
- c02 half B: the high-dose balanced-log random arm changed mean log throughput by +0.01063 and JOINT_BREAK_RUN3 by -0.01672; the state-centred throughput slope was -0.12592 (95% matrix bootstrap (-0.1572776771914544, -0.09787838268136434)).
- c03 half A: the high-dose balanced-log random arm changed mean log throughput by +0.03769 and JOINT_BREAK_RUN3 by -0.01484; the state-centred throughput slope was -0.13183 (95% matrix bootstrap (-0.15914078646190194, -0.10545534242732064)).
- c03 half B: the high-dose balanced-log random arm changed mean log throughput by +0.03769 and JOINT_BREAK_RUN3 by -0.01328; the state-centred throughput slope was -0.11825 (95% matrix bootstrap (-0.14798679526248992, -0.09049168378458047)).

A log-balanced perturbation is not automatically neutral in ordinary catalytic throughput: exponentiating positive and negative log changes can raise weighted arithmetic support even when their unweighted log sum is zero. This is the prospective rationale for P3c's throughput-neutral control.

## Claim boundary

These associations are post-hoc. They can explain why the old random arm moved the outcome, but cannot prove mediation or turn P3b into a formal pass. That requires P3c's fresh pilot and untouched confirmation.
