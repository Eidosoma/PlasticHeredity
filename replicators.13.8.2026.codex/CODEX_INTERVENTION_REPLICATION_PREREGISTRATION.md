# Codex JOINT_BREAK_RUN3 intervention preregistration

Status: protocol implementation complete; no scientific intervention matrix had been generated when this document was written.

This is a new clean-room causal-intervention program. It is separate from the closed strict coherent-eight occurrence and prediction program. Its only endpoint is the already validated Codex `JOINT_BREAK_RUN3` process: within twelve future fissions, a strict inheritance break (`H <= 0.9`) is followed strictly later by three consecutive inherited fissions (`H > 0.9`). An uninterrupted inherited run does not qualify. Certification before later extinction remains positive; extinction before certification is negative.

## Independence and frozen inputs

The program uses only the two sealed Codex simulator candidates, Codex feature code, Codex states and matrices, new Codex seed domains, and the candidate-separated full predictor from the successful 5× development run. The frozen archive must retain SHA-256 `9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af`, and its portable predictions must reproduce the archived implementation within `1e-12`. There is no refitting, recalibration, regularization search, threshold change, predictor-family selection, or use of the failed strict-eight predictor.

Fable code, models, matrices, states, seeds, selected edits, controllers, and results are not imported. Fable values may be considered only as descriptive benchmarks after the corresponding Codex result has been sealed.

Candidate 02 retains exposure 0.10, whole-assembly overshoot trimming, fixed-size fission, and continuation of the first daughter. Candidate 03 retains exposure 0.125, admission only to remaining capacity, binomial fission, and continuation of the second daughter. Results from the candidates may not be pooled to rescue disagreement.

## Registered edit and randomness contracts

A molecular intervention removes one molecule of a present type and adds one molecule of a different type, preserving total mass. The instantaneous edit changes composition only; all already observed history and clock values remain fixed. Every legal edit is enumerated for the model-guided arm, using a batched algebraic evaluation of the unchanged 195-coordinate feature map. Scalar-versus-batched agreement is a mandatory validation.

For a given phase, candidate, matrix, landmark, and branch number, all arms start with the same purpose-keyed future seed. Arm identity is absent from that seed. These are common random streams, not a promise of identical realized futures after paths diverge. Random edit selection and random beta surgery use independent streams and never consume future-simulation randomness. Matrix generation, natural trajectories, selection, futures, bootstrap, randomization, replay, and later controller actions use distinct registered domains.

No intervention future is retried. Natural main trajectories use only the existing Codex 100-attempt extinction/retry contract. Matrices are never replaced after their seed list is sealed. A complete second execution must reproduce selected interventions, endpoint values, process outcomes, final states, and full fission-record digests.

## Inference

The catalytic matrix is always the inference unit. Its candidates, landmarks, arms, and fixed branch halves remain together in every draw. Major one-shot phases use 4,096 shared whole-matrix bootstrap draws and 4,096 shared paired whole-matrix sign randomizations. Raw randomization p-values are Holm-adjusted across candidate 02 half A, candidate 02 half B, candidate 03 half A, and candidate 03 half B.

The full registered cell gate requires:

1. targeted up minus targeted down is positive;
2. its 95% whole-matrix bootstrap lower bound is positive;
3. its Holm-adjusted randomization p-value is below 0.05;
4. targeted up minus no-op has a positive bootstrap lower bound;
5. no-op minus targeted down has a positive bootstrap lower bound;
6. random is equivalent to no-op by a `+/-0.025` TOST margin and its absolute point difference is no more than 25% of the targeted up-minus-down effect; and
7. exact replay passes.

Prediction log loss is ordinary branch-level Bernoulli log loss. Reported Brier scores are ordinary branch-level squared errors. No state-fraction squared error is called a proper Brier score.

Because the three initial stages are explicitly developmental pilots, the separate pilot-eligibility rule is narrower: up-minus-down must be positive and random-minus-no-op must lie within `+/-0.025` in all four cells, with exact replay. The complete gate is still reported. Pilot eligibility is not confirmatory evidence.

## Budgeted serial program

The elapsed-runtime goal is less than 30 hours on the current 14-worker machine. This is not a mid-phase kill rule: a started phase completes, replays, seals, and reports before stopping. No subsequent scientific phase starts automatically.

### P1 — CR1 model-guided molecular pilot

- 40 entirely new matrices shared across candidates.
- Natural restored landmarks 20, 35, 50, 65, and 80.
- Arms: `MODEL_UP`, `MODEL_DOWN`, `RANDOM`, and `NOOP`.
- 32 F12 branches per arm per state; halves A=0–15 and B=16–31.
- 400 states, 51,200 primary futures, and 51,200 replay futures.
- Every legal one-molecule substitution is scored by the frozen predictor. Exact maxima/minima are chosen with deterministic lexicographic tie resolution.
- Mandatory stop after the technical report, lay summary, ledger update, and checksum seal.

### P2 — CR3 physical catalytic-support-rule pilot

Run only after a new user instruction following review of P1. Use a new 40-matrix cohort with the same landmarks and branch counts. Under Codex storage `beta[target,catalyst]`, catalytic support is `beta @ x`. `RULE_DOWN` removes the least-supported present type and adds the most-supported type; `RULE_UP` reverses that choice. Include uniform random and no-op arms. Stop and seal before P3.

### P3 — CR4 fixed-composition beta-surgery pilot

Run only after a new user instruction following review of P2. Use another new 40-matrix cohort. Composition and history are identical across arms. On the present–present beta block, `TIGHTEN` and `LOOSEN` apply positive common multiplicative changes with exact Frobenius norm `0.05 * ||beta[P,P]||_F`. `RANDOM_SURGERY` changes the same number of independently selected edges with a balanced log perturbation, positivity, and the same norm. Include no-op. Stop and seal before any mechanism selection.

### Untouched chosen-mechanism confirmation

After all three pilots, the user may choose exactly one mechanism for a separately versioned confirmation. That registration freezes the family without changing its pilot algorithm and uses 160 entirely new matrices, both candidates, five landmarks, four arms, 32 F12 branches per arm, and complete replay: 204,800 primary plus 204,800 replay futures. Pilot outcomes are not confirmation data.

Only after a successful untouched mechanism confirmation may compact dose-response, closed-loop maintenance, and steer–release experiments be registered. The full resistance/resilience decomposition, parameter-regime transfer, half-life ladder, and internalization ladder are deferred from the current compute budget.

The three pilots plus the selected confirmation contain 716,800 F12 futures including replay. Disposable benchmarks on this machine put the expected elapsed time comfortably below 30 hours; uncertainty is dominated by simulator path length and filesystem/checkpoint overhead. The phase boundaries, not an optimistic ETA, govern execution.

## Mandatory pre-scientific checks

The executable CR0 suite covers the 24 checks specified in the research directive: molecular legality, mass and history invariance, permutation invariance, frozen-model persistence, exact exhaustive enumeration and tie behavior, random-arm uniformity, stream separation and arm pairing, no-op identity, endpoint threshold/horizon/extinction fixtures, daughter semantics, matrix-block inference, replay, beta-surgery positivity and norms, callback identity, and zero action after release. It adds two audits: real-dimension scalar/batched feature and predictor agreement, and explicit verification that Codex’s propensity equation uses `beta @ x`.

A non-scientific smoke may exercise I/O, checkpoints, selection, futures, and replay. Its durable output may not reveal effect sizes, event rates, candidate ordering, or arm ordering.

## Claim boundary

A successful untouched one-shot confirmation could show that small targeted changes causally alter the probability of an operational break-and-renewal event in these independently reconstructed GARD dynamics under common random streams. A successful simple-rule result could show that current catalytic support captures part of the control law. Successful feedback could show externally maintained organization while the controller acts.

No stage may be described as evidence for Phi/PhiID intervention, strict-eight control, autonomous agency, biological memory, error correction, a living organism, real prebiotic chemistry, or a universal origin-of-life mechanism. A pilot cannot establish cross-clean-room replication. If later release does not preserve and restore the written state, the permitted term is “controller-maintained compotype-like state,” not “installed compotype.”

