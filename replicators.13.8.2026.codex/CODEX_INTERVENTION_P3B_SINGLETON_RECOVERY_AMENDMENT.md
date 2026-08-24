# P3b singleton-state recovery amendment

Status: this amendment and its executable source must be checksum-sealed before
the interrupted P3b campaign resumes or any completed branch outcome is loaded
for scientific inference.

## Failure being recovered

The prospectively sealed P3b campaign stopped during primary generation after
363 of 960 restored states.  It had written 81,312 of 215,040 primary-future
checkpoints and had not started replay, inference, or result reporting.  The
next registered case was `INTP3B_DOSE_BRIDGE_V1-c02-m030-g060`.

The restored state had fewer than two occupied molecule types.  Targeted
present-present multiplication is algebraically possible for a one-entry
block, but the registered random control is not: a nonzero, zero-mean log-space
direction cannot exist in one dimension.  The runner correctly stopped instead
of silently changing the random-control location or dropping the state.

The exact exception was:

    ValueError: P3b surgery requires at least two present types

## Frozen recovery rule

This is an additive procedural amendment.  It does not edit the sealed P3b
source, registration, seeds, eligible-state interventions, endpoint, branches,
cohort, inference, or gates.

For every restored state with fewer than two occupied types, regardless of
candidate, matrix, landmark, or position in the cohort:

1. retain the state and its matrix in every analysis;
2. replace every registered arm by an unchanged-beta action at that state;
3. launch all seven arm-labelled futures under the originally registered,
   arm-free common random stream;
4. require predictions and realized records to be bitwise identical across all
   seven arms for each branch;
5. give the state a zero paired contribution to every targeted, dose, and
   random-control contrast; and
6. report it explicitly as a `STRUCTURAL_NO_ACTION` state.

This is an intention-to-treat no-action rule.  It is conservative for the
targeted causal contrast because it cannot create a positive targeted effect.
It also avoids fabricating a one-edge "balanced random" operation, moving the
random surgery outside the present-present block, excluding the state, or
replacing its matrix.

The same universal rule applies to all later singleton or empty restored states
encountered in the already frozen cohort.  Their prevalence is not used to
change the rule.

## Checkpoint and outcome firewall

Before sealing this amendment:

- hash the original P3b registration and sealed sources;
- hash the original generation checkpoint contract;
- hash, by filename, the immutable prefix `state_0000.pkl` through
  `state_0362.pkl` without deserializing any checkpoint;
- record that replay and final result artifacts do not exist;
- run only non-scientific fixtures and repository tests; and
- do not compute event rates, arm means, causal contrasts, confidence
  intervals, p-values, or candidate differences.

After sealing:

- rebuild the same deterministic natural-state cohort and verify every state ID
  and state digest against the original checkpoint contract;
- reuse the 363 existing generation checkpoints byte-for-byte;
- generate only missing primary checkpoints with the amended worker;
- generate a complete 960-state replay with the amended worker;
- require complete exact replay across all states and arms;
- audit every structural no-action state and every eligible-state surgery;
- run the unchanged P3b whole-matrix inference and gates; and
- seal the result and stop without launching another phase.

No matrix, state, landmark, arm, branch, or adverse outcome may be removed or
replaced.  Intervention futures remain without retries.

## Claim boundary

The recovery makes P3b's natural-cohort estimand explicit: it measures the
registered intervention policy, including no action where its matched balanced
random control is structurally undefined.  Any passing result must disclose the
number and identity of structural no-action states.

This procedural recovery cannot turn P3b into evidence for life, biological
memory, autonomy, agency, error correction, real prebiotic chemistry, Phi or
PhiID intervention, strict-eight control, or a universal origin-of-life
mechanism.
