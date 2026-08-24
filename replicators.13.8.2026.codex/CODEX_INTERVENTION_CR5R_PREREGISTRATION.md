# CR5R preregistration: shared-break resilience confirmation

Status: this additive protocol is frozen before any CR5R scientific matrix is
generated. It does not amend, reopen, rescue, or reinterpret sealed CR5.

## Prior result and question

CR5 established predictor-guided causal control of first-break resistance. Its
registered resilience stage was not launched because only 177 of 200 matrices
per candidate yielded an eligible natural break, whereas CR5 required all 200.
That outcome remains `inconclusive_incomplete_matrix_coverage`.

CR5R asks one narrower question prospectively: starting every arm from the
exact same naturally broken daughter, can a single mass-preserving molecular
substitution selected by the already frozen Codex renewal student change the
probability of rebuilding three consecutive inherited fissions within F8?

## Frozen simulator, endpoint, and model

The sealed Codex candidate 02 and 03 simulator contracts are unchanged.
Inheritance is the strict, unrounded float64 comparison `H > 0.9`.

The endpoint is run3 within eight fissions from an already observed natural
post-break daughter. Certification before later extinction remains positive;
extinction before certification is negative. This is not the strict-eight
endpoint and it does not condition separately on treatment-created breaks.

CR5R copies the candidate-separated `q_R` students from the sealed CR5
confirmation registration. Their transforms, PCA, coefficients, ridge
penalties, priors, and prediction mapping must reproduce exactly. There is no
refitting, recalibration, threshold change, model search, or use of any CR5R
outcome in edit selection.

## Cohort and eligibility

- 250 completely fresh catalytic matrices shared across candidates;
- both Codex candidates;
- untreated natural landmarks 20, 35, 50, 65, and 80;
- 2,500 untreated landmark sources;
- each source advanced without intervention for at most 60 fissions;
- the first strict natural break is saved as its exact selected post-break
  daughter, together with the pre-break parent anchor;
- extinct and no-break sources remain in the acquisition ledger;
- no source or matrix is retried, replaced, filtered by risk, or selected using
  a future outcome.

Eligibility is assessed separately by candidate. Intervention futures launch
only if each candidate has at least 200 distinct matrices with one or more
eligible broken daughters. If this threshold is missed, CR5R is sealed
inconclusive and no intervention future is generated. If it is met, every
eligible broken daughter is retained; CR5R does not select only 200 matrices or
one preferred state per matrix. Candidate cohorts are not pooled and need not
contain identical eligible matrix IDs.

## Intervention arms

Every legal one-molecule substitution is scored exhaustively with the frozen
candidate-specific `q_R` student. Ties use the first legal edit in the frozen
enumeration order. The arms are:

1. `RENEWAL_UP`: largest frozen predicted increase from no-op;
2. `RENEWAL_DOWN`: largest frozen predicted decrease from no-op;
3. `RANDOM`: one uniform legal edit from a separate selection stream;
4. `NOOP`: the identical unedited broken daughter.

All edits preserve total mass, require a present removed type, add a distinct
type, and hold already observed history fixed at the instant of editing.

Each eligible state receives 64 F8 futures per arm. Fixed halves are A =
branches 0--31 and B = branches 32--63. Arm identity is absent from the future
seed, so paired arms use common random streams, not necessarily identical
realized futures after divergence. Random edit selection never consumes a
future stream. Every future is generated a second time for exact replay.

## Outcomes and inference

Primary outcome: run3 within F8 from the shared naturally broken daughter.

Registered secondary outcomes: run5, time to run3, inherited-boundary count,
similarity to the pre-break parent, survival, growth updates, final entropy,
and occupied molecular types.

The catalytic matrix is the inference unit. All eligible landmarks and broken
states belonging to a matrix travel together. Because eligibility is
candidate-specific, bootstrap indices and sign vectors are generated at the
eligible-matrix width of each candidate and reused across its two branch
halves. There are 4,096 whole-matrix bootstrap draws and 4,096 paired
whole-matrix sign randomizations. Holm correction covers the four
candidate-by-half primary cells.

CR5R passes only if all four cells have:

1. mean paired `RENEWAL_UP - RENEWAL_DOWN > 0`;
2. a 95% whole-matrix bootstrap lower bound above zero;
3. Holm-adjusted one-sided matrix-randomization `p < 0.05`;
4. `RANDOM` equivalent to `NOOP` under a prewritten TOST margin of
   `+/-0.025`, implemented as a 90% bootstrap interval strictly inside the
   margin; and
5. absolute `RANDOM - NOOP` no greater than 25% of
   `RENEWAL_UP - RENEWAL_DOWN`.

`RENEWAL_UP - NOOP` and `NOOP - RENEWAL_DOWN` are reported but are not gates.
No pooling can rescue a failed candidate or half. Fable's approximately
`+0.026/+0.027` result is post-seal comparison context only, not a fitting
target, margin, or pass threshold.

## Integrity, resources, and stop

Before scientific generation, the complete validation suite, source hashes,
protocol, seed domains, predecessor checksums, frozen-model hashes, endpoint
checks, inference checks, and a non-scientific smoke test must be sealed.

The campaign is checkpoint-resumable, requires at least 4 GB free at launch,
and accepts a declared CPU budget from 8 through 14 hours. The expected run is
about 8--10 CPU hours, so the cumulative CR5 plus CR5R program remains below
the requested 30 CPU-hour ceiling. The exact realized CPU time is reported by
the detached runner.

CR5R stops after its sealed report. It does not automatically launch CR6,
closed-loop steering, transfer, or any other phase.

## Claim boundary

A pass supports causal molecular control of short-run hereditary recovery from
an identical naturally broken state in both independent Codex simulator
candidates. A failure is a failed registered recovery-control prediction. An
eligibility shortfall is inconclusive, not negative evidence.

CR5R cannot establish biological repair, biological memory, agency, life,
autonomous organization, an installed attractor, strict-eight control, real
prebiotic chemistry, a universal origin-of-life mechanism, or Phi/PhiID
intervention.
