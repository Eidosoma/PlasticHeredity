# Codex CR10 exploratory internalization ladder preregistration

## Status and scope

CR10 is the final, bounded, exploratory phase of the independently registered
Codex JOINT_BREAK_RUN3 intervention program. It may not rescue, replace, or
reinterpret any confirmatory result. In particular:

- CR7 remains the sealed demonstration that repeated external feedback can
  maintain hereditary stability in both Codex simulator candidates.
- CR8 remains the sealed release-and-challenge result, including its registered
  zero restoring-basin radius.
- CR9 remains the failed mature-launch accumulating-hysteresis experiment.
- CR9M remains the separate successful launch-maturity moderation experiment.
- strict-eight occurrence and prediction are outside this program.

No Fable code, states, matrices, seeds, fitted objects, selected actions, or
result files are used. The policies below come only from the already supplied
external protocol and are implemented against Codex's sealed simulator and
frozen JOINT_BREAK_RUN3 predictor.

## Scientific question

Can the successful external control law be reduced to local, physically
interpretable rules, sparse triggers, or a simple retention mechanism while
retaining useful hereditary maintenance and post-disturbance recovery?

This is an internalization *ladder*, not a claim that the simulated assembly is
autonomous. L0--L3 remain external callbacks. The kinetic prototype changes one
registered physical rate and is explicitly a model extension.

## Frozen simulator and target

The unchanged Codex candidate 02 and 03 contracts are used. The primary process
remains JOINT_BREAK_RUN3, with strict inheritance defined by unrounded float64
cosine similarity `H > 0.9`. The main long-lineage outcomes are hereditary
maintenance and recovery, not strict-eight.

All molecular actions are legal mass-preserving one-molecule substitutions.
Observed history is unchanged by the instantaneous action and then evolves
normally.

## Upstream authorization

CR10 may be registered only if checksum verification confirms:

- the full CR3 outgoing physical-rule four-cell gate and replay/readback pass;
- the original CR7 60-fission closed-loop gate, no-op identity, replay, and
  readback pass;
- the frozen 5x JOINT_BREAK_RUN3 predictor has its expected hash and reproduces
  its archived predictions.

CR6, CR8, CR9, and CR9M do not tune CR10 and are not phase-advancement gates.

## L3 development and freezing

Before any CR10 scientific matrix is generated, reconstruct the original 5x
development cohort exactly: 200 matrices, both candidates, and untreated
landmarks 20, 35, 50, 65, and 80. Exact agreement of the reconstructed 195
state/graph coordinates, nine history coordinates, and 195 beta coordinates
with the retained development arrays is mandatory.

For every development state, exhaustively select the already frozen
candidate-specific MODEL_DOWN edit. This selection uses no CR10 outcome.

For each candidate, fit one remove-side and one add-side
`DecisionTreeClassifier` with this fixed architecture:

- `criterion="gini"`;
- `splitter="best"`;
- `max_depth=3`;
- `min_samples_leaf=25`;
- `class_weight="balanced"`;
- deterministic, domain-separated random state.

Each molecule type is one training row. The only four permitted inputs are:

1. abundance share;
2. outgoing catalytic-influence percentile, where outgoing influence is
   `x @ beta == beta.T @ x` under Codex's `beta[target,catalyst]` convention;
3. incoming boost percentile, `beta @ x`;
4. presence indicator.

The selected MODEL_DOWN removal or addition type is the sole positive label for
its corresponding tree. Candidates are never pooled. At use time, L3 chooses
the highest remove score among present types and the highest add score among
all different types, with first numeric type resolving ties. The joint swap is
not rescored by the frozen predictor. Tree arrays, feature order, architecture,
development fidelity, predicted-risk regret, and hashes are frozen before
scientific generation. No refit follows registration.

## Scientific cohorts

### Home-regime maintenance and recovery

- 48 entirely fresh catalytic matrices shared across candidates;
- default beta regime `(A,sigma)=(-4,4)`;
- one untreated natural generation-60 state per candidate and matrix;
- three replicate lineages per policy and condition;
- 60 fissions per lineage;
- seven policies;
- two paired conditions: `UNCHALLENGED` and `CHALLENGED_K8`;
- no controlled-future retry and no matrix replacement;
- complete deterministic replay.

The two conditions use the same future and controller-action streams until the
challenge boundary. In `CHALLENGED_K8`, after fission 30 and after that
boundary's ordinary controller action, exactly eight molecules are transported
to different labels using a separate challenge stream. Control resumes after
fission 31. The perturbation removes eight molecule instances without
replacement and adds eight labels with replacement outside the set of removal
labels, ensuring exact transport distance eight, fixed mass, and nonnegative
integer composition. Because policy states can differ, this is a common random
challenge stream, not necessarily identical realized edit labels.

### Zero-shot regime transfer

Use 24 fresh matrices per regime, both candidates, one untreated natural
generation-60 state, two replicates per policy, 60 fissions, and complete
replay in each of the three externally specified positive regimes:

- `POS_A_M4_S5`: `(-4,5)`;
- `POS_A_M3_S4`: `(-3,4)`;
- `POS_A_M5_S4`: `(-5,4)`.

Only unchallenged maintenance is tested in transfer. Trees, predictor, action
rules, and thresholds remain frozen. Earlier CR6 transfer results remain
unchanged; CR10 transfer is exploratory and cannot rescue them.

## Frozen policies

All callbacks occur after each successful fission, including the last, unless
the sparse trigger is false.

1. `L0_RULE_CONTINUOUS`: apply the outgoing RULE_DOWN substitution after every
   fission.
2. `L1_RULE_AFTER_BREAK`: apply RULE_DOWN only when the just-observed boundary
   is a break (`H <= 0.9`).
3. `L2_RULE_UNTIL_RUN3`: apply RULE_DOWN only when the current trailing strict
   inherited run has length less than three.
4. `L3_LOCAL_TREE`: apply the frozen local-tree edit after every fission.
5. `MODEL_DOWN`: exhaustively apply the frozen predictor minimum after every
   fission.
6. `RANDOM`: apply one uniformly selected legal edit after every fission from a
   stream separated from future simulation and challenge randomness.
7. `NOOP`: apply no molecular edit.

Action count, distinct actions, repeated actions, and immediate reversals are
reported. Sparse policies are not described as superior solely from raw
inheritance without their edit budgets.

## Retention-only kinetic prototype

On the same 48 home-regime matrices and natural generation-60 launch states,
use three fresh replicate streams for 60 fissions under:

`leave_rate(type) *= 1 / (1 + lambda * outgoing_influence_percentile(type))`

for `lambda` in `{0, 0.1, 0.3}`. The percentile is recomputed from the current
composition at every molecular growth update. Joins, fission, daughter choice,
and all other baseline paths are unchanged. Lambda zero dispatches directly to
the plain frozen simulator and must be bitwise identical. No molecular
intervention callback is active in this prototype. This is a retention-only
model extension, not a replacement baseline or evidence that real chemistry
implements the rule.

## Randomness

Domain-separated seeds are frozen for development reconstruction, tree fitting,
home matrices, transfer matrices, initial states, natural trajectories, future
simulation, random policy actions, K8 challenges, kinetic futures, bootstrap,
randomization, smoke, and replay.

For a fixed phase, candidate, matrix, and replicate, the policy name is absent
from the future seed. Paired policies therefore receive common random streams,
not guaranteed identical realized futures after their states diverge. Random
actions and challenges never consume future-simulation randomness.

## Outcomes

For every long lineage report:

- completed horizon and observed fissions;
- inherited-boundary fraction and count;
- total breaks;
- non-overlapping JOINT_BREAK_RUN3 episode count;
- longest inherited run;
- final entropy, occupied types, top-1 share, throughput, and frozen risk;
- action frequency and action diversity;
- growth updates and out-of-development-envelope fraction.

For the K8 condition additionally report, over fissions 31--60:

- post-challenge inherited fraction;
- post-challenge breaks;
- first run3-certification delay, adversely censored at 31 when not observed;
- inherited boundaries in the final six;
- challenge-induced change relative to its paired unchallenged lineage.

For the kinetic prototype report the same maintenance and composition outcomes,
with lambda contrasts against zero.

## Inference

The catalytic matrix is always the resampling and randomization unit. Replicates,
conditions, policies, candidates, and repeated observations from a matrix stay
together. Candidates and transfer regimes are never pooled to rescue a result.

Use 4,096 whole-matrix bootstrap draws and 4,096 paired whole-matrix sign
randomizations. Report candidate-separated means and confidence intervals for:

- every policy minus NOOP under home maintenance;
- every policy minus NOOP after K8 challenge;
- challenged minus unchallenged within policy;
- local-policy recovery of the MODEL_DOWN gain;
- RANDOM minus NOOP with a descriptive TOST interval of `+/-0.025`;
- every policy minus NOOP in each transfer regime;
- lambda 0.1 and 0.3 minus lambda 0.

Holm-adjust randomization p-values within each reported analysis family.
Ratios whose MODEL_DOWN-minus-NOOP denominator is nonpositive are marked
undefined rather than repaired. Both observed-boundary fractions and fixed
registered-horizon fractions are retained; registered maintenance inference
uses the fixed horizon, so failed lineages remain adverse and unobserved
boundaries are never credited as inherited.

CR10 has no confirmatory pass/fail gate. Positive lower confidence bounds are
descriptive exploratory signals only. No CR10 outcome changes an earlier gate.

## Validation, replay, and artifacts

Before registration, validate at least:

- exact development-state reconstruction and frozen-model replay;
- permitted local feature order and permutation equivariance;
- candidate separation, tree depth, deterministic serialization, and legal L3
  tie handling;
- exact L0/L1/L2 triggers and outgoing orientation;
- uniform legal RANDOM selection and stream separation;
- exact K8 mass, nonnegativity, transport distance, history preservation, and
  stream separation;
- arm-free future streams;
- NOOP callback/plain simulator bitwise identity;
- lambda-zero kinetic/plain simulator bitwise identity;
- strict endpoint fixtures;
- whole-matrix bootstrap and sign-randomization blocks;
- exact state, action, challenge, endpoint, process, and RNG replay;
- checksum and machine-readable artifact readback.

Persist the development freeze, selected actions, state/matrix/lineage tables,
trajectory arrays, inference arrays or deterministic seeds, replay audit,
scientific report, lay summary, claim boundaries, manifest, checksums, and an
updated cumulative intervention ledger. Large regenerable arrays and work
checkpoints are ignored by git.

## Operational and claim boundaries

The campaign is resumable by immutable per-case checkpoints and stops after
sealing CR10. It does not launch a new search or phase.

A positive result may support only the exploratory statement that a substantial
fraction of Codex's externally demonstrated hereditary control can be expressed
by simple local rules, sparse triggers, and/or a retention-only model extension.

It may not establish autonomous agency, biological memory, error correction,
life, real prebiotic chemistry, a universal origin-of-life mechanism, Phi or
PhiID, control of strict-eight, or an installed compotype. A null kinetic result
tests one retention-only embodiment and cannot establish that chemical
internalization is impossible.
