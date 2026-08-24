# Codex CR9M launch-state moderation preregistration

## Status and scientific boundary

CR9 is sealed and remains a failed registered accumulating-hysteresis test for
natural generation-60 launch states. CR9M is a new, prospective moderation
experiment motivated by a post-result source audit of the independent Fable G3
implementation. It cannot rescue, replace, or reinterpret CR9.

The scientific question is:

> Does the relationship between steering duration and transient post-release
> persistence depend on whether steering starts from a nascent, diffuse assembly
> or an already evolved assembly, and is that moderation robust to the differing
> pulse/anchor conventions used by the two clean rooms?

CR9M targets transient compositional persistence after active `MODEL_DOWN`
steering. It does not target strict-eight, does not alter `JOINT_BREAK_RUN3`, and
does not test autonomous restoration, biological memory, agency, life, or a
universal origin-of-life mechanism.

## Frozen upstream objects

Use Codex's unchanged candidate-02 and candidate-03 simulator contracts and the
immutable 5x-development candidate-separated `JOINT_BREAK_RUN3` predictor used
by CR1, CR7, CR8, and CR9. Use the Codex exhaustive legal mass-preserving
one-molecule `MODEL_DOWN` selector with first-lexicographic tie resolution.

Do not import Fable code, models, matrices, states, seeds, selected edits, or
result objects. The externally described Fable convention is implemented anew
with Codex machinery as a frozen protocol factor. There is no refitting,
recalibration, threshold search, selector change, or candidate-specific rescue.

Before scientific generation, reproduce and hash the frozen model, verify the
sealed CR9 registration and result, hash this registration and source tree, and
complete the non-scientific validation and smoke suite.

## Fresh paired cohort

Generate 48 entirely new catalytic matrices and Codex-native initial
compositions. Each matrix and initial composition is shared across both
candidates. For each candidate and matrix construct two paired launch states:

1. `NASCENT`: the untouched generation-0 Codex initial composition, with empty
   observed inheritance history and zero observational clocks.
2. `MATURE`: the natural untreated post-fission generation-60 descendant of that
   same initial composition under that candidate's contract.

The mature main path may use only the already frozen restoration retry contract.
No scientific pulse or release lineage is retried, replaced, or dropped.

Use three replicate lineages per candidate, matrix, launch state, pulse
convention, and pulse length. The catalytic matrix is the inference unit.

## Registered 2 x 2 factorial

Cross launch state (`NASCENT`, `MATURE`) with pulse convention:

1. `RELAXED`: for pulse length P, simulate P fissions, apply `MODEL_DOWN` after
   successful fissions 1 through P-1, and use the final unedited daughter as the
   written anchor. P=1 therefore applies exactly zero edits.
2. `POST_EDIT`: for pulse length P, simulate P fissions, apply `MODEL_DOWN` after
   every successful fission 1 through P, and use the final post-edit state as the
   written anchor.

Use pulse lengths:

`P in {1, 2, 4, 8, 16, 32, 60}`.

After the anchor is formed, disable the controller completely and simulate 60
untreated release fissions. Release invokes no callback and applies exactly zero
interventions.

The same frozen Codex predictor and exhaustive selector are used in all four
factorial cells. Thus CR9M isolates launch maturity and pulse/anchor convention;
it does not confound those factors with the Fable marginal selector.

## Randomness

Seal purpose-separated streams for matrix generation, initial composition,
mature main trajectory, pulse/release futures, bootstrap, randomization, replay,
validation, and smoke testing.

For a fixed candidate, matrix, and replicate, the future seed excludes launch
state, convention, and pulse length. All factorial cells and pulse lengths begin
from common random streams. Once their states differ they may consume random
values differently; these are common random streams, not identical realized
futures. The mature main-trajectory stream is separate from the intervention
future stream.

## Endpoint and adverse handling

The written anchor is the composition specified by the registered convention.
At every release boundary compute unrounded float64 cosine similarity to that
anchor. Persistence is the first release fission at which similarity is strictly
below 0.7.

- A complete F60 release without departure receives the right-censor cap 61.
- If the pulse becomes incomplete, release does not start and persistence is 1.
- If release becomes incomplete before departure, persistence is the first
  unobserved registered release boundary.
- A departure already observed before later incompleteness remains observed.

No incomplete or extinct row is dropped. A survivor-only analysis may be
reported solely as an explicitly non-primary comparison with the historical
Fable calculation.

Track release risk, strict inheritance (`H > 0.9`), growth updates, entropy,
top-one share, occupied types, catalytic throughput `x^T beta x`, completion,
and final similarity. Track launch and anchor values and their changes for risk,
entropy, top-one share, occupied types, and throughput.

## Primary estimand and gate

For every candidate, matrix, launch state, and convention, average persistence
over the three replicates at each pulse length. Compute one seven-point Spearman
correlation between pulse length and mean persistence. A constant persistence
vector has registered Spearman zero.

For each matrix define the launch-moderation contrast:

```
0.5 * [rho(NASCENT, RELAXED)  - rho(MATURE, RELAXED)
     + rho(NASCENT, POST_EDIT) - rho(MATURE, POST_EDIT)]
```

Evaluate candidates separately. The registered launch-moderation gate passes
only if both candidate cells have:

1. mean contrast greater than zero;
2. 95% whole-matrix bootstrap lower bound greater than zero;
3. Holm-adjusted one-sided whole-matrix sign-randomization p < 0.05;
4. complete exact replay and artifact readback.

Use 4,096 whole-matrix bootstrap draws and 4,096 paired whole-matrix sign
randomizations. Holm adjustment for the primary family is across the two
candidate cells. Candidates are never pooled to rescue disagreement.

## Registered supporting classification

`Protocol-robust nascent hysteresis` requires positive mean Spearman, positive
95% bootstrap lower bound, and Holm-adjusted randomization p < 0.05 in all four
candidate-by-convention `NASCENT` cells. This classification is separate from
the primary launch-moderation gate.

Report without using them to rescue either gate:

- all eight candidate-by-launch-by-convention cell estimates;
- P60 minus P1 persistence;
- the relaxed-minus-post-edit convention main effect;
- the launch-by-convention interaction;
- per-matrix effects, sign counts, and maximum leave-one-matrix influence;
- 90% and 95% intervals;
- completion, extinction, censoring, and landmark-specific state summaries.

No confidence interval that merely crosses zero is called equivalence.

## Registered packing diagnostics

The external mechanistic hypothesis is that nascent assemblies have greater
remaining consolidation headroom. For every cell, report matrix-level Spearman
relationships between pulse length and:

- increasing top-one share;
- decreasing entropy;
- decreasing occupied-type count;
- increasing log catalytic-throughput ratio relative to launch;
- decreasing frozen risk.

Also report the association between those anchor changes and persistence. These
are registered mechanism diagnostics, not formal causal-mediation estimates.
They cannot rescue a failed primary gate.

Before scientific execution, test on non-scientific states whether changing
only history clocks and history arrays can change the exact selected
mass-preserving `MODEL_DOWN` edit. Report the result; do not add a clock arm after
outcomes are observed.

## Integrity and stopping

Before scientific generation validate at minimum:

1. Codex-native nascent composition and history construction.
2. Paired mature-state construction from the same beta and initial composition.
3. `RELAXED` edit steps are exactly 1 through P-1.
4. `POST_EDIT` edit steps are exactly 1 through P.
5. P=1 `RELAXED` is a true zero-edit pulse.
6. Both conventions use the exact exhaustive frozen selector.
7. All edits are legal and mass preserving.
8. Anchor construction matches the registered convention.
9. Factor identities and pulse length are absent from future seed keys.
10. Candidate-specific daughter semantics remain unchanged.
11. Release applies exactly zero interventions.
12. Threshold, censoring, incompleteness, and constant-Spearman fixtures.
13. Whole-matrix bootstrap and sign-randomization blocks remain intact.
14. Frozen-model serialization and hash reproduction.
15. Complete deterministic replay of state, action, process, endpoint, and RNG.
16. Exact machine-readable artifact readback.

Generate 8,064 scientific pulse/release lineages and replay all 8,064 exactly.
Seal the result and stop. CR9M does not automatically launch CR10 or any further
moderator search.

## Claim boundary and decision map

If launch moderation and protocol-robust nascent hysteresis both pass, CR9M may
support:

> In Codex's independent reconstruction, longer state-dependent steering
> progressively consolidates nascent diffuse assemblies into a temporarily more
> persistent post-release organization, whereas already evolved assemblies have
> less remaining consolidation headroom.

If only the convention factor separates cells, report that accumulating
hysteresis is protocol-sensitive. If nascent cells fail, report that Codex did
not reproduce the Fable nascent-hysteresis result. If only one candidate passes,
report candidate-specific moderation.

Under every outcome, retain these boundaries: transient persistence is not a
self-restoring basin; active external editing is not autonomous memory; and no
result establishes life, agency, error correction, real prebiotic chemistry, or
a universal origin-of-life mechanism.
