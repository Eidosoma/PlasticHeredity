# Plastic heredity in cellular automata — clean-room reconstruction

This repository independently reconstructs the cellular-automata results that
transfer the Plastic Heredity observer from simulated molecular assemblies to
noisy elementary cellular automata (ECA) and Conway's Life.

The implementation was written from the Plastic Heredity preprint, the sibling
experiments' written contracts and retained result tables, and a later
code-free golden-trace pack. No sibling `src/`, `tests/`, or executable script
was opened or reused. See
[`CLEANROOM_PROTOCOL.md`](CLEANROOM_PROTOCOL.md) for the evidence boundary and
the few choices that the public result record leaves unresolved.  The completed
reference-scale findings and gate-by-gate adjudication are in
[`DISCOVERY_LOG_EIDOSOMA_SCIENTIST.md`](DISCOVERY_LOG_EIDOSOMA_SCIENTIST.md).

## What is reconstructed

- all 256 ECA rules and their 88 reflection/conjugation orbits;
- activity-gated noisy lineages on a periodic ring;
- the terminal 4-mer (raw-texture) observer;
- local inheritance, break-by-8, and strict coherent-eight plastic heredity;
- the ECA atlas and noise-phase experiments;
- the figure/ground particle observer based on a rule-specific 3x3
  spatiotemporal domain dictionary;
- a Life-like bitboard engine, literature fixtures, and the named-object versus
  density-matched-random comparison;
- an evolutionary search on the eight-bit ECA rule hypercube; and
- a fresh-truth GPS test on that hypercube;
- a 1,024-rule Life-like B/S atlas with activity-clock, launch, horizon, and
  spatial-scale stress tests; and
- explicit comparison to the permitted sibling result data.

The GARD↔ECA↔XENO catalogue bridge is deliberately excluded.  It tests a
downstream cross-formalism dictionary and requires frozen GARD/XENO embeddings;
it is not part of the CA heredity simulator itself.

## Golden-trace result

The reconciled engine exactly reproduces the retained 88-rule ECA atlas:
all four numerical endpoints and every rule's form-library IDs match for all
88 rules. The independent prerequisite also replays all 907 disclosed sweeps
and 15 final4 spectra bit-for-bit. The five-point phase grid has the same
registered gate outcomes and very high rank agreement, but not identical
Monte Carlo cells because its fresh RNG stream was not disclosed.

The complete run is in
[`results/golden-reconciliation/`](results/golden-reconciliation/), with a
compact [`REPORT.md`](results/golden-reconciliation/REPORT.md) and full
machine-readable [`RESULTS.json`](results/golden-reconciliation/RESULTS.json).

## Quick start

The exact engine pins NumPy 2.5.2. Use an isolated environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.5.2
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m plastic_ca golden-suite \
  --workers 16 \
  --reference-root .. \
  --output results/golden-reconciliation \
  --resume
```

Useful profiles are `smoke` (fast wiring check), `standard` (independent
reproduction), and `reference` (16 seeds x 128 futures, matching the retained
campaign's Monte Carlo scale).  Every result file records the complete
simulation contract, seed namespace, and clean-room semantic choices.

Run `.venv/bin/python -m plastic_ca --help` for the experiment commands.

The next-round E23/E24 campaign is frozen in
[`CA_CAMPAIGN_PROTOCOL.md`](CA_CAMPAIGN_PROTOCOL.md). A complete reference run
is launched or resumed with:

```bash
.venv/bin/python -m plastic_ca ca-campaign \
  --profile reference \
  --workers 16 \
  --reference-root .. \
  --dev-atlas results/golden-reconciliation/atlas/eca_rules.csv \
  --output results/ca-campaign-round-1 \
  --resume
```

The campaign checkpoints each rule and condition. During a detached run,
`results/ca-campaign-round-1/STATUS.json` reports the active stage, completed
rule count, throughput, and ETA without requiring an active polling loop.

The frozen semantic-reconciliation campaign is documented in
[`SENSITIVITY_PROTOCOL.md`](SENSITIVITY_PROTOCOL.md) and can be launched or
resumed with:

```bash
python3 -m plastic_ca sensitivity \
  --design overnight \
  --workers 16 \
  --reference-root .. \
  --output results/sensitivity-round-1 \
  --resume
```

The causal follow-up is preregistered in
[`CAUSAL_HEREDITY_PROTOCOL.md`](CAUSAL_HEREDITY_PROTOCOL.md). It physically
fragments or rearranges acquired donor states, follows branching pedigrees,
and tests noise, environmental memory, observer agreement, and cross-rule
transplantation. Launch the reference run in a separate session with:

```bash
.venv/bin/python -m plastic_ca causal-heredity \
  --profile reference \
  --workers 20 \
  --max-hours 24 \
  --output results/causal-heredity-round-1 \
  --detach
```

The launcher immediately returns a PID. Progress and ETA are available in
`STATUS.json`, output is appended to `run.log`, and `--resume` reuses only
checkpoints with the complete matching design digest.

The preregistered round-2 Life carrier campaign is in
[`LIFE_CARRIER_PROTOCOL.md`](LIFE_CARRIER_PROTOCOL.md). It independently audits
the nearly saturated positive rule, searches multi-form rules for reciprocal
A/B transmission, seals candidates before holdout, and maps only confirmed
leads. Launch it detached with:

```bash
.venv/bin/python -m plastic_ca life-carrier \
  --profile reference \
  --workers 20 \
  --max-hours 48 \
  --output results/life-carrier-round-2 \
  --detach
```

The run is atomically checkpointed and resumable. Screening results cannot
establish a claim; confirmatory verdicts use fresh donors and RNG streams after
the child-selection manifest is sealed.

Round 3 is preregistered in
[`CA_CARRIER_V3_PROTOCOL.md`](CA_CARRIER_V3_PROTOCOL.md). It separates the
strong rule-31649 pair-specific texture result from the stronger claims that a
form generalizes across fresh donors, survives pedigrees, and is visible to a
global observer. It also searches all 256 ECA rules and the complete retained
1,024-rule Life-like registry using continuous forms rather than pooled support
IDs. Launch or resume the 48-hour run with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-carrier-v3 \
  --profile reference \
  --workers 20 \
  --max-hours 48 \
  --output results/ca-carrier-round-3 \
  --resume \
  --detach
```

The command returns immediately. `STATUS.json` is the polling interface;
`run.log` contains the durable stage log, and every selection seal records the
full design digest used by its checkpoints.

Round 4 is preregistered in
[`CA_LINEAGE_FIELD_PROTOCOL.md`](CA_LINEAGE_FIELD_PROTOCOL.md). It tests two
equally powered slow-field mechanisms after a bitwise-complete visible-board
reset, with zero/shuffle/read/write/no-rewrite, ablation, rescue,
opposite-history, compression, and matched visible-state controls. The
reference job is hard-capped at eight wall hours:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-lineage-field \
  --profile reference \
  --workers 20 \
  --max-hours 8 \
  --output results/ca-lineage-field-round-4 \
  --detach
```

The timing benchmark may reduce pair and future counts symmetrically for both
mechanisms, but it cannot inspect A/B outcomes. The selected profile and every
clean-room input are sealed in `DESIGN.json` before the main trajectories.

The next five-stage programme begins with the motif-carrier upper bound
preregistered in
[`CA_MOTIF_LINEAGE_STAGE1_PROTOCOL.md`](CA_MOTIF_LINEAGE_STAGE1_PROTOCOL.md).
Stage 1 tests context-indexed and 3x3 motif-energy carriers on fresh, disjoint
Rule-31649 discovery and validation cohorts. Later stages remain locked behind
reviewed decision artifacts:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-motif-lineage \
  --stage upper-bound \
  --profile reference \
  --workers 20 \
  --max-hours 8 \
  --output results/ca-motif-lineage-stage-1 \
  --detach
```

`STATUS.json` is the polling interface. `QUEUE.json` and
`STAGE_DECISION.json` keep Stages 2--5 blocked; a Stage-1 controllability result
cannot by itself receive a Plastic Heredity verdict.

After a reviewed Stage-1 pass, the frozen-reader generalization experiment is
preregistered in
[`CA_MOTIF_LINEAGE_STAGE2_PROTOCOL.md`](CA_MOTIF_LINEAGE_STAGE2_PROTOCOL.md):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-motif-lineage \
  --stage generalize \
  --profile reference \
  --stage1-root results/ca-motif-lineage-stage-1 \
  --workers 20 \
  --max-hours 8 \
  --output results/ca-motif-lineage-stage-2 \
  --detach
```

This stage excludes every earlier pair and cannot retune the imported reader.
Stage 3 remains locked until the new resets, transfers, symmetries, controls,
and carrier-dose response have been reviewed.

After that reviewed Stage-2 pass, the renewed-lineage experiment is frozen in
[`CA_MOTIF_LINEAGE_STAGE3_PROTOCOL.md`](CA_MOTIF_LINEAGE_STAGE3_PROTOCOL.md).
It resets the visible daughter before every generation, restricts inherited
carrier reading to sweeps 1--32, and requires the daughter to rewrite the next
carrier from its own visible texture during sweeps 33--64. The 16-generation
causal ladder includes stopped rewriting, ablation, same-history rescue,
opposite-history rescue, and repeated carrier corruption:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-motif-lineage \
  --stage lineage \
  --profile reference \
  --stage1-root results/ca-motif-lineage-stage-1 \
  --stage2-root results/ca-motif-lineage-stage-2 \
  --workers 20 \
  --max-hours 8 \
  --output results/ca-motif-lineage-stage-3 \
  --detach
```

The first two Stage-3 pairs are quarantined for engineering smoke tests; the
64-pair reference cohort remains untouched until those tests pass. Stage 4 is
never launched automatically.

The negative Stage-3 renewal result is followed by the semantic-closure repair
campaign preregistered in
[`CA_MOTIF_LINEAGE_STAGE3R_PROTOCOL.md`](CA_MOTIF_LINEAGE_STAGE3R_PROTOCOL.md).
It reuses the now-exposed Stage-3 cohort to measure parent-to-daughter carrier
drift and fit label-blind universal repairs, then screens those repairs on 64
fresh pairs:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-motif-lineage \
  --stage repair \
  --profile reference \
  --stage1-root results/ca-motif-lineage-stage-1 \
  --stage2-root results/ca-motif-lineage-stage-2 \
  --stage3-root results/ca-motif-lineage-stage-3 \
  --workers 20 \
  --max-hours 8 \
  --output results/ca-motif-lineage-stage-3r \
  --detach
```

That invocation stops at `awaiting_human_review`; it cannot simulate the sealed
96-pair confirmation cohort. After reviewing `SELECTION_DECISION.json`, an
explicit confirmation invocation is required:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-motif-lineage \
  --stage repair \
  --phase confirm \
  --authorize-confirmation \
  --profile reference \
  --stage1-root results/ca-motif-lineage-stage-1 \
  --stage2-root results/ca-motif-lineage-stage-2 \
  --stage3-root results/ca-motif-lineage-stage-3 \
  --workers 20 \
  --max-hours 8 \
  --output results/ca-motif-lineage-stage-3r \
  --resume \
  --detach
```

The positive Stage-3R result advances to the preregistered compression and
robustness experiment in
[`CA_MOTIF_LINEAGE_STAGE4_PROTOCOL.md`](CA_MOTIF_LINEAGE_STAGE4_PROTOCOL.md).
Stage 4 compares 50 label-blind codecs, applies the full carrier causal ladder,
maps copying damage and parameter drift, and probes two already-exposed nearby
rules. Its reference-scale preconfirmation run is resumable and capped at
eight wall hours:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-motif-lineage \
  --stage compression \
  --profile reference \
  --stage1-root results/ca-motif-lineage-stage-1 \
  --stage2-root results/ca-motif-lineage-stage-2 \
  --stage3-root results/ca-motif-lineage-stage-3 \
  --stage3r-root results/ca-motif-lineage-stage-3r \
  --workers 20 \
  --max-hours 8 \
  --output results/ca-motif-lineage-stage-4 \
  --detach
```

That run cannot open the 128 sealed confirmation pairs. After reviewing its
`SELECTION_DECISION.json`, confirmation requires a separate invocation with
`--phase confirm --authorize-confirmation --resume`. Stage 5 is never launched
automatically.

After the reviewed positive Stage-4 result, physical localization is frozen in
[`CA_MOTIF_LINEAGE_STAGE5_PROTOCOL.md`](CA_MOTIF_LINEAGE_STAGE5_PROTOCOL.md).
Stage 5 places the exact 16 Walsh channels in a 16x16 local field, permits only
nearest-neighbour transport and local motif read/write operations, and lets
only a one-, four-, or sixteen-site patch cross each visible reset. Its
reference preconfirmation campaign uses only already-exposed pairs:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-motif-lineage \
  --stage localization \
  --profile reference \
  --stage1-root results/ca-motif-lineage-stage-1 \
  --stage2-root results/ca-motif-lineage-stage-2 \
  --stage3-root results/ca-motif-lineage-stage-3 \
  --stage3r-root results/ca-motif-lineage-stage-3r \
  --stage4-root results/ca-motif-lineage-stage-4 \
  --workers 20 \
  --max-hours 8 \
  --output results/ca-motif-lineage-stage-5 \
  --detach
```

The run stops at human review without touching the 128-pair Stage-5
confirmation cohort. After reviewing `SELECTION_DECISION.json`, the locked
objects can be tested with a separate invocation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-motif-lineage \
  --stage localization \
  --phase confirm \
  --authorize-confirmation \
  --profile reference \
  --stage1-root results/ca-motif-lineage-stage-1 \
  --stage2-root results/ca-motif-lineage-stage-2 \
  --stage3-root results/ca-motif-lineage-stage-3 \
  --stage3r-root results/ca-motif-lineage-stage-3r \
  --stage4-root results/ca-motif-lineage-stage-4 \
  --workers 20 \
  --max-hours 8 \
  --output results/ca-motif-lineage-stage-5 \
  --resume \
  --detach
```

Stage 5R is the preregistered follow-up to Stage 5's local-diffusion negative.
It keeps the visible CA and 64-bit Walsh message frozen, but replaces passive
diffusion with an eight-step nearest-neighbour copying wave and replaces the
failed pointwise writer with an audited 30-step local consolidation circuit.
Its protocol is [`CA_MOTIF_LINEAGE_STAGE5R_PROTOCOL.md`](CA_MOTIF_LINEAGE_STAGE5R_PROTOCOL.md).
Run the exposed preconfirmation campaign with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-motif-lineage \
  --stage regeneration \
  --profile reference \
  --stage1-root results/ca-motif-lineage-stage-1 \
  --stage2-root results/ca-motif-lineage-stage-2 \
  --stage3-root results/ca-motif-lineage-stage-3 \
  --stage3r-root results/ca-motif-lineage-stage-3r \
  --stage4-root results/ca-motif-lineage-stage-4 \
  --stage5-root results/ca-motif-lineage-stage-5 \
  --workers 20 \
  --max-hours 8 \
  --output results/ca-motif-lineage-stage-5r \
  --detach
```

This invocation can use only Stage-5-exposed pairs. If and only if a local
candidate qualifies, it freezes at most three confirmation objects and stops.
The first 96 still-untouched Stage-5 reserve pairs can then be opened only by a
separate `--phase confirm --authorize-confirmation --resume` invocation; the
other 62 remain untouched.

Inherited boundary bits, temporary developmental-field storage, and shared
codebook bits are reported separately. Stage 6 is never launched
automatically.

Stage 6 is the five-gate minimality, scale, compression, ecology/evolution,
and final-audit programme registered in
[`CA_MOTIF_LINEAGE_STAGE6_PROTOCOL.md`](CA_MOTIF_LINEAGE_STAGE6_PROTOCOL.md).
Each round is a separate resumable invocation capped at four wall hours. Start
only Round 6A with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-motif-lineage \
  --stage minimality \
  --round locality \
  --profile reference \
  --stage1-root results/ca-motif-lineage-stage-1 \
  --stage2-root results/ca-motif-lineage-stage-2 \
  --stage3-root results/ca-motif-lineage-stage-3 \
  --stage3r-root results/ca-motif-lineage-stage-3r \
  --stage4-root results/ca-motif-lineage-stage-4 \
  --stage5-root results/ca-motif-lineage-stage-5 \
  --stage5r-root results/ca-motif-lineage-stage-5r \
  --workers 4 \
  --max-hours 4 \
  --output results/ca-motif-lineage-stage-6 \
  --detach
```

After inspecting that round's `RESULTS.json`, `REPORT.md`, and
`STAGE_DECISION.json`, invoke `scale`, then `compression`, then `ecology`
with the same command and output root, changing `--round` and adding
`--resume`. A failed reference gate stops the next round unless the reviewer
deliberately adds `--authorize-gate-override`; nothing advances
automatically. The final 62-pair audit additionally requires
`--round audit --resume --authorize-final-audit` and seals
`FINAL_AUDIT_DESIGN.json` before opening any reserved trajectory.

Round 6A completed with no bounded candidate passing its gate. Post-run
forensics showed that the v1 bounded writer spatially averaged its observations
and then divided by the window area again; its independently relocated seed
could also leave the parent's causal region. The original result and hashes are
preserved. The correction is tested separately under
[`CA_MOTIF_LINEAGE_STAGE6AR_PROTOCOL.md`](CA_MOTIF_LINEAGE_STAGE6AR_PROTOCOL.md),
not by overwriting or overriding Round 6A.

Start only the Stage 6A-R correction audit with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-motif-lineage \
  --stage minimality-repair \
  --phase audit \
  --profile reference \
  --stage6-root results/ca-motif-lineage-stage-6 \
  --stage5r-root results/ca-motif-lineage-stage-5r \
  --workers 4 \
  --max-hours 4 \
  --output results/ca-motif-lineage-stage-6ar \
  --detach
```

After reviewing each positive `STAGE_DECISION.json`, invoke `bridge`, `screen`,
`qualify`, and `endurance` one at a time with the same output root and
`--resume`. Qualification additionally requires `--authorize-confirmation`.
Every invocation is capped at four workers and four wall hours. No repair phase
opens the 62-pair reserve or launches a successor automatically.

Stage 6B-R follows the frozen negative Stage-6A-R local screen without changing
that result. It tests the mechanistic leads exposed by the forensics: transient
radius-five inheritance, centred writers, closed-loop writer fitting, a slower
renewed latch, error protection, radii six and seven, and several cooperating
local seeds. Its five checkpointed phases continue automatically under one
persisted eight-hour clock; no more than four workers are permitted. The last
1.75 hours are reserved for adjudication.

The protected 62-pair reserve is still conditional. It is loaded only if a
finite candidate first passes generation-16 qualification and generation-64
endurance, and only after candidate models and the full audit design have been
sealed. The registered invocation is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m plastic_ca ca-motif-lineage \
  --stage renewal-repair \
  --phase all \
  --profile reference \
  --stage6-root results/ca-motif-lineage-stage-6 \
  --stage6ar-root results/ca-motif-lineage-stage-6ar \
  --stage5r-root results/ca-motif-lineage-stage-5r \
  --workers 4 \
  --max-hours 8 \
  --auto-final-audit \
  --output results/ca-motif-lineage-stage-6br \
  --detach
```

The full design is frozen in
[`CA_MOTIF_LINEAGE_STAGE6BR_PROTOCOL.md`](CA_MOTIF_LINEAGE_STAGE6BR_PROTOCOL.md).
Progress is pollable in `STATUS.json`; completed work is checkpointed and can
be continued with the same command plus `--resume` while the persisted campaign
deadline remains open.

## Interpretation

Exact equality is claimed for the retained raw ECA atlas, not indiscriminately
for every downstream experiment. Particle-domain launch rows and the phase
stream remain undisclosed, and the Life run still differs numerically despite
using the disclosed round-5 pooling rule. Those stages are reported as
gate-level or strong statistical replications with their limitations intact.
