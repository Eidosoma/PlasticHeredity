# Preregistered CA motif-lineage Stage 5R: regenerative localization

## Motivation and ancestry

Stage 5 found a sharp mechanistic boundary. The frozen 16-coordinate Walsh
carrier remained strong when globally broadcast (generation-8 crossover
0.949 and generation-16 crossover 0.941 in qualification), and its exposed
spectral anatomy was distributed across modes. Every local diffusive carrier
was null or harmful: the best exposed local candidate reached -0.003 at
generation 8 and 0.000 at generation 16. Pure diffusion conserved total field
mass but diluted a small seed before it could guide the lattice.

Stage 5R changes only that failed physical layer. It asks whether the same
frozen 64-bit Walsh payload can enter at one lattice site, reproduce outward
through strictly nearest-neighbour communication, guide the reset daughter,
and be reconstructed from the daughter's local visible motifs. All Stage-5
engineering and qualification pairs are exposed. None of the 128 Stage-5
confirmation pairs or 30 later-audit pairs has been simulated. Stage 5R uses
only exposed pairs for fitting, calibration, bridge tests, screening, and
qualification.

This remains a synthetic CA experiment. A positive result is not evidence of
metabolism, agency, biological life, consciousness, or nonphysical memory. No
Wagner or Fable implementation source may be read, imported, hashed, or
executed.

## Frozen visible and inherited semantics

The visible rule (31649), 16x16 torus, bitwise-identical daughter reset,
64 visible sweeps, process noise 0.002, read window 1--32, write window 49--64,
reader strength 0.25, phenotype observers, pair-level bootstrap, and strict
causal thresholds remain frozen from Stages 3R--5.

The inherited object remains the exact Stage-4 Walsh basis, channel order,
per-channel four-bit scales, and exact-zero quantizer. The primary boundary is
one spatial site containing 16 four-bit values: 64 inherited bits. A 2x2
redundant patch contains 256 inherited bits and is a registered fallback. The
shared basis, writer maps, propagation rule, clocks, and developmental buffers
are machinery rather than inherited state and are counted separately.

## Regenerative carrier wave

At reset, only the inherited patch is occupied; every other field site is
exactly zero and unoccupied. Before visible development, the field receives an
eight-step germination prelude while the visible board remains frozen. In each
synchronous step an unoccupied site with one or more occupied Moore neighbours
copies their mean 16-channel value and becomes occupied. Existing occupied
sites either retain their value (`flood-retain`) or join the same local average
(`flood-consensus`). A bistable reaction-diffusion alternative is included in
outcome-blind calibration. Exact zero is a fixed point. No site can be affected
outside the one-site-per-step Chebyshev light cone.

On a 16x16 torus, a one-site `flood-retain` seed fills the lattice in eight
steps and is then bitwise uniform. Its subsequent local Walsh reader is
mathematically identical to the Stage-4 global reader. Runtime may use that
proved uniform-field equivalence as a vectorized optimization; sparse and
damaged fields always use the explicit local reader. Translation, propagation
ablation, pure-diffusion replacement, and wavefront timing are registered
controls.

## Local daughter writers

During visible sweeps 49--64, every site observes only its own current 3x3
motif. Three label-blind writer classes are frozen:

- `hist512-exact`: each site conceptually accumulates a local 512-bin motif
  histogram. A nearest-neighbour reduction produces the global histogram,
  after which the frozen reference, repair gain 0.50, Walsh projection, and
  four-bit quantizer produce the next 16 values.
- `moment16-ridge`: each site accumulates only the 16 selected Walsh parity
  moments. A universal uncentred/affine 16x16 ridge map, fitted without lineage
  labels from exposed Stage-3R traces, predicts the repaired Walsh values.
- `moment16-diagonal`: the same 16 moments use independent affine channel
  maps, providing the smallest developmental writer.

The spatial reduction is an explicit 30-step nearest-neighbour circuit. A
packet is shifted one site and accumulated for 15 horizontal steps, then the
row result is shifted and accumulated for 15 vertical steps. Its endpoint is
exactly the spatial mean at every site; no intermediate signal travels faster
than one edge per step. Because the visible lattice is frozen during these 30
carrier-only consolidation steps, the implementation may execute the audited
algebraically equivalent endpoint reduction. Disabling consolidation leaves
only the extraction site's local observations.

The exact histogram writer is deliberately a high-machinery existence test,
not a compact-development claim. Its per-site 512-bin accumulator and routing
buffers are reported separately from the 64 inherited bits. The moment writers
test whether the same result survives progressive developmental compression.

## Outcome-blind calibration and bridge assays

Propagation calibration uses synthetic signed payloads only. It measures
coverage after each step, reconstruction error, translation invariance, exact
zero stability, collision behaviour, and light-cone compliance for
`flood-retain`, `flood-consensus`, and bistable reaction-diffusion settings.
The top two distinct propagation classes advance. No lineage labels or
phenotype outcomes enter calibration.

An exposed 16-pair bridge assay separately tests: the frozen global anchor;
regenerative transport with the exact writer; replacement of regeneration by
pure diffusion; transport disabled; consolidation disabled; and a
founder-clamped propagation upper bound. It reports whether failure lies in
seed transport, local reading, or daughter rewrite. It may eliminate broken
mechanisms but cannot change thresholds or inspect untouched pairs.

The registered candidate atlas crosses the two selected propagators, three
writers, and 1x1/2x2 patches, capped at 12 candidates. Screening uses 64
exposed pairs, 16 futures per history, and eight generations. At most four
local candidates plus the global anchor advance. Qualification uses 96 exposed
pairs, 32 futures per history, and 16 generations.

## Causal qualification

Every qualifying local candidate receives the inherited-memory ladder:

- intact renewal;
- zero and channel-shuffled boundaries;
- read disabled, founder write disabled, daughter write disabled, and no
  rewrite;
- ablation after generation 2, correct rescue entering generation 4, and
  opposite-history rescue;
- opposite founder and one-percent sign corruption;
- transport disabled, regeneration replaced by pure diffusion, and writer
  consolidation disabled;
- translated patch, spatially shuffled patch, half-width bottleneck, and both
  half-channel recombinations.

A local candidate must pass the frozen strict renewal gate and show positive,
interval-supported advantages over transport disabled, regeneration disabled,
and consolidation disabled. Its translated-patch effect must retain at least
70 percent of intact performance. Payload accounting, exact-zero behaviour,
uniform/global reader equivalence, the propagation light cone, and the
30-step writer circuit must all pass.

Rule-31648 and rule-70366 transfer panels use already-exposed pairs and cannot
nominate or rescue a candidate.

## Confirmation seal and verdicts

Stage 5R confirmation never launches automatically. If an exposed local
candidate qualifies, at most three objects are frozen: the Stage-4 global
anchor, the strongest qualified one-site exact writer, and the smallest
qualified moment writer. Confirmation requires a separate invocation with
`--resume` and `--authorize-confirmation`.

The first 96 still-untouched Stage-5 reserve pairs form the Stage-5R
confirmation cohort; the other 62 remain untouched for a later audit. The
reference confirmation uses 64 futures per history, 16 generations, ordinary
and moderate-joint damage, 10,000 pair-cluster bootstrap resamples, and alpha
0.005 per object. No model, candidate, threshold, or cohort may change after
the confirmation design is written.

Verdicts are hierarchical:

- `ROBUST_REGENERATIVE_LOCAL_64BIT_CA_PLASTIC_HEREDITY`: a one-site local
  candidate passes ordinarily and under moderate damage;
- `REGENERATIVE_LOCAL_64BIT_CA_PLASTIC_HEREDITY`: it passes ordinarily only;
- `REGENERATIVE_LOCAL_DISTRIBUTED_CA_PLASTIC_HEREDITY`: only the 2x2 fallback
  passes ordinarily;
- `LOCAL_PROPAGATION_WITHOUT_RENEWAL`: the bridge proves local propagation and
  reading but no daughter writer passes;
- `GLOBAL_BROADCAST_ONLY`: only the frozen global anchor passes;
- `NO_STAGE4_REPLICATION`: the global anchor fails freshly.

Results must distinguish inherited bits, developmental carrier-field bits,
developmental writer-buffer bits, occupancy bits, and shared parameter bits.

## Execution

The phases are `audit`, `fit`, `calibrate`, `bridge`, `screen`, `qualify`,
`transfer`, `adjudicate`, and `confirm`. Reference invocations use at most 20
workers and eight wall hours, reserving 30 minutes for orderly checkpointing.
All random streams are semantic and paired across histories. Design, model,
cohort, protocol, checkpoint, PID, status, ETA, log, report, and queue artifacts
are atomic and resume-bound. Stage 6 never launches automatically.
