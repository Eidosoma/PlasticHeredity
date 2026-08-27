# Preregistered causal Plastic Heredity campaign

Date frozen: 2026-08-21, before opening any causal-campaign reference
outcomes.

## Evidence boundary

This is a clean-room follow-up to the completed E19/E23/E24 cellular-
automaton reconstructions.  It asks whether physical information inherited
from an established parent is *causally sufficient* to regenerate and
propagate that parent's acquired form.  The existing exact ECA and frozen
Life-family execution conventions are reused without consulting sibling
implementation code.  Existing atlases are used only to choose rules and the
desired donor class; all donors and tests use fresh namespaces.

Survival is not heredity.  A recovery is positive only when the final eight
completed compositions are each more than 0.90 similar to the donor's
established centroid and mutually coherent above 0.90.  For switchers, all
eight must also remain at or below 0.85 similarity to the historical
break-causing-daughter anchor used by E19.

## Frozen panels and donor acquisition

The ECA panel is `0, 8, 11, 18, 22, 30, 35, 41, 43, 45, 54, 57, 90, 106,
110, 122, 126, 146, 150, 184`.  Champions use the raw final-4 observer;
class-3/class-4 candidates use the registered particle observer; rules 0 and
8 are raw negative controls.

The Life panel contains the eight named rules, the eight highest in-band
strict rules in the frozen B48 atlas, the four remaining rules with largest
form libraries, and four strict-zero controls nearest to the first four
strict selections in standardized break rate, generation clock, and survival.
Ties use ascending rule id and selections are filled after deduplication.

Reference acquisition targets sixteen donors per rule and examines at most
16,384 fresh lineages.  Rules with development-atlas strict >= 0.005 seek
strict switchers; the remainder seek lineages with sixteen unbroken coherent
boundaries.  Failure to acquire the target is a result and cannot remove a
rule from reporting.

## Causal interventions

Each donor is tested in an independent common garden for sixteen completed
generations with sixteen stochastic replicates.  Conditions are intact and
pre-break/launch ancestor controls plus 0.25, 0.50, and 0.75 transmission as:

- one- and two-interval ECA fragments;
- square and strip/two-lobe Life fragments;
- dispersed equal-area masks;
- position-shuffled equal-live-mass states;
- density-matched random states.

Pedigrees branch twice per division to depth five from eight donors per rule.
They compare intact copying, complementary spatial halves, and shuffled
halves.  Final leaves receive an eight-generation certification garden.

The additional registered probes are a 3x3 process/copy-noise grid on intact
and half transmission; four-generation zero-versus-double-process-noise
preconditioning followed by cue removal; and intact transplantation into all
one-bit neighboring ECA/B/S rules.  Cross-rule ECA identity is scored with the
universal raw observer.

## Observer panel and statistics

ECA uses registered final-4 or particle identity plus raw final-4 and a
concatenated, separately normalized cyclic k=2..6 spectrum.  Life uses the
registered accumulated live-2x2 census plus terminal live-2x2 and toroidal
connected-component size spectra.

Headline differences use donor-clustered deterministic bootstrap intervals
with 10,000 reference resamples.  Rule-level paired permutation tests use
Benjamini-Hochberg q=0.05.  Headline causal gates use strict switcher donors;
stable maintainers are a separately reported conservative-heredity control
and never substitute for an acquired-form event.  The frozen gates are:

1. `causal_transmission`: structured 50% transmission beats density-random
   by at least 0.15 with a positive 95% lower bound in both substrates.
2. `structure_matters`: structured beats shuffled transmission in both.
3. `dose_response`: structured recovery is monotonically increasing over
   0.25, 0.50, 0.75, and 1.00 with positive bootstrap slope.
4. `pedigree_persistence`: depth-five complementary-half retention beats
   shuffled-half retention in both substrates.
5. `observer_robustness`: the primary direction is shared by at least one
   independent observer family in each substrate.
6. `environmental_memory`: removed-cue identity is classified above 0.65 and
   permutation p<0.05, including after daughter transmission.
7. `rule_specificity`: native-rule retention exceeds one-bit-neighbor
   retention with a positive bootstrap lower bound.

Every reversal, observer disagreement, missing donor, extinction, and
budget-truncated cell is retained.  No exploratory condition may replace a
registered condition.

## Operational contract

The reference profile uses 20 workers and a 24-hour ceiling.  Every rule-stage
checkpoint is atomic and bound to the complete design digest.  Detached runs
write `RUN.pid`, `run.log`, and atomic `STATUS.json`; a deadline finishes the
current rule checkpoints, produces an explicitly partial report, and remains
exactly resumable.  `COMPLETE` is written only after all registered stages and
adjudication finish.
