# Suggested manuscript additions

These passages are generated from the completed, verified post-hoc audit. They
are concrete replacements for the earlier placeholder text. They must remain
labelled as post-hoc replay and robustness analyses.

## Results: non-degeneracy of the registered strict event

> A post-hoc non-degeneracy audit characterized every earliest qualifying
> registered-cosine window in the retained confirmation futures. Across
> candidates 02 and 03, respectively, the eight selected daughters had mean
> effective species numbers of 5.36 and 5.49, mean occupied-type counts of
> 11.86 and 12.07, and mean largest-species shares of 0.583 and 0.566. Only
> 0.13% and 0.04% of events assigned at least 80% of the composition to one
> type in all eight daughters; the corresponding two-type fractions were
> 1.06% and 0.30%. Registered-cosine coherence therefore was not generally a
> trivial consequence of universal one- or two-species domination.

## Results: matched concentration and turnover comparison

> The same events were nevertheless compositionally concentrated relative to
> hard negative controls. Each event was matched without replacement to a
> non-event branch from the same natural starting state that still reached a
> post-break inherited run of eight. Across 2,199 and 2,596 matched pairs in
> candidates 02 and 03, event windows had 5.48 and 5.30 fewer effective
> species, 4.82 and 4.61 fewer occupied types, and leading-species shares
> larger by 0.219 and 0.215. Their adjacent total-variation turnover was lower
> by 0.109 and 0.108, and they required 7.26 and 6.73 fewer growth updates.
> Every whole-matrix bootstrap interval excluded zero. Thus strict-event
> coherence is associated with a marked concentration/low-turnover regime,
> although it is not normally complete collapse to one or two types.

## Results: alternative-metric geometry

> Because boundary inheritance, mutual daughter coherence, and old-anchor
> separation are geometrically different relations, a post-hoc development-
> only calibration mapped their cosine cutoffs separately to Bray–Curtis
> similarity. The resulting relation-specific Bray endpoint occurred in 606
> of 128,000 candidate-02 futures (0.47%) and 840 of 128,000 candidate-03
> futures (0.66%), compared with 1.84% and 2.10% under registered cosine. Its
> overlap with the cosine event remained limited (Jaccard 0.158 and 0.176),
> although it exceeded the prior single-global-map overlap (0.073 and 0.063).
> On the exact cosine-selected windows, the pairwise-coherence condition was
> the principal source of cross-metric disagreement. Event frequency and
> membership should therefore be interpreted as composition-metric dependent,
> not as one invariant class recovered by interchangeable similarity measures.

## Results or predictor discussion

> Target-specific development fits did not establish a robust strict-event
> predictor. Adding the retained state block to ten-boundary history passed one
> of four registered-cosine candidate-by-half cells and zero of four cells for
> either Bray–Curtis target. The original cosine-trained predictor also had
> small negative gains when transferred to the globally mapped Bray labels.
> These post-hoc results leave prospective occurrence intact while separating
> it from a claim of reproducibly successful pre-event prediction.

## Supplementary Methods

> For every retained future, three endpoint definitions were replayed without
> changing its state, branch, seed, or trajectory: registered cosine, the prior
> globally percentile-mapped Bray–Curtis rule, and a relation-specific
> Bray–Curtis rule. The latter mapped the boundary, within-window pairwise, and
> old-anchor cutoffs separately using fixed development branches only and did
> not match endpoint prevalence. For each earliest qualifying event window we
> measured Shannon effective species number, occupied-type count, largest- and
> two-largest-species shares, adjacent composition turnover, occupied-set
> turnover, growth updates, and the registered boundary, coherence, and anchor
> margins. Controls were the earliest eligible post-break run-8 windows in
> negative branches from the same state and were matched deterministically
> without replacement. Uncertainty and randomization resampled complete
> catalytic matrices. Target-specific prediction models used development
> outcomes only and were sealed before confirmation scoring.

## Abstract or first strict-event mention

Use:

> Rare selected-lineage episodes comprised eight mutually coherent inherited
> daughters around a composition distinct from the old pre-break anchor.

Do not replace this with “eight faithful offspring” or language implying that
both daughters from each fission were followed.

## Limitations

> The strict coherent-eight endpoint follows one contract-selected daughter at
> each fission and does not measure both-daughter reproductive fidelity.
> Registered-cosine events were rarely universally one- or two-species
> dominated, but they were substantially more concentrated and lower-turnover
> than matched same-state run-8 controls. Their prevalence and membership were
> also strongly composition-metric dependent: percentile-matched Bray–Curtis
> rules selected rarer and mostly different positive futures. These post-hoc
> diagnostics constrain the endpoint's interpretation without replacing its
> registered cosine definition or creating a new confirmation claim.

## Required source tables

- `artifacts/output/event_nondegeneracy_summary.csv`
- `artifacts/output/matched_nondegeneracy_effects.csv`
- `artifacts/output/gate_waterfall.csv`
- `artifacts/output/event_overlap.csv`
- `artifacts/output/prediction_comparisons.csv`
- `artifacts/output/figures/matched_nondegeneracy_effects.png`
- `artifacts/output/figures/event_overlap_and_strata.png`
