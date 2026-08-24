# Strict-event geometry, non-degeneracy, and target-specific prediction audit

## Bottom line

The relation-specific Bray target recovered closer event membership than the
prior global mapping in both candidates, but its target-specific state-block
comparison passed the frozen exploratory gate in 0/4 candidate-by-half cells.
Registered-cosine events were not universally dominated by one molecular type
across all eight daughters; they were nevertheless markedly more concentrated
and lower-turnover than matched same-state run-8 controls.

This is a locked post-hoc robustness audit. It replays the existing deterministic development and confirmation futures; it does not generate a new prospective confirmation cohort and does not alter the manuscript.

## What was tested

Three otherwise identical strict-event definitions were applied to every retained future: the registered cosine definition, the previous globally percentile-mapped Bray–Curtis definition, and a new Bray–Curtis definition whose boundary, within-window coherence, and old-anchor cutoffs were calibrated separately. Relation-specific cutoffs used only fixed development branches and did not match event prevalence.

| Relation | Cosine cutoff | Matched Bray cutoff | Development comparisons |
|---|---:|---:|---:|
| Boundary | 0.90000 | 0.71250 | 1,024,000 |
| Coherence | 0.90000 | 0.65828 | 1,433,796 |
| Anchor | 0.85000 | 0.58105 | 284,000 |

## Event counts and power

| Endpoint | Candidate | Development events | Confirmation events | Confirmation rate | Event matrices | Power rule |
|---|---:|---:|---:|---:|---:|---|
| Registered cosine | 02 | 2,705 | 2,354 | 1.84% | 140 | adequate |
| Registered cosine | 03 | 3,154 | 2,687 | 2.10% | 149 | adequate |
| Global-mapped Bray | 02 | 316 | 251 | 0.20% | 49 | adequate |
| Global-mapped Bray | 03 | 542 | 259 | 0.20% | 64 | adequate |
| Relation-mapped Bray | 02 | 756 | 606 | 0.47% | 93 | adequate |
| Relation-mapped Bray | 03 | 1,218 | 840 | 0.66% | 103 | adequate |

The frozen descriptive power rule requires at least 100 events and 20 event-bearing matrices in both development and confirmation for each endpoint–candidate cell. An underpowered cell remains reported but cannot pass the exploratory prediction gate.

## Failure-gate localization

| Endpoint | Candidate | Break | Later run-8 | Coherent window | Strict event |
|---|---:|---:|---:|---:|---:|
| Registered cosine | 02 | 69.07% | 52.74% | 2.00% | 1.84% |
| Registered cosine | 03 | 73.86% | 56.24% | 2.38% | 2.10% |
| Global-mapped Bray | 02 | 68.39% | 52.33% | 0.20% | 0.20% |
| Global-mapped Bray | 03 | 77.42% | 58.38% | 0.23% | 0.20% |
| Relation-mapped Bray | 02 | 68.39% | 52.33% | 0.51% | 0.47% |
| Relation-mapped Bray | 03 | 77.42% | 58.38% | 0.82% | 0.66% |

These are cumulative fractions. They distinguish failure to break, failure to regain eight consecutive inherited selected-lineage fissions, failure of mutual daughter coherence, and failure of old-anchor separation.

## Event overlap and same-window geometry

| Candidate | Pair | Intersection | Union | Jaccard |
|---|---|---:|---:|---:|
| 02 | Cosine vs Bray global | 177 | 2,428 | 0.073 |
| 02 | Cosine vs Bray relation | 404 | 2,556 | 0.158 |
| 03 | Cosine vs Bray global | 175 | 2,771 | 0.063 |
| 03 | Cosine vs Bray relation | 529 | 2,998 | 0.176 |

| Candidate | Cosine-event window evaluated under | Boundary pass | Pairwise pass | Anchor pass | All conditions |
|---|---|---:|---:|---:|---:|
| 02 | Global-mapped Bray | 80.25% | 1.19% | 81.05% | 0.98% |
| 02 | Relation-mapped Bray | 80.25% | 5.27% | 80.25% | 4.84% |
| 03 | Global-mapped Bray | 87.05% | 1.04% | 88.31% | 0.93% |
| 03 | Relation-mapped Bray | 87.05% | 5.73% | 86.64% | 5.36% |

The same-window table asks whether the exact eight-daughter window that qualifies under cosine also satisfies each Bray condition. It therefore localizes geometric disagreement without changing the temporal window.

## Non-degeneracy of all strict events

| Endpoint | Candidate | Effective species | Occupied types | Largest-species share | Adjacent TV | All 8 top-1 dominated | All 8 top-2 dominated |
|---|---:|---:|---:|---:|---:|---:|---:|
| Registered cosine | 02 | 5.36 | 11.86 | 0.583 | 0.193 | 0.13% | 1.06% |
| Registered cosine | 03 | 5.49 | 12.07 | 0.566 | 0.196 | 0.04% | 0.30% |
| Global-mapped Bray | 02 | 2.86 | 7.20 | 0.738 | 0.112 | 10.36% | 30.28% |
| Global-mapped Bray | 03 | 3.26 | 7.95 | 0.687 | 0.124 | 3.09% | 16.22% |
| Relation-mapped Bray | 02 | 3.81 | 8.89 | 0.650 | 0.142 | 2.48% | 10.23% |
| Relation-mapped Bray | 03 | 4.23 | 9.41 | 0.598 | 0.153 | 0.60% | 3.57% |

Effective species number is exp(Shannon entropy). Occupied-type and composition statistics are evaluated on the eight selected daughters in the earliest qualifying window. `All 8 top-1 dominated` means every daughter assigns at least 80% of its normalized composition to one type; the top-2 column applies the same rule to the two largest types. By construction, all eight event boundaries exceed the endpoint-specific inheritance cutoff; the retained event table also reports the minimum boundary, pairwise-coherence, and anchor-distinctness margins.

## Matched non-event comparison

| Endpoint | Candidate | Pairs | Statistic | Event − control | 95% matrix-bootstrap CI |
|---|---:|---:|---|---:|---:|
| Registered cosine | 02 | 2,199 | effective_species_mean | -5.4848 | [-5.8441, -5.1215] |
| Registered cosine | 02 | 2,199 | occupied_types_mean | -4.8223 | [-5.2199, -4.4006] |
| Registered cosine | 02 | 2,199 | top1_share_mean | 0.2193 | [0.2020, 0.2366] |
| Registered cosine | 02 | 2,199 | adjacent_total_variation_mean | -0.1089 | [-0.1157, -0.1021] |
| Registered cosine | 02 | 2,199 | growth_steps_mean | -7.2584 | [-8.8823, -5.6519] |
| Registered cosine | 03 | 2,596 | effective_species_mean | -5.2969 | [-5.6379, -4.9497] |
| Registered cosine | 03 | 2,596 | occupied_types_mean | -4.6062 | [-4.9957, -4.2129] |
| Registered cosine | 03 | 2,596 | top1_share_mean | 0.2153 | [0.2005, 0.2303] |
| Registered cosine | 03 | 2,596 | adjacent_total_variation_mean | -0.1084 | [-0.1146, -0.1019] |
| Registered cosine | 03 | 2,596 | growth_steps_mean | -6.7267 | [-7.9718, -5.5353] |
| Global-mapped Bray | 02 | 225 | effective_species_mean | -7.1773 | [-7.9095, -6.4346] |
| Global-mapped Bray | 02 | 225 | occupied_types_mean | -8.4313 | [-9.1797, -7.6764] |
| Global-mapped Bray | 02 | 225 | top1_share_mean | 0.3069 | [0.2668, 0.3460] |
| Global-mapped Bray | 02 | 225 | adjacent_total_variation_mean | -0.1779 | [-0.1944, -0.1620] |
| Global-mapped Bray | 02 | 225 | growth_steps_mean | -16.2832 | [-19.5978, -13.1159] |
| Global-mapped Bray | 03 | 249 | effective_species_mean | -6.6611 | [-7.3881, -5.9598] |
| Global-mapped Bray | 03 | 249 | occupied_types_mean | -7.7938 | [-8.5368, -7.0382] |
| Global-mapped Bray | 03 | 249 | top1_share_mean | 0.2980 | [0.2605, 0.3352] |
| Global-mapped Bray | 03 | 249 | adjacent_total_variation_mean | -0.1571 | [-0.1716, -0.1422] |
| Global-mapped Bray | 03 | 249 | growth_steps_mean | -12.4931 | [-14.8373, -10.2542] |
| Relation-mapped Bray | 02 | 578 | effective_species_mean | -6.3132 | [-6.8760, -5.7617] |
| Relation-mapped Bray | 02 | 578 | occupied_types_mean | -6.7270 | [-7.3217, -6.1607] |
| Relation-mapped Bray | 02 | 578 | top1_share_mean | 0.2513 | [0.2231, 0.2804] |
| Relation-mapped Bray | 02 | 578 | adjacent_total_variation_mean | -0.1419 | [-0.1544, -0.1299] |
| Relation-mapped Bray | 02 | 578 | growth_steps_mean | -12.1180 | [-14.3484, -10.0163] |
| Relation-mapped Bray | 03 | 829 | effective_species_mean | -5.9563 | [-6.4181, -5.5029] |
| Relation-mapped Bray | 03 | 829 | occupied_types_mean | -6.4107 | [-6.8882, -5.9552] |
| Relation-mapped Bray | 03 | 829 | top1_share_mean | 0.2439 | [0.2178, 0.2706] |
| Relation-mapped Bray | 03 | 829 | adjacent_total_variation_mean | -0.1383 | [-0.1487, -0.1283] |
| Relation-mapped Bray | 03 | 829 | growth_steps_mean | -9.7182 | [-11.3694, -8.0440] |

Each event is matched without replacement to a negative branch from the same natural state that nevertheless reached a post-break inherited run of eight. Event windows use the earliest qualifying strict window; controls use the earliest eligible run-8 precursor. Intervals resample catalytic matrices.

## Target-specific prediction

| Evaluation target | Fit target | Cell | Log-loss gain | 95% CI | Holm p | Gate |
|---|---|---:|---:|---:|---:|---|
| Registered cosine | Registered cosine | 02-A | 0.00052 | [-0.00006, 0.00120] | 0.1108 | no pass |
| Registered cosine | Registered cosine | 02-B | 0.00028 | [-0.00023, 0.00082] | 0.1555 | no pass |
| Registered cosine | Registered cosine | 03-A | 0.00039 | [0.00010, 0.00073] | 0.0361 | pass |
| Registered cosine | Registered cosine | 03-B | 0.00036 | [0.00003, 0.00068] | 0.0557 | no pass |
| Global-mapped Bray | Registered cosine | 02-A | -0.00017 | [-0.00062, 0.00027] | not tested | transfer control |
| Global-mapped Bray | Registered cosine | 02-B | -0.00027 | [-0.00071, 0.00019] | not tested | transfer control |
| Global-mapped Bray | Global-mapped Bray | 02-A | -0.00004 | [-0.00013, 0.00004] | 1.0000 | no pass |
| Global-mapped Bray | Global-mapped Bray | 02-B | -0.00004 | [-0.00014, 0.00007] | 1.0000 | no pass |
| Global-mapped Bray | Registered cosine | 03-A | -0.00005 | [-0.00026, 0.00019] | not tested | transfer control |
| Global-mapped Bray | Registered cosine | 03-B | -0.00012 | [-0.00033, 0.00011] | not tested | transfer control |
| Global-mapped Bray | Global-mapped Bray | 03-A | 0.00002 | [-0.00001, 0.00005] | 0.5028 | no pass |
| Global-mapped Bray | Global-mapped Bray | 03-B | 0.00001 | [-0.00002, 0.00004] | 0.9541 | no pass |
| Relation-mapped Bray | Registered cosine | 02-A | -0.00009 | [-0.00051, 0.00032] | not tested | transfer control |
| Relation-mapped Bray | Registered cosine | 02-B | -0.00026 | [-0.00071, 0.00019] | not tested | transfer control |
| Relation-mapped Bray | Relation-mapped Bray | 02-A | -0.00001 | [-0.00014, 0.00012] | 1.0000 | no pass |
| Relation-mapped Bray | Relation-mapped Bray | 02-B | -0.00004 | [-0.00020, 0.00012] | 1.0000 | no pass |
| Relation-mapped Bray | Registered cosine | 03-A | 0.00002 | [-0.00022, 0.00028] | not tested | transfer control |
| Relation-mapped Bray | Registered cosine | 03-B | -0.00003 | [-0.00026, 0.00022] | not tested | transfer control |
| Relation-mapped Bray | Relation-mapped Bray | 03-A | -0.00000 | [-0.00011, 0.00011] | 1.0000 | no pass |
| Relation-mapped Bray | Relation-mapped Bray | 03-B | -0.00001 | [-0.00010, 0.00008] | 1.0000 | no pass |

Positive gain means the original no-PCA `h10 + state` model has lower held-out log loss than `h10` alone. Each target-specific model suite was refit on development labels and sealed before the new relation-specific confirmation labels were scored. Holm adjustment applies only to the four target-matched candidate-by-half cells per endpoint; cosine-trained rows are transfer controls, not additional hypothesis tests.

## Endpoint reliability

| Endpoint | Candidate | Split-half Spearman | Matrix-centered Spearman |
|---|---:|---:|---:|
| Registered cosine | 02 | 0.624 | 0.221 |
| Registered cosine | 03 | 0.647 | 0.229 |
| Global-mapped Bray | 02 | 0.305 | 0.066 |
| Global-mapped Bray | 03 | 0.320 | 0.019 |
| Relation-mapped Bray | 02 | 0.443 | 0.145 |
| Relation-mapped Bray | 03 | 0.499 | 0.070 |

## Interpretation

- A better relation-specific Bray match would show that much of the previous metric sensitivity came from forcing one global percentile map onto three different geometric relations. It would not make cosine and Bray equivalent.
- A retained target-matched prediction gain would show that present-state information predicts that metric's event beyond the fixed history block. It would not establish a causal mechanism.
- The non-degeneracy results determine whether coherence is typically associated with compositional collapse. Concentration can be a mechanism rather than an artefact, but it changes the biological interpretation.
- Every result concerns parent-to-one-selected-daughter lineage continuity. It does not establish fidelity of both daughters or whole-population reproduction.

## Reproducibility and claim boundary

The model seal is `e35eb620079b0ea38545eac2c5f9d8220459e7edf326c896e0de63722e5c41d1`. All replay labels are checked against the archived registered cosine labels and onsets; the globally mapped Bray confirmation labels and onsets are also checked against the prior sensitivity audit. The analysis uses 4,096 matrix bootstraps and 4,096 matrix-block randomizations. This remains a post-hoc diagnostic of simulated selected-lineage geometry and predictability, not an intervention, causal test, or new prospective confirmation.

## Files

- `artifacts/output/event_characteristics.csv.gz`: one row per strict event.
- `artifacts/output/event_nondegeneracy_summary.csv`: all-event summaries.
- `artifacts/output/matched_event_control_pairs.csv.gz`: exact matched pairs and differences.
- `artifacts/output/prediction_comparisons.csv`: target-matched and transfer prediction tests.
- `artifacts/output/gate_waterfall.csv`: cumulative and terminal failure gates.
- `artifacts/output/figures/`: calibration, gate, overlap, non-degeneracy, and prediction plots.
