# Chapter 5 Φ-r / plastic-heredity pilot report

This is the pre-specified 24-matrix pilot. It estimates directions and feasibility; it is not the independent 48-matrix confirmation.

Registration: `9bf0cfc0050726d0fb6893cfa6f1789363612b50a6154a399477e3748f7726cf`. Candidates were analyzed separately and the catalytic matrix was the inference unit.

## Six-arm causal bridge

Contrasts are stabilizing minus destabilizing for model/rule arms. The table uses the final 30 fissions.

| Contrast | Reading | Candidate | Replicate | Mean [95% matrix CI] | Holm p |
| --- | --- | --- | ---: | ---: | ---: |
| model | inherited | 02 | 0 | +0.1583 [+0.0917, +0.2306] | 0.0009763 |
| model | inherited | 02 | 1 | +0.1306 [+0.0750, +0.1925] | 0.0009763 |
| model | inherited | 03 | 0 | +0.1903 [+0.1222, +0.2611] | 0.0009763 |
| model | inherited | 03 | 1 | +0.1917 [+0.1227, +0.2653] | 0.0009763 |
| model | molecular_revised | 02 | 0 | -0.1348 [-0.2057, -0.0598] | 1 |
| model | molecular_revised | 02 | 1 | -0.1817 [-0.2748, -0.0909] | 1 |
| model | molecular_revised | 03 | 0 | -0.1710 [-0.2558, -0.0862] | 1 |
| model | molecular_revised | 03 | 1 | -0.2441 [-0.3178, -0.1723] | 1 |
| model | molecular_typeset | 02 | 0 | +3.6333 [+1.5927, +5.6854] | 0.002929 |
| model | molecular_typeset | 02 | 1 | +5.3047 [+2.9975, +7.3912] | 0.0009763 |
| model | molecular_typeset | 03 | 0 | +4.1050 [+1.6734, +6.2779] | 0.002929 |
| model | molecular_typeset | 03 | 1 | +4.8090 [+2.6485, +6.9949] | 0.002197 |
| model | generational_revised | 02 | 0 | -0.0660 [-0.1110, -0.0254] | 1 |
| model | generational_revised | 02 | 1 | -0.0290 [-0.0698, +0.0129] | 1 |
| model | generational_revised | 03 | 0 | -0.0347 [-0.0763, +0.0093] | 1 |
| model | generational_revised | 03 | 1 | -0.0375 [-0.0791, +0.0044] | 1 |
| rule | molecular_revised | 02 | 0 | -0.1099 [-0.1832, -0.0315] | 1 |
| rule | molecular_revised | 02 | 1 | -0.1891 [-0.2566, -0.1189] | 1 |
| rule | molecular_revised | 03 | 0 | -0.1241 [-0.1878, -0.0576] | 1 |
| rule | molecular_revised | 03 | 1 | -0.0649 [-0.1362, +0.0099] | 1 |

- Frozen-control heredity validity gate: **True**.
- Revised Φ-r response gate: **False**.

## Hereditary-state reading

The state contrast is within-lineage: readings during a trailing inherited run of at least five fissions minus all other states.

- Revised Φ-r state-reading gate: **False**.

## Prospective foresight

All correlations are centered within catalytic matrix and kept separate by candidate and fixed branch half.

| Predictor/reading | Candidate | Half | Centered Spearman [95% matrix CI] |
| --- | --- | --- | ---: |
| frozen_prediction | 02 | A | +0.519 [+0.380, +0.636] |
| frozen_prediction | 02 | B | +0.603 [+0.474, +0.691] |
| frozen_prediction | 03 | A | +0.592 [+0.461, +0.699] |
| frozen_prediction | 03 | B | +0.587 [+0.418, +0.715] |
| molecular_revised | 02 | A | +0.062 [-0.055, +0.181] |
| molecular_revised | 02 | B | -0.002 [-0.104, +0.106] |
| molecular_revised | 03 | A | +0.093 [-0.019, +0.195] |
| molecular_revised | 03 | B | +0.120 [+0.006, +0.234] |
| generational_revised | 02 | A | +0.123 [+0.004, +0.240] |
| generational_revised | 02 | B | +0.063 [-0.046, +0.169] |
| generational_revised | 03 | A | -0.125 [-0.257, +0.006] |
| generational_revised | 03 | B | -0.139 [-0.263, -0.015] |

- Frozen-predictor validity gate: **True**.
- Revised Φ-r ±0.10 correlation-equivalence gate: **False**.
- Incremental Φ log-loss ±0.005 equivalence gate: **False**.

## Bounded Φ-directed probe

The selector screened a fixed 64-edit set with four short common-random-stream probes, then confirmed the selected extremes on fresh streams.

| Outcome | Candidate | Replicate | Φ-up minus Φ-down [95% matrix CI] |
| --- | --- | ---: | ---: |
| molecular_revised | 02 | 0 | -0.0360 [-0.2000, +0.1371] |
| molecular_revised | 02 | 1 | -0.1395 [-0.2972, +0.0242] |
| molecular_revised | 03 | 0 | +0.0417 [-0.0888, +0.1706] |
| molecular_revised | 03 | 1 | -0.1012 [-0.2238, +0.0326] |
| inherited_fraction | 02 | 0 | -0.0208 [-0.0538, +0.0115] |
| inherited_fraction | 02 | 1 | -0.0087 [-0.0399, +0.0289] |
| inherited_fraction | 03 | 0 | +0.0278 [-0.0069, +0.0660] |
| inherited_fraction | 03 | 1 | -0.0139 [-0.0451, +0.0156] |
| joint_break_run3 | 02 | 0 | +0.1250 [-0.0833, +0.3750] |
| joint_break_run3 | 02 | 1 | +0.0000 [-0.2083, +0.2083] |
| joint_break_run3 | 03 | 0 | +0.0000 [-0.2083, +0.2083] |
| joint_break_run3 | 03 | 1 | -0.0833 [-0.2917, +0.1250] |

- Probe moves revised Φ-r gate: **False**.
- Probe heredity-equivalence gate: **False**.

## Dose response and instrument separation

- Revised Φ-r dose gate: **False**.
- The unnormalized typeset equation, the text-extraction ratio, revised Φ-r, all 16 atoms, causation, emergence, synergy-persistence, molecular and generational clocks, and a growth-only sensitivity were retained as distinct readings.

## Integrity and claim boundary

- No-op traced callback exactly matched the plain simulator: **True**.
- Complete generation and replay results are compared matrix by matrix in `replay_audit.json`.
- No raw molecular trace was saved; only rolling-window scores and compact state/outcome records were retained.
- A Φ-r response is an information-statistical gauge response. It is not evidence of consciousness, life, agency, biological memory, a universal origin-of-life mechanism, or a portal to a Platonic space.
- The public PhiRL code belongs to a companion RL paper; this program does not validate the unavailable private GARD-paper pipeline.

## Phase decision

The mandatory next step is human review of this pilot. The software will not create or launch the 48-matrix confirmation without a separate authorization artifact.
