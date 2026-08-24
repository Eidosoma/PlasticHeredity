# Exploratory endpoint-definition sensitivity appendix

## Status

This appendix is post-hoc and exploratory. All definitions, metrics, models, contrasts, and figures were frozen before alternate grid results were opened. No model or feature transform was refitted or recalibrated.

F16 is a deterministic extension of the archived F12 seed streams, not pure rescoring of retained F12 trajectories. The independent-lineage reference newly applies the frozen L36 two-lineage, dominant-H090-component-centroid design to scaled5 restored states because the historical L36/L37 raw Parquet trajectories were not retained locally.

## Empirical H context

| Distribution | Candidate | n | q05 | Median | q95 | Fraction at or below focal cutoff |
|---|---:|---:|---:|---:|---:|---:|
| Parent-to-selected-daughter (cutoff 0.90) | 02 | 767,918 | 0.878 | 0.960 | 0.997 | 0.107 |
| Parent-to-selected-daughter (cutoff 0.90) | 03 | 767,995 | 0.871 | 0.960 | 0.997 | 0.117 |
| Between independent reference-lineage centroids (cutoff 0.85) | 02 | 999 | 0.132 | 0.979 | 0.999 | 0.266 |
| Between independent reference-lineage centroids (cutoff 0.85) | 03 | 1,000 | 0.140 | 0.976 | 0.999 | 0.290 |

Figure S1 shows the complete empirical reference distributions. The independent-lineage CDF supplies a dataset-specific scale for the 0.85 distinctness cutoff despite the high cosine floor of nonnegative composition vectors. No formal claim that the boundary-H distribution is bimodal, or that 0.90 was originally selected from an empirical antimode, is made.

## F12-family sensitivity

Across the local neighborhood (`H=0.88-0.92`, F10-F16), prevalence spans 0.171-0.677, centered split-half reliability spans 0.492-0.746, and mean frozen full-over-history log-loss gain spans 0.0165-0.0414.

The complete 60-definition table is retained; stress thresholds 0.85 and 0.95 are not omitted. Figures S2-S3 display all candidates and renewal lengths separately.

### Local qualitative stability

| Candidate | Full-history gain > 0, half A | Full-history gain > 0, half B | 95% CI entirely > 0, A/B |
|---|---:|---:|---:|
| 02 | 27/27 | 27/27 | 27/27; 27/27 |
| 03 | 27/27 | 27/27 | 26/27; 27/27 |

### Registered F12 baseline readback

| Candidate | Prevalence | Centered reliability | Log-loss gain A/B |
|---|---:|---:|---:|
| 02 | 0.3711 | 0.6644 | 0.02919 / 0.02894 |
| 03 | 0.4167 | 0.6960 | 0.03087 / 0.03327 |

## CR1 intervention-direction sensitivity

Across the same local neighborhood, MODEL_UP minus MODEL_DOWN spans 0.0589-0.1411 across candidates and fixed branch halves. All four registered contrasts and all 60 endpoint definitions are reported in the machine-readable table.

| Candidate | Half | MODEL_UP - MODEL_DOWN > 0 | 95% CI entirely > 0 |
|---|---:|---:|---:|
| 02 | A | 27/27 | 27/27 |
| 02 | B | 27/27 | 27/27 |
| 03 | A | 27/27 | 27/27 |
| 03 | B | 27/27 | 27/27 |

### Registered CR1 baseline readback

| Candidate | Half | MODEL_UP - MODEL_DOWN [95% CI] |
|---|---:|---:|
| 02 | A | 0.1233 [0.1114, 0.1359] |
| 02 | B | 0.1141 [0.0999, 0.1279] |
| 03 | A | 0.1123 [0.0992, 0.1249] |
| 03 | B | 0.1071 [0.0936, 0.1200] |

## Strict-F32 sensitivity

The strict grid varies its coupled adjacent/all-pairs threshold, run length, and old-anchor cutoff without changing horizon, inputs, or predictor. It is descriptive and cannot rescue the registered predictor or the later failed ensemble.

| Candidate | h10+state gain > 0, half A | h10+state gain > 0, half B | Prevalence range | Centered reliability range |
|---|---:|---:|---:|---:|
| 02 | 27/27 | 27/27 | 0.009-0.032 | 0.086-0.227 |
| 03 | 27/27 | 27/27 | 0.011-0.036 | 0.102-0.251 |

| Candidate | Baseline prevalence | Centered reliability | h10+state over h10 gain A/B |
|---|---:|---:|---:|
| 02 | 0.0184 | 0.2207 | 0.000525 / 0.000284 |
| 03 | 0.0210 | 0.2294 | 0.000388 / 0.000356 |

The separately registered direct-plus-hurdle ensemble remains a failed all-candidate hypothesis: its existing baseline confirmation passed both candidate-03 halves and failed both candidate-02 halves. It was not refit or used to select an alternate definition here.

## Interpretation boundary

The tables describe whether estimates vary continuously around the registered definitions. They do not establish that any one cutoff is natural, prospectively validate an alternate endpoint, or license selection of the most favorable cell. F12 causal effects do not establish causal control of the strict F32 event.
