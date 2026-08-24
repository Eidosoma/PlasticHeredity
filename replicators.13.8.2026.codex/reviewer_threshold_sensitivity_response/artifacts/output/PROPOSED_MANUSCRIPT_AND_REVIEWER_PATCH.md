# Proposed manuscript additions (not applied)

## Methods insertion: Exploratory endpoint-definition sensitivity

After all registered analyses, we froze a post-hoc no-refit sensitivity protocol. The F12 family crossed strict parent-daughter inheritance thresholds 0.85, 0.88, 0.90, 0.92 and 0.95; horizons 8, 10, 12 and 16; and renewal lengths two, three and four. Archived full and direct-history predictions were applied unchanged. A separate F32 grid crossed coupled adjacent/all-pairs thresholds 0.88, 0.90 and 0.92; run lengths seven, eight and nine; and inclusive old-anchor thresholds 0.80, 0.85 and 0.90. Candidates and fixed branch halves remained separate, and uncertainty resampled whole catalytic matrices. F16 used a deterministic extension of the original seed streams. CR1 arms were rescored without changing their edits or common random streams.

## Results insertion

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


## Replacement for Limitation 3

> The inheritance, horizon, run-length, coherence, and old-anchor choices remain operational rather than uniquely validated. A post-hoc, no-refit replay and deterministic-extension sensitivity across nearby definitions is reported in Appendix X; it tests local qualitative stability but does not convert any alternate definition into a confirmatory endpoint.

## Reviewer response

We agree that acknowledging operational choices without showing their local consequences was insufficient. We added empirical parent-daughter and independent-lineage H reference distributions and complete, no-refit F12 and strict-F32 sensitivity grids. The appendix reports prevalence, ordinary and matrix-centered branch-half reliability, frozen predictor advantage, and CR1 intervention direction without selecting favorable combinations. We also distinguish exact rescoring through F12 from deterministic extension to F16 and retain the post-hoc limitation explicitly.
