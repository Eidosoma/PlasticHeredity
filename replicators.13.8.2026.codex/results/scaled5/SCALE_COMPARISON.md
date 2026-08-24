# Nested 1× to 5× scale comparison

The fivefold run preserves the horizon, landmarks, branches per state, candidate contracts, and master seed. Only the number of independent catalytic matrices changes from 40 to 200 in each cohort.

## Design and replay audit

| Cohort | Matrices | States | Futures | Extinct futures | Shared branch rows exact | Shared arrays exact |
|---|---:|---:|---:|---:|---:|---:|
| Development | 40 → 200 | 400 → 2000 | 12800 → 64000 | 0 | True (12800 rows) | True |
| Confirmation | 40 → 200 | 400 → 2000 | 25600 → 128000 | 14 | True (25600 rows) | True |

Exact regeneration of all scaled confirmation futures: **True**.
Portable frozen-model predictions reproduce the saved confirmation predictions within 1e-12: **True**.

## Primary confirmation estimates

| Candidate | Metric | 1× | 5× | Change |
|---|---|---:|---:|---:|
| 02 | branch_half_reliability | 0.9341 | 0.9304 | -0.0037 |
| 02 | centered_branch_half_reliability | 0.6850 | 0.6644 | -0.0205 |
| 02 | full_centered_spearman | 0.5538 | 0.5609 | 0.0071 |
| 02 | history_centered_spearman | 0.3158 | 0.3600 | 0.0442 |
| 02 | full_minus_history_centered | 0.2380 | 0.2009 | -0.0371 |
| 02 | mean_log_loss_gain | 0.0370 | 0.0291 | -0.0080 |
| 02 | minimum_log_loss_gain_lower_95 | 0.0234 | 0.0230 | -0.0004 |
| 02 | mean_q_brier_gain | 0.0127 | 0.0101 | -0.0026 |
| 02 | minimum_q_brier_gain_lower_95 | 0.0094 | 0.0082 | -0.0012 |
| 03 | branch_half_reliability | 0.9270 | 0.9190 | -0.0080 |
| 03 | centered_branch_half_reliability | 0.7219 | 0.6960 | -0.0259 |
| 03 | full_centered_spearman | 0.6107 | 0.6057 | -0.0050 |
| 03 | history_centered_spearman | 0.2708 | 0.3370 | 0.0662 |
| 03 | full_minus_history_centered | 0.3399 | 0.2687 | -0.0712 |
| 03 | mean_log_loss_gain | 0.0350 | 0.0321 | -0.0029 |
| 03 | minimum_log_loss_gain_lower_95 | 0.0182 | 0.0227 | 0.0045 |
| 03 | mean_q_brier_gain | 0.0147 | 0.0120 | -0.0028 |
| 03 | minimum_q_brier_gain_lower_95 | 0.0086 | 0.0096 | 0.0011 |

## Interpretation

The qualitative discovery survives the fivefold increase in independent matrices for both candidates: state-conditioned fate remains reliable, and the frozen full state/graph/history model retains a positive within-matrix and out-of-sample calibration advantage over history alone. The exact manuscript-number signature remains unreproduced under this disclosed clean-room contract.

Descriptive agreement with supplied numerical ranges changed from 11/32 at 1× to 8/32 at 5×. These ranges were never used for fitting or tuning.

See `scale_comparison.csv` for every confirmation and process estimate and `scale_audit.json` for machine-readable nesting checks.
