# Sequence-history comparator analysis report

**Status:** Reviewer-prompted post-hoc rescore.  No new confirmation futures,
model recalibration, manuscript edit, or pooling across candidates/halves was
performed.  Protocol ID: `bfeff2ea8e4b6d32c87e04b0887a121f658d3ce34a55315ca9ef125dd47b9913`.

## Executive finding

The frozen composite retained a statistically supported log-loss advantage over the development-selected ordered-history ridge in every candidate, branch half, and clean-room implementation.

This supports incremental predictive content beyond the tested ordered sequence history, while remaining a post-hoc robustness result rather than a new prospective test.

The Appendix C transition gains are not numerically commensurate with this
analysis: Appendix C scores bits per realized post-break transition, whereas
this report scores nats per complete F12 future from information available at
launch.

## Primary composite-versus-lagged comparison

| Implementation | Candidate | Half | Gain (nats) | 95% matrix CI | Raw p | Holm p | Cell gate |
|---|---|---|---|---|---|---|---|
| IT1 Codex | 2 | A | 0.01990 | [0.01399, 0.02559] | 0.00024 | 0.00195 | PASS |
| IT1 Codex | 2 | B | 0.01960 | [0.01355, 0.02576] | 0.00024 | 0.00195 | PASS |
| IT1 Codex | 3 | A | 0.02440 | [0.01659, 0.03209] | 0.00024 | 0.00195 | PASS |
| IT1 Codex | 3 | B | 0.02175 | [0.01378, 0.02950] | 0.00024 | 0.00195 | PASS |
| IT2 Fable | 2 | A | 0.02984 | [0.02447, 0.03536] | 0.00024 | 0.00195 | PASS |
| IT2 Fable | 2 | B | 0.02781 | [0.02188, 0.03346] | 0.00024 | 0.00195 | PASS |
| IT2 Fable | 3 | A | 0.03335 | [0.02714, 0.04012] | 0.00024 | 0.00195 | PASS |
| IT2 Fable | 3 | B | 0.03380 | [0.02774, 0.04042] | 0.00024 | 0.00195 | PASS |

Positive gain favors the frozen composite.  The strong claim requires all eight
cells to have positive gain, positive CI lower bound, and Holm-adjusted
`p < 0.05`.

## Development-only model selection

| Cohort | Candidate | Selected lags | C | CV log loss |
|---|---|---|---|---|
| codex_headline | 2 | 10 | 0.01 | 0.51206 |
| codex_headline | 3 | 5 | 0.01 | 0.53220 |
| codex_primary | 2 | 20 | 0.01 | 0.53210 |
| codex_primary | 3 | 20 | 0.01 | 0.54632 |
| fable_headline | 2 | 5 | 0.10 | 0.54246 |
| fable_headline | 3 | 5 | 0.10 | 0.53695 |
| fable_primary | 2 | 40 | 0.10 | 0.55495 |
| fable_primary | 3 | 40 | 0.10 | 0.56239 |

Each lag supplies continuous H, strict-H status, and an observation mask, in
addition to that cohort's registered direct-history variables.  Selection used
five-fold development-matrix-grouped cross-validation only.

## Primary score inventory

| Cohort | Candidate | Half | Model | Log loss | Brier | Spearman | Centered Spearman |
|---|---|---|---|---|---|---|---|
| codex_primary | 2 | A | composite | 0.54481 | 0.18642 | 0.845 | 0.550 |
| codex_primary | 2 | A | direct | 0.57400 | 0.19648 | 0.765 | 0.363 |
| codex_primary | 2 | A | lagged | 0.56471 | 0.19415 | 0.781 | 0.314 |
| codex_primary | 2 | A | markov | 0.65761 | 0.23235 | 0.345 | 0.246 |
| codex_primary | 2 | A | semimarkov | 0.63069 | 0.21966 | 0.499 | 0.250 |
| codex_primary | 2 | B | composite | 0.54434 | 0.18610 | 0.845 | 0.571 |
| codex_primary | 2 | B | direct | 0.57328 | 0.19627 | 0.763 | 0.357 |
| codex_primary | 2 | B | lagged | 0.56394 | 0.19388 | 0.780 | 0.312 |
| codex_primary | 2 | B | markov | 0.65688 | 0.23198 | 0.342 | 0.239 |
| codex_primary | 2 | B | semimarkov | 0.62684 | 0.21786 | 0.504 | 0.266 |
| codex_primary | 3 | A | composite | 0.56276 | 0.19228 | 0.843 | 0.598 |
| codex_primary | 3 | A | direct | 0.59363 | 0.20400 | 0.759 | 0.330 |
| codex_primary | 3 | A | lagged | 0.58716 | 0.20299 | 0.764 | 0.265 |
| codex_primary | 3 | A | markov | 0.68006 | 0.24350 | 0.354 | 0.202 |
| codex_primary | 3 | A | semimarkov | 0.64840 | 0.22807 | 0.507 | 0.237 |
| codex_primary | 3 | B | composite | 0.56023 | 0.19173 | 0.835 | 0.614 |
| codex_primary | 3 | B | direct | 0.59351 | 0.20391 | 0.751 | 0.344 |
| codex_primary | 3 | B | lagged | 0.58198 | 0.20085 | 0.772 | 0.317 |
| codex_primary | 3 | B | markov | 0.67955 | 0.24325 | 0.360 | 0.216 |
| codex_primary | 3 | B | semimarkov | 0.64678 | 0.22730 | 0.501 | 0.251 |
| fable_primary | 2 | A | composite | 0.48718 | 0.16534 | 0.906 | 0.687 |
| fable_primary | 2 | A | direct | 0.53074 | 0.17916 | 0.793 | 0.314 |
| fable_primary | 2 | A | lagged | 0.51702 | 0.17530 | 0.829 | 0.302 |
| fable_primary | 2 | A | markov | 0.65408 | 0.23055 | 0.338 | 0.182 |
| fable_primary | 2 | A | semimarkov | 0.59897 | 0.20464 | 0.534 | 0.228 |
| fable_primary | 2 | B | composite | 0.48956 | 0.16646 | 0.909 | 0.669 |
| fable_primary | 2 | B | direct | 0.53282 | 0.18030 | 0.796 | 0.291 |
| fable_primary | 2 | B | lagged | 0.51736 | 0.17566 | 0.838 | 0.294 |
| fable_primary | 2 | B | markov | 0.65492 | 0.23097 | 0.338 | 0.163 |
| fable_primary | 2 | B | semimarkov | 0.60188 | 0.20602 | 0.527 | 0.194 |
| fable_primary | 3 | A | composite | 0.50798 | 0.17335 | 0.905 | 0.694 |
| fable_primary | 3 | A | direct | 0.55567 | 0.19004 | 0.770 | 0.329 |
| fable_primary | 3 | A | lagged | 0.54133 | 0.18554 | 0.811 | 0.307 |
| fable_primary | 3 | A | markov | 0.67855 | 0.24272 | 0.310 | 0.134 |
| fable_primary | 3 | A | semimarkov | 0.62360 | 0.21620 | 0.477 | 0.186 |
| fable_primary | 3 | B | composite | 0.51189 | 0.17498 | 0.903 | 0.692 |
| fable_primary | 3 | B | direct | 0.55938 | 0.19165 | 0.764 | 0.327 |
| fable_primary | 3 | B | lagged | 0.54569 | 0.18740 | 0.805 | 0.309 |
| fable_primary | 3 | B | markov | 0.67883 | 0.24286 | 0.305 | 0.149 |
| fable_primary | 3 | B | semimarkov | 0.62496 | 0.21685 | 0.474 | 0.198 |

Markov and semi-Markov probabilities were estimated from natural development
paths with an absorbing terminal outcome and integrated exactly over the F12
event.  They are diagnostics motivated by Appendix C; the ordered lagged ridge
is the predeclared primary sequence comparator.

## Secondary and scope qualifications

- The matched 40-matrix headline results are retained in `scores.csv` and
  `comparisons.csv` as secondary replication checks; they cannot rescue the
  primary gate.
- Independent test 1 retains its registered H9 direct block.  Independent test
  2 v2 retains its deduplicated H8 block.  The analysis does not silently force
  one clean room into the other's representation.
- The originating L53/L54 workflow is excluded because its state-level frozen
  predictions and branch outcomes are not present locally.
- Fable v2 did not retain its original float64 branch flags.  Its outcomes were
  reconstructed from the retained float32 H64 arrays and branch lengths.  The
  resulting archived-score discrepancy is at most the value recorded in
  `replay_audit.json` (tolerance `2e-5` nats); no confirmation branch was
  resimulated to remove that representation-level discrepancy.
- This is a robustness rescore on already-observed outcomes, not a new untouched
  confirmation.

## Reproducibility

The isolated folder contains the frozen protocol, per-matrix replay checkpoints,
development-fitted models, complete CV audit, retained-outcome prediction files,
matrix-aware inference, replay audit, figure, verification report, and SHA-256
manifest.  All source artifacts were read-only.
