# Strict-8 prediction-mechanism diagnosis

Status: post-hoc internal diagnosis. This report does not edit the manuscript;
the later-added `SUGGESTED_TEXT.md` translates its verified outputs into
appropriately qualified optional wording.

## Question

Why is the frozen predictor weak or unstable for the strict coherent-eight endpoint, and do starting-composition concentration, residual state/network information, finite-branch label noise, or the endpoint's multi-gate geometry explain it?

## Frozen design

- All three endpoint implementations have equal status: cosine registered, globally calibrated Bray–Curtis, and relation-specific Bray–Curtis.
- Models were fitted on development only and sealed before confirmation scoring.
- Four conditional transitions were evaluated: first break, run of eight after a break, mutual coherence after a run, and old-anchor separation after coherence.
- H, H+C, H+S, and H+C+S compare the retained 10-variable `h10` history block, six exact concentration summaries, and the 26-dimensional state block.
- The fresh intervention reuses all 2,000 retained confirmation states, 11 frozen arms, 64 common-random-stream branches per arm, and 32 future fissions: 1,408,000 newly scored futures.

## Conditional prediction

- anchor_given_coherence, concentration_beyond_history: mean held-out gain 0.00964; 9/12 positive, 0 pass the frozen exploratory gate.
- anchor_given_coherence, residual_state_beyond_concentration: mean held-out gain -0.02068; 2/12 positive, 0 pass the frozen exploratory gate.
- anchor_given_coherence, state_beyond_history: mean held-out gain -0.01709; 4/12 positive, 0 pass the frozen exploratory gate.
- break, concentration_beyond_history: mean held-out gain 0.00788; 12/12 positive, 12 pass the frozen exploratory gate.
- break, residual_state_beyond_concentration: mean held-out gain 0.00036; 9/12 positive, 0 pass the frozen exploratory gate.
- break, state_beyond_history: mean held-out gain 0.00836; 12/12 positive, 12 pass the frozen exploratory gate.
- coherence_given_run8, concentration_beyond_history: mean held-out gain 0.01116; 8/12 positive, 4 pass the frozen exploratory gate.
- coherence_given_run8, residual_state_beyond_concentration: mean held-out gain -0.00176; 6/12 positive, 4 pass the frozen exploratory gate.
- coherence_given_run8, state_beyond_history: mean held-out gain 0.00914; 8/12 positive, 6 pass the frozen exploratory gate.
- run8_given_break, concentration_beyond_history: mean held-out gain 0.00201; 11/12 positive, 2 pass the frozen exploratory gate.
- run8_given_break, residual_state_beyond_concentration: mean held-out gain 0.00063; 10/12 positive, 0 pass the frozen exploratory gate.
- run8_given_break, state_beyond_history: mean held-out gain 0.00260; 11/12 positive, 3 pass the frozen exploratory gate.

## Continuous margins

- anchor_given_coherence, concentration_beyond_history: mean MSE gain 0.000918125; 8/12 positive, 0 pass.
- anchor_given_coherence, residual_state_beyond_concentration: mean MSE gain -0.0012958; 2/12 positive, 0 pass.
- anchor_given_coherence, state_beyond_history: mean MSE gain -0.000551892; 5/12 positive, 0 pass.
- pairwise_given_run8, concentration_beyond_history: mean MSE gain 0.000101074; 6/12 positive, 6 pass.
- pairwise_given_run8, residual_state_beyond_concentration: mean MSE gain -0.000188447; 4/12 positive, 0 pass.
- pairwise_given_run8, state_beyond_history: mean MSE gain -8.16992e-05; 5/12 positive, 4 pass.

## Finite-branch reliability

See `reliability_by_budget.csv` and `figures/transition_reliability.png`. These quantify whether apparent between-state risk differences stabilize as the branch budget rises from 8 to 64, rather than treating a noisy 128-branch label as ground truth.

## Exact strict-window geometry

Every selected event window and its frozen same-state control was exactly replayed. `geometry_window_summary.csv`, `geometry_rank_band_summary.csv`, and the compressed pair table identify which of the 28 pairwise comparisons bind and whether Bray–Curtis distance is carried by the dominant type, ranks 2–5, or the tail.

## Fresh causal perturbation

- evenness_concentrate_minus_flatten: mean event-rate effect -0.0011; 2/12 directional positives, 0 primary passing cells; minimum joint full-dose fraction 0.949.
- richness_contract_minus_expand: mean event-rate effect -0.0006; 3/12 directional positives, 0 primary passing cells; minimum joint full-dose fraction 0.908.

The largest absolute mean localized gate effect is currently richness_contract_minus_expand at run8_given_break (-0.0211). This localization is diagnostic: it distinguishes effects on reaching a renewed run from effects on mutual coherence or separation from the old anchor.

## Interpretation boundary

Prediction gains are held-out associations among retained surviving/observable selected-lineage states. The intervention supplies causal evidence only for the exact one- or four-molecule editing policies, on those same retained states, under the simulator and common random streams. A null result can mean either that the proposed axis is not causal at these doses or that the strict event is too rare/noisy for the available branch budget; the power and reliability tables distinguish those cases where possible.

## Audit trail

The protocol, model seals, exact replay audit, full-dose validation, deterministic replay audit, checksums, and result manifest are all under this subfolder. No new campaign states or new landmark futures were generated; only futures from already retained states were rescored.
