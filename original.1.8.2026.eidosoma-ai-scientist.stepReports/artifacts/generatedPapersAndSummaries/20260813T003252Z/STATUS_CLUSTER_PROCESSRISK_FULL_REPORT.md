# Cluster Report: Process-Risk Confirmation

## Cluster scope

`PROCESSRISK` covers S19-L48, L49, L49R, L50–L54, and the deterministic S20 closeout. It takes the process outcome identified in L44—an inheritance break followed by a newly certified three-fission hereditary episode—and asks whether its finite-horizon probability can be measured efficiently, understood as a switching process, inferred from past-observable state, and confirmed on untouched catalytic matrices.

## Estimand

At each selected post-fission state, define inheritance at a future fission as strict parent-to-selected-daughter cosine `H>0.9`. The primary F12 event is:

1. a future inheritance break (`H<=0.9`); then
2. three consecutive inherited fissions;
3. all within the next 12 fission opportunities.

This is a joint regime-transition event, not first appearance of a replicator. Break probability and resumption conditional on a break are reported separately.

## Measurement and model development

### Shooting is reliable but costly

L48 showed that the conservative branch-half contract required the full 64-future estimator half; a registered adaptive allocation did not improve uniform allocation. This established stochastic shooting as a valid computational measurement, not yet a practical past-only biomarker.

L49 stopped before science when a late landmark lacked 12 future fissions. L49R repaired only the availability rule and retained the failed step unchanged. The repaired analysis found cross-sectional reliability but no universal monotone risk trajectory along molecular-clock landmarks.

### Event-aligned horizons expose process structure

L50 aligned states to fissions 20, 35, 50, 65, and 80 and measured nested F4/F8/F12 probabilities on 80 matrices, 800 states, and 51,200 branches. The joint event was reliable, but shooting did not add robust realized-future value beyond direct history in both candidates.

L51 compared IID, Markov, semi-Markov, and matrix-specific transition models. Duration-dependent switching and longitudinal updating improved one-step transition prediction, while simple registered process models did not reconstruct the empirical committor. Most reliable risk variation was between catalytic matrices.

![Duration-dependent inheritance transition hazards for the two simulator candidates.](figures/process_l51_duration_hazards.png)

*Figure PROCESSRISK-1. L51 duration hazards. The probability that the next fission is inherited depends strongly on both the current regime and its dwell duration. This supports a semi-Markov/regime-switching description, while the irregular tail and between-matrix variation warn against a single universal trajectory.*

L52 used one branch half to learn local duration hazards and predicted the independent half. State-local branch-derived models compressed much of the committor, but by design they were not past observable. They served as a teacher.

### Past-observable distillation

L53 compared three frozen representations:

- nine direct history/phase variables;
- beta-network structure alone;
- a target-blind complete state/catalytic graph representation plus history.

The full representation encoded 195 molecule-permutation-invariant graph/state coordinates, reduced to 12 PCA components using development data, then combined them with the nine history/phase coordinates in a ridge logistic model (`C=0.1`). Beta alone failed. The full model improved held-out branch proper scores and within-matrix ranks in both candidates, generating the `PAST_OBSERVABLE_PROCESS_RISK_LEAD`. Because L53 was adaptive, it required untouched confirmation.

## Untouched L54 confirmation

L54 generated 40 new shared catalytic matrices, 80 primary trajectories, and 400 preregistered post-fission states (five landmarks per matrix and candidate). Each state received 64 independent F12 branches, for 25,600 primary futures, plus an exact second branch campaign for replay. No feature transform, PCA basis, coefficient, prior, target, threshold, horizon, landmark, or candidate rule was refit.

### Measurement reliability

| Candidate | States | Intermediate-probability states | Branch-half Spearman (95% lower) | Within-matrix-centered Spearman (95% lower) |
|---|---:|---:|---:|---:|
| 02 | 200 | 138 | 0.938 (0.903) | 0.625 (0.456) |
| 03 | 200 | 149 | 0.924 (0.872) | 0.606 (0.475) |

The empirical event probability was therefore reproducibly measurable both across matrices and, more modestly, among states within a matrix.

### Frozen predictive transfer

Across the two preregistered branch-half directions, the full model's overall Spearman correlations were 0.895–0.918 and its within-matrix-centered correlations were 0.550–0.697. Direct history ranked risk well overall (0.742–0.822) but only 0.198–0.345 after centering. Beta-only ranks were approximately zero. The full representation improved branch log loss over direct history by 0.041–0.052, with all 95% lower bounds positive, and q-Brier score by 0.012–0.018, again with all lower bounds positive. All 512 whole-matrix permutation controls returned `p=0.001949` in the intended direction.

![Frozen history, beta-only, and full-state graph-plus-history rank transfer on untouched matrices.](figures/process_l54_rank_transfer.png)

*Figure PROCESSRISK-2. L54 rank transfer. Overall panels combine stable between-matrix propensity and changing state risk; centered panels isolate within-matrix ordering. The frozen full-state graph-plus-history coordinate outperforms direct history and beta-only structure in both candidates, including within matrices.*

### Calibration

![Frozen predictions against independently estimated process probabilities.](figures/process_l54_prediction_calibration.png)

*Figure PROCESSRISK-3. L54 prospective calibration view. Each point compares a frozen past-observable prediction with an independent branch-half estimate of the F12 joint-event probability. Scatter remains—especially within matrices—but both candidates show strong monotone transfer without refitting.*

All availability, reliability, proper-score, overall-rank, within-matrix-rank, permutation, and replay gates passed in both candidates. The registered classification was:

- `UNTOUCHED_PAST_OBSERVABLE_PROCESS_RISK_COORDINATE_CONFIRMED`;
- `PLASTIC_HEREDITY_SWITCHING_PROPENSITY_PREDICTABLE`;
- `SIMULATOR_PROCESS_EARLY_WARNING_CONFIRMED`;
- `NOT_PAPER_REPLICATION`.

## What the coordinate is and is not

The confirmed coordinate is a frozen mapping from present integer composition, mass/phase, catalytic-network-conditioned state features, and recent heredity history to the probability of a bounded future break-and-renewal process. It is target-blind in the sense that no completed-run centroid, future branch outcome, Phi value, or intervention result enters its input.

It is **not**:

- Phi or PhiID;
- a reconstruction of the manuscript's Figure 5 task;
- prediction of the first replicator;
- entry into a privileged attractor;
- return to an old molecular composition;
- evidence of homeostasis or error correction;
- an intervention result or causal-control demonstration;
- evidence about physical chemistry outside the reconstructed simulator.

## Validation and scope

- Catalytic matrix is the higher-level independent unit.
- Candidates 02 and 03 remain separate.
- Confirmation used 4,096 matrix bootstraps and 512 whole-matrix permutations.
- Seed roots, matrices, initial states, trajectories, states, and branches were firewalled from development evidence.
- Primary and regeneration campaigns replayed exactly.
- Candidate direction, model, features, preprocessing, and target were frozen before L54 outcomes.
- S20 added no scientific result; it preserved the complete E01/S19 record and closed the experiment deterministically.

## Cluster conclusion

This is the strongest positive finding in the completed program. A network-conditioned, past-observable coordinate predicts the independently measured probability of a future plastic-heredity regime transition on new catalytic matrices in two reconstructed simulator candidates. The result supports a simulator-specific precursor to changing hereditary organization. It does not rescue the paper's PhiID, first-replicator, or causal-control claims.

