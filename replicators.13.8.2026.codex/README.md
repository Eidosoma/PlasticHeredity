# Plastic-heredity discovery replication

This folder contains a completed clean-room replication of the potential new discovery in the supplied pre-print: a past-observable state/graph/history coordinate for the probability that compositional heredity breaks and then renews. PhiID reconstruction and first-replicator prediction remain out of scope. Interventions were outside the completed replication campaigns summarized below; a newly commissioned, separately governed intervention program is described below and has no result yet.

The authoritative chronological overview is [RESULTS_LEDGER.md](RESULTS_LEDGER.md). It tracks the original replication, 5× rerun, every review-driven correction, all current claim decisions, and links to the sealed evidence underneath each conclusion.

The completed 1× verdict is in [results/full/REPLICATION_RESULTS.md](results/full/REPLICATION_RESULTS.md). The nested 5× rerun and direct convergence audit are in [results/scaled5/REPLICATION_RESULTS.md](results/scaled5/REPLICATION_RESULTS.md) and [results/scaled5/SCALE_COMPARISON.md](results/scaled5/SCALE_COMPARISON.md).

The first [prospective mechanistic-ablation result](results/mechanistic_confirmation/MECHANISTIC_RESULTS.md) separated unique history, growth-clock history, current composition, static catalytic-network structure, and mass-free network-conditioned state. Current state and the network-conditioned block passed that registered `MECHCONF` representation, while static network structure did not.

A subsequent beta-completeness review found that the first ablation compressed 87 distinct beta directions to 12 unsupervised PCs. The corrected workflow sealed a [provenance-complete, no-PCA registration](results/beta_complete_registration/DEVELOPMENT_AUDIT.md), applied it to old MECHCONF only as a [post-hoc diagnostic](results/beta_complete_diagnostic/DIAGNOSTIC_RESULTS.md), and then ran [MECHCONF2 on 200 untouched matrices](results/beta_complete_confirmation/BETA_COMPLETE_RESULTS.md). None of the state, comprehensive-beta, or beta-conditioned-state contrasts passed the corrected prospective gates. The original composite predictor remains a valid registered algorithm comparison, but its mechanistic decomposition is unresolved.

The completed [prospective inheritance-dependence result](results/memory_confirmation/MEMORY_RESULTS.md) responds to the L44 IID-support review. A [retrospective diagnostic](results/memory_diagnostic/MEMORY_DIAGNOSTIC.md) first regenerated all 128,000 retained 5× futures exactly and measured the old fitting-support bias. A separately [sealed protocol](results/memory_registration/MEMORY_REGISTRATION.md) then tested support-matched IID, first-order Markov, and duration-aware semi-Markov models on 200 untouched matrices with 32-fission futures. First-order and duration-aware predictive dependence passed in both candidates. This is evidence of statistical sequence dependence, not biological memory, error correction, molecular storage, or causality.

A reviewer correctly noted that three inherited fissions do not necessarily form one coherent new compositional regime. The [post-hoc episode-coherence audit](results/episode_coherence_audit/EPISODE_COHERENCE_AUDIT.md) regenerated all 145,516 positive episodes from the scaled5, MECHCONF, and MECHCONF2 confirmations. Only 4.5–6.5% placed all three daughters in one mutually `H>0.9` neighbourhood. The supported target is therefore a **break-and-renewal event**, not confirmed regime switching. A shorter explanation is available in the [lay summary](results/episode_coherence_audit/LAY_SUMMARY.md).

The prospective target is fixed as:

> Within the next 12 fissions, observe a strict parent-to-selected-daughter inheritance break (`H <= 0.9`) and subsequently certify three consecutive inherited fissions (`H > 0.9`).

This endpoint does not require episode-wide coherence, distinctness from the old composition, recurrence, or persistence beyond three fissions.

The subsequent strict-regime campaign is implemented separately in `plastic_heredity.regime_prediction`. Its primary endpoint is deliberately narrower and harder: an inheritance break followed by eight inherited fissions whose daughters are mutually coherent (`H > 0.90`) and all distinct from the pre-break parent (`H <= 0.85`). It decomposed the event into break, later eight-run, and strict-geometry stages; added analytic local drift, noise, stability, and recent-velocity summaries; and compared a fixed six-family model menu on 80 new matrices × 128 futures. The pilot had adequate event power and four families improved both candidates and branch halves, but direct ridge and hurdle bootstrap wins split 43.63%/55.44%; neither reached the registered 75% stability gate, so no family was frozen and the original 200-matrix confirmation was blocked.

A pilot-derived equal-probability direct-plus-hurdle ensemble was then treated as a new hypothesis, not a rescue of the failed pilot. It was [separately registered](results/regime_ensemble_registration/REGISTRATION.md) without refitting or recalibration and tested on 200 untouched matrices and 256,000 futures. The [ensemble confirmation](results/regime_ensemble_confirmation/CONFIRMATION_REPORT.md) passed both candidate-03 halves but failed both candidate-02 halves, so its required all-candidate verdict was false. Strict-event occurrence was again observed at about 1.71%/2.12% in candidates 02/03. These workflows test prediction, not causal control or general attractor switching.

The [current research directive](FULL_FABLE_REPLICATION_INSTRUCTIONS.md) now closes further strict-eight predictor search and opens a distinct clean-room replication of Fable's causal-intervention findings on the already validated F12 `JOINT_BREAK_RUN3` target. It requires Codex's own contracts and frozen F12 predictor, fresh seed domains, pre-outcome sealing, whole-matrix inference and exact replay; importing Fable code, models, matrices, states, seeds or selected edits is prohibited. The directive covers one-shot and graded molecular control, an externally specified catalytic-support rule, fixed-composition beta surgery, resistance/resilience decomposition, transfer and a predicted null, closed-loop steering, release/challenge, hysteresis, feedback cost and exploratory internalization. It is a protocol, not evidence that any Fable causal claim has replicated.

The pipeline generates independent development and untouched confirmation catalytic matrices, restores five post-fission states per trajectory, shoots independent futures, and evaluates frozen candidate-separated models against separately sampled confirmation branch halves. The original predictor uses PCA/ridge; the beta-completeness correction uses provenance-selected, no-PCA sequential offset ridge.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest
MPLCONFIGDIR=/tmp/plastic-heredity-mpl .venv/bin/python -m plastic_heredity --profile quick --output results/quick
MPLCONFIGDIR=/tmp/plastic-heredity-mpl .venv/bin/python -m plastic_heredity --profile full --output results/full --workers 12
MPLCONFIGDIR=/tmp/plastic-heredity-mpl .venv/bin/python -m plastic_heredity --profile scaled5 --output results/scaled5 --workers 14
.venv/bin/python -m scripts.compare_scales results/full results/scaled5
.venv/bin/python -m plastic_heredity.mechanistic prepare --source results/scaled5 --registration results/mechanistic_registration
MPLCONFIGDIR=/tmp/plastic-heredity-mpl .venv/bin/python -m plastic_heredity.mechanistic confirm --registration results/mechanistic_registration --output results/mechanistic_confirmation --workers 14
.venv/bin/python -m plastic_heredity.mechanistic_v2 prepare --source results/scaled5 --diagnostic-source results/mechanistic_confirmation --registration results/beta_complete_registration
.venv/bin/python -m plastic_heredity.mechanistic_v2 diagnose --registration results/beta_complete_registration --output results/beta_complete_diagnostic
MPLCONFIGDIR=/tmp/plastic-heredity-mpl-v2 .venv/bin/python -m plastic_heredity.mechanistic_v2 confirm --registration results/beta_complete_registration --output results/beta_complete_confirmation --workers 14
.venv/bin/python -m plastic_heredity.memory diagnose --source results/scaled5 --output results/memory_diagnostic --workers 14
.venv/bin/python -m plastic_heredity.memory prepare --diagnostic results/memory_diagnostic --registration results/memory_registration
MPLCONFIGDIR=/tmp/plastic-heredity-mpl .venv/bin/python -m plastic_heredity.memory confirm --registration results/memory_registration --output results/memory_confirmation --workers 14
.venv/bin/python -m plastic_heredity.episode_coherence --workers 12
.venv/bin/python -m plastic_heredity.regime_prediction diagnose --source results/regime_confirmation --output results/regime_prediction_diagnostic
.venv/bin/python -m plastic_heredity.regime_prediction register-design --output results/regime_prediction_registration
.venv/bin/python -m plastic_heredity.regime_prediction smoke --output results/regime_prediction_smoke --workers 2
# Run costly stages detached. Per-state checkpoints make the same command resumable.
nohup .venv/bin/python -m plastic_heredity.regime_prediction pilot --registration results/regime_prediction_registration --output results/regime_prediction_pilot --work-dir results/.regime_prediction_pilot.work --workers 14 > results/regime_prediction_pilot.log 2>&1 &
.venv/bin/python -m plastic_heredity.regime_prediction status --work-dir results/.regime_prediction_pilot.work
# Confirmation refuses a failed pilot and has its own disjoint checkpoints.
nohup .venv/bin/python -m plastic_heredity.regime_prediction confirm --registration results/regime_prediction_registration --pilot results/regime_prediction_pilot --output results/regime_prediction_confirmation --work-dir results/.regime_prediction_confirmation.work --workers 14 > results/regime_prediction_confirmation.log 2>&1 &
.venv/bin/python -m plastic_heredity.regime_prediction status --work-dir results/.regime_prediction_confirmation.work
# The pilot failed; this separate ensemble registration and confirmation test a new fixed hypothesis.
.venv/bin/python -m plastic_heredity.regime_ensemble_confirmation prepare --design results/regime_prediction_registration --pilot results/regime_prediction_pilot --output results/regime_ensemble_registration
nohup .venv/bin/python -m plastic_heredity.regime_ensemble_confirmation confirm --registration results/regime_ensemble_registration --output results/regime_ensemble_confirmation --work-dir results/.regime_ensemble_confirmation.work --workers 14 > results/regime_ensemble_confirmation.log 2>&1 &
.venv/bin/python -m plastic_heredity.regime_ensemble_confirmation status --work-dir results/.regime_ensemble_confirmation.work
.venv/bin/python -m plastic_heredity.regime_ensemble_confirmation verify --registration results/regime_ensemble_registration
```

The full profile uses 40 shared confirmation matrices, two explicit simulator candidates, five landmarks, 64 futures per state, 4,096 matrix bootstraps, 512 whole-matrix permutations, and exact regeneration. It produces three discovery-only figures, state and branch tables, numerical comparisons, a machine-readable manifest, and a concise results report.

The `scaled5` profile preserves that design but uses 200 development and 200 confirmation matrices. Its matrix IDs and seeds nest the full profile, so matrices 0-39 reproduce the original sample and matrices 40-199 provide the fivefold extension.

`scripts.compare_scales` verifies that nested identity at the raw feature, target, and branch-record levels, then writes a machine-readable scale audit and side-by-side metric table without refitting either run.

The mechanistic workflow is deliberately two-stage. `prepare` refuses to overwrite an existing registration, reconstructs the 200-matrix development trajectories without reshooting outcomes, verifies all retained features and 64,000 target rows, fits the complete ablation suite, and seals code/input/model hashes. `confirm` refuses to run if any registered input or scientific source changed, then evaluates the frozen suite on 200 new matrices and 128,000 futures with exact regeneration. The existing `full` and `scaled5` artifacts are never modified.

The beta-completeness workflow is deliberately three-stage. `prepare` assigns explicit provenance, constructs a fixed threshold-free beta panel including the complete normalized singular spectrum, retains every unique nonconstant direction without PCA, selects ridge penalties using whole-matrix development folds, and seals a new seed domain. `diagnose` applies the seal to old MECHCONF without changing it or making a claim. `confirm` generates and exactly replays 128,000 MECHCONF2 futures. Existing bundles remain immutable.

The memory workflow is deliberately three-stage. `diagnose` regenerates the existing 12-fission confirmation sequences and quantifies the mismatched-IID sensitivity without making a claim. `prepare` seals common transition support, whole-matrix cross-fitting, fixed duration bins, Beta(1,1) smoothing, matrix inference, multiplicity correction, gates, and a new seed domain. `confirm` verifies the seal before generating 128,000 untouched 32-fission futures, exactly replays every variable-length sequence, and reports transition-weighted gains plus equal-state robustness summaries. Existing result bundles remain immutable.

The episode-coherence workflow is explicitly post-hoc. It verifies the three source bundles, deterministically rebuilds their 6,000 restored states, replays every archived positive F12 future, and measures continuous episode geometry plus a fixed threshold-sensitivity grid. Cohorts, candidates, and branch halves are reported separately with whole-matrix bootstrap intervals. It creates no new cohort, changes no endpoint, and cannot establish a distinct regime prospectively.

The strict-regime prediction workflow is four-stage and uses seed domains disjoint from every earlier campaign. `diagnose` is read-only and post-hoc. `register-design` seals source, features, endpoints, model menu, seeds, stop rule, and inference before pilot generation. `pilot` generated and exactly replayed 102,400 F32 futures, recorded every eligible eight-run window, performed whole-matrix cross-fitting, and stopped because no common family was selected stably. Its `confirm` stage was therefore never authorized. Generation and replay use separate source/experiment/state-bound per-state checkpoints; rerunning the same command resumes completed states, while `status` only reads their progress. Checkpoints are retained rather than automatically deleted. The old `regime_confirmation` code and artifacts were not modified.

The separate `regime_ensemble_confirmation` workflow sealed the pilot-developed `0.5 × direct + 0.5 × hurdle` ensemble before any new matrices were generated, verified the frozen models and registration, then generated and exactly replayed 256,000 untouched F32 futures. Its candidate-03 gains passed interval and Holm gates, its candidate-02 gains did not, and the four-cell primary gate failed. The pilot remains a registered failure regardless of this later test.

`results/full/frozen_models.npz` preserves every fitted scaler, PCA transform, class prior, and logistic coefficient; `model_contract.json` maps the archive to the 195 state/graph and nine history feature names.
The public `predict_frozen_archive` helper applies that archive without refitting, and each result bundle's `SHA256SUMS` covers every delivered artifact.

See [REPLICATION.md](REPLICATION.md) for the clean-room boundary and every inferred choice.

To extract only the three supplied discovery-reference figures from the base64 HTML, run `.venv/bin/python -m scripts.extract_discovery_figures`. To reconstruct the portable model archive from retained development arrays, run `.venv/bin/python -m scripts.refreeze_models results/full`.
