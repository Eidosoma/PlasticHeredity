# E01 ambiguity and discrepancy ledger

- **Ledger version:** `E01-S02-v1.0.0`
- **Specification registry:** `E01-specification-registry-v0.2.0`
- **Research step:** `S02`
- **Ambiguity count:** 105
- **Schema validation:** PASS
- **Registry executable:** `false`
- **No-silent-default rule:** Every primary value is fixed, an explicit branch set, or an unresolved/conflict sentinel.

## Resolution status

| Status | Count | Meaning |
| --- | ---: | --- |
| `CONFLICT_PRESERVED` | 8 | Contradictory source interpretations remain separate and block unqualified execution. |
| `DEFERRED_EVIDENCE` | 14 | Resolution requires evidence owned by a later authorized step. |
| `FROZEN_BRANCH_SET` | 21 | All listed branches are retained; later runs need distinct specification IDs. |
| `PAPER_FIXED` | 4 | Explicit in the supplied paper. |
| `PLAN_FIXED` | 7 | Frozen prospectively by FULL_PLAN. |
| `PROVISIONAL_PRIMARY` | 5 | Explicit provisional choice; must be logged and sensitivity-audited. |
| `RECONCILED` | 1 | Source statements can coexist after preserving the exact scopes or denominators. |
| `UNRESOLVED_REQUIRED` | 45 | Required method value is absent and execution must reject the sentinel. |

## Ambiguity items

| ID | Category | Parameter | Materiality | Status | Primary value or sentinel | Owner | Claims | S01 discrepancies |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| `E01-A001` | source_provenance | `source.paper.complete_formula_source` | critical | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A001` | `S03` | 59 | D08 |
| `E01-A002` | source_provenance | `source.author_code.revision` | critical | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A002` | `S03` | 59 | none |
| `E01-A003` | source_provenance | `source.gard_historical.revision` | high | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A003` | `S03` | 59 | none |
| `E01-A004` | source_provenance | `source.phiid_reference.revision` | critical | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A004` | `S03` | 59 | D08 |
| `E01-A005` | gard_dynamics | `gard.model.variant` | critical | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A005` | `S04` | 59 | none |
| `E01-A006` | gard_dynamics | `gard.beta.distribution_parameterization` | critical | `PLAN_FIXED` | `beta_ij=exp(A+sigma*epsilon_ij), epsilon_ij~N(0,1)` | `S04` | 59 | none |
| `E01-A007` | gard_dynamics | `gard.beta.orientation_and_diagonal` | high | `FROZEN_BRANCH_SET` | `BRANCH_SET::historical_orientation_with_diagonal\|transpose_audit\|diagonal_exclusion_audit` | `S04` | 59 | none |
| `E01-A008` | gard_dynamics | `gard.beta.lifetime_per_run` | high | `PROVISIONAL_PRIMARY` | `one fixed beta per run; same beta across paired intervention treatments` | `S04` | 59 | none |
| `E01-A009` | gard_dynamics | `gard.kinetics.k_f` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A009` | `S04` | 59 | none |
| `E01-A010` | gard_dynamics | `gard.kinetics.k_b` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A010` | `S04` | 59 | none |
| `E01-A011` | gard_dynamics | `gard.environment.rho_i` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A011` | `S04` | 59 | none |
| `E01-A012` | gard_dynamics | `gard.kinetics.propensity_equations` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::historical_reference\|paper_poisson_reconstruction\|modern_gillespie` | `S04` | 59 | none |
| `E01-A013` | gard_dynamics | `gard.update.kernel` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::paper_vector_poisson\|historical_loop\|direct_gillespie` | `S04` | 59 | none |
| `E01-A014` | gard_dynamics | `gard.update.poisson_exposure` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A014` | `S04` | 59 | none |
| `E01-A015` | gard_dynamics | `gard.time.molecular_step_definition` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A015` | `S04` | 59 | none |
| `E01-A016` | gard_dynamics | `gard.update.loss_nonnegativity` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A016` | `S04` | 59 | none |
| `E01-A017` | gard_dynamics | `gard.growth.boundary_handling` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A017` | `S04` | 59 | none |
| `E01-A018` | gard_dynamics | `gard.growth.max_steps_terminal_semantics` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A018` | `S04` | 59 | none |
| `E01-A019` | gard_dynamics | `gard.initial_state.construction` | critical | `PROVISIONAL_PRIMARY` | `40 distinct uniformly sampled types, one molecule per selected type` | `S04` | 59 | none |
| `E01-A020` | gard_dynamics | `gard.initial_state.rng_stream` | medium | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A020` | `S06` | 59 | none |
| `E01-A021` | gard_dynamics | `gard.environment.reservoir_semantics` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A021` | `S04` | 59 | none |
| `E01-A022` | gard_dynamics | `gard.fission.sampling_semantics` | high | `PLAN_FIXED` | `n_i_followed ~ Binomial(n_i_pre,0.5); other daughter is complement` | `S04` | 59 | none |
| `E01-A023` | gard_dynamics | `gard.fission.daughter_selection` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A023` | `S04` | 59 | none |
| `E01-A024` | gard_dynamics | `gard.fission.post_fission_continuation` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A024` | `S04` | 59 | none |
| `E01-A025` | preprocessing | `preprocessing.state_sampling_instant` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A025` | `S06` | 59 | none |
| `E01-A026` | preprocessing | `preprocessing.composition.normalization` | high | `PLAN_FIXED` | `x_i=n_i/sum_j(n_j); zero-total state is invalid and must raise a recorded failure` | `S09` | 59 | none |
| `E01-A027` | preprocessing | `preprocessing.zero.policy` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::pseudocount_grid\|multiplicative_replacement` | `S09` | 59 | none |
| `E01-A028` | preprocessing | `preprocessing.zero.pseudocount_delta` | high | `FROZEN_BRANCH_SET` | `BRANCH_SET::1e-6\|1e-4\|1e-2\|0.1\|0.5\|1` | `S09` | 59 | none |
| `E01-A029` | preprocessing | `preprocessing.clr.logarithm_and_formula` | high | `PROVISIONAL_PRIMARY` | `natural-log CLR of closed pseudocount-adjusted composition` | `S09` | 59 | D08 |
| `E01-A030` | preprocessing | `preprocessing.clr.component_removal` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::drop_last\|drop_each_component\|ILR` | `S09` | 59 | none |
| `E01-A031` | preprocessing | `preprocessing.coordinate_family` | high | `FROZEN_BRANCH_SET` | `BRANCH_SET::CLR\|ILR\|raw_proportions\|Hellinger\|principal_log_ratio` | `S09` | 59 | none |
| `E01-A032` | preprocessing | `preprocessing.feature_orientation_and_scaling` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A032` | `S09` | 59 | none |
| `E01-A033` | preprocessing | `preprocessing.invalid_transform_policy` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A033` | `S09` | 59 | none |
| `E01-A034` | replicator_labels | `labels.similarity.metric` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::historical_H\|Euclidean\|Aitchison` | `S08` | 45 | none |
| `E01-A035` | replicator_labels | `labels.similarity.threshold` | critical | `PLAN_FIXED` | `historical label uses H>0.9; sensitivity grid must be versioned before S08 results` | `S08` | 45 | none |
| `E01-A036` | replicator_labels | `labels.compotype.reference_definition` | critical | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A036` | `S08` | 45 | none |
| `E01-A037` | replicator_labels | `labels.clustering.algorithm_and_hyperparameters` | critical | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A037` | `S08` | 45 | none |
| `E01-A038` | replicator_labels | `labels.persistence_and_homeostasis_rule` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A038` | `S08` | 45 | none |
| `E01-A039` | replicator_labels | `labels.temporal_information_scope` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::paper_retrospective\|past_only_online\|training_derived_reference` | `S08` | 45 | D03 |
| `E01-A040` | replicator_labels | `labels.molecular_step_alignment` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A040` | `S08` | 45 | none |
| `E01-A041` | replicator_labels | `labels.ineligible_and_drift_policy` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A041` | `S08` | 45 | D11 |
| `E01-A042` | replicator_labels | `labels.family_registry` | high | `FROZEN_BRANCH_SET` | `BRANCH_SET::Y_H\|Y_E\|Y_A\|continuous_recurrence\|parent_daughter_fidelity\|basin_return` | `S08` | 45 | none |
| `E01-A043` | phi_estimation | `phi.atom.identity_and_formula` | critical | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A043` | `S03` | 59 | D08 |
| `E01-A044` | phi_estimation | `phi.redundancy.function` | critical | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A044` | `S03` | 59 | D08 |
| `E01-A045` | phi_estimation | `phi.estimator.family` | critical | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A045` | `S10` | 59 | D08 |
| `E01-A046` | phi_estimation | `phi.estimator.bias_and_regularization` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A046` | `S10` | 59 | none |
| `E01-A047` | phi_estimation | `phi.information_units` | medium | `PAPER_FIXED` | `natural logarithm; report nats` | `S10` | 59 | none |
| `E01-A048` | phi_estimation | `phi.window.length` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::32\|64\|128\|256\|whole_trajectory_descriptive` | `S11` | 59 | D08 |
| `E01-A049` | phi_estimation | `phi.lag` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::1\|2\|4\|8` | `S11` | 59 | D08 |
| `E01-A050` | phi_estimation | `phi.window.mode` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::sliding_past_only\|expanding_past_only\|whole_trajectory_descriptive` | `S11` | 59 | D03 |
| `E01-A051` | phi_estimation | `phi.temporal_information_access` | critical | `PLAN_FIXED` | `prospective estimates use s<=t; future access is validation failure` | `S11` | 59 | D03 |
| `E01-A052` | phi_estimation | `phi.temporal_pairing_and_alignment` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A052` | `S11` | 59 | D03 |
| `E01-A053` | phi_estimation | `phi.early_window.policy` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A053` | `S11` | 59 | D03 |
| `E01-A054` | phi_estimation | `phi.mib.objective` | critical | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A054` | `S10` | 59 | D08 |
| `E01-A055` | phi_estimation | `phi.mib.normalization` | critical | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A055` | `S10` | 59 | D08 |
| `E01-A056` | phi_estimation | `phi.partition.search` | critical | `DEFERRED_EVIDENCE` | `UNRESOLVED::E01-A056` | `S10` | 59 | D08 |
| `E01-A057` | phi_estimation | `phi.partition.tie_and_recomputation` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A057` | `S11` | 59 | none |
| `E01-A058` | phi_estimation | `phi.sample_sufficiency_and_failure_policy` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A058` | `S10` | 59 | none |
| `E01-A059` | phi_estimation | `phi.negative_value_policy` | high | `PAPER_FIXED` | `preserve signed Phi-r; derived positive/negative/absolute spike catalogs are separate branches` | `S11` | 59 | D07 |
| `E01-A060` | phi_estimation | `phi.backend_precision_and_cadence` | high | `PLAN_FIXED` | `CPU float64 reference evaluated every molecular step; accelerated results require cross-validation` | `S10` | 59 | none |
| `E01-A061` | descriptive_statistics | `aggregate.variable_length_alignment` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A061` | `S14` | 2 | none |
| `E01-A062` | descriptive_statistics | `aggregate.center_and_spread` | medium | `PAPER_FIXED` | `paper-like summary is median plus/minus sample SD; robust alternatives remain separate` | `S14` | 2 | none |
| `E01-A063` | descriptive_statistics | `aggregate.trend_regression` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A063` | `S14` | 1 | none |
| `E01-A064` | descriptive_statistics | `spike.threshold.scope` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A064` | `S14` | 4 | D07 |
| `E01-A065` | descriptive_statistics | `spike.definition.direction_and_scale` | high | `FROZEN_BRANCH_SET` | `BRANCH_SET::positive_3SD\|negative_3SD\|absolute_3SD\|positive_MAD\|negative_MAD\|absolute_MAD` | `S14` | 4 | D07 |
| `E01-A066` | descriptive_statistics | `spike.episode_and_prevalence_rule` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A066` | `S14` | 4 | D07 |
| `E01-A067` | descriptive_statistics | `spike.reported_run_count` | medium | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A067` | `S14` | 1 | D07 |
| `E01-A068` | descriptive_statistics | `association.phi_level_or_change` | critical | `CONFLICT_PRESERVED` | `CONFLICT::paper_results_level\|figure_caption_first_difference` | `S15` | 4 | D02 |
| `E01-A069` | descriptive_statistics | `association.time_alignment_and_ties` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A069` | `S15` | 7 | D02 |
| `E01-A070` | descriptive_statistics | `association.positive_significant_denominators` | medium | `RECONCILED` | `raw_count=54; total_denominator=100; positive_subset_denominator=73; report both proportions` | `S15` | 2 | D01 |
| `E01-A071` | descriptive_statistics | `inference.alpha_sidedness_and_multiplicity` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::paper_like_alpha0.05_uncorrected\|multiplicity_corrected\|block_aware` | `S15` | 47 | D11;D12 |
| `E01-A072` | descriptive_statistics | `metric_comparison.definition_and_aggregation` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A072` | `S15` | 12 | none |
| `E01-A073` | descriptive_statistics | `state_comparison.mann_whitney_scope` | critical | `CONFLICT_PRESERVED` | `CONFLICT::pooled_steps\|within_run_tests\|run_summary_test` | `S15` | 3 | D11 |
| `E01-A074` | descriptive_statistics | `state_comparison.fisher_combination` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A074` | `S15` | 2 | D11 |
| `E01-A075` | descriptive_statistics | `temporal_structure.ljung_box_and_differencing` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A075` | `S15` | 3 | none |
| `E01-A076` | descriptive_statistics | `spike_timing.observation_unit_and_outcome_mapping` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A076` | `S16` | 3 | D07 |
| `E01-A077` | descriptive_statistics | `spike_height.reported_statistic_and_equivalence` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A077` | `S16` | 1 | D12 |
| `E01-A078` | prediction | `ml.target.endpoint` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::paper_like_state_trajectory\|strict_first_event` | `S16` | 6 | D03 |
| `E01-A079` | prediction | `ml.temporal_split.boundary_unit` | high | `PROVISIONAL_PRIMARY` | `floor(0.25*T) molecular-step boundary, with boundary rule logged` | `S16` | 6 | D03 |
| `E01-A080` | prediction | `ml.sequence.layout` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A080` | `S16` | 6 | D03 |
| `E01-A081` | prediction | `ml.feature_family.definitions` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A081` | `S16` | 6 | D03 |
| `E01-A082` | prediction | `ml.mlp.architecture` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A082` | `S16` | 6 | none |
| `E01-A083` | prediction | `ml.training_and_scaling` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A083` | `S16` | 6 | D03 |
| `E01-A084` | prediction | `ml.run_split.stratification_and_reuse` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A084` | `S16` | 6 | D03 |
| `E01-A085` | prediction | `ml.repetition.bootstrap_and_seed_semantics` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A085` | `S16` | 6 | D03 |
| `E01-A086` | prediction | `ml.dummy_baseline.prevalence_scope` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A086` | `S16` | 2 | D03 |
| `E01-A087` | prediction | `ml.performance_aggregation_and_inference` | critical | `CONFLICT_PRESERVED` | `CONFLICT::micro_vs_macro_weighting\|paired_vs_unpaired_inference` | `S16` | 6 | D03 |
| `E01-A088` | prediction | `ml.sensitivity_and_preonset_policy` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::paper_25_75\|versioned_split_grid\|strict_preonset_first_event` | `S16` | 2 | D03 |
| `E01-A089` | intervention_outcomes | `intervention.action.set` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::paper_plus_minus\|modern_noop_plus_minus_feasible` | `S17` | 26 | none |
| `E01-A090` | intervention_outcomes | `intervention.action.magnitude_and_deletion` | high | `PLAN_FIXED` | `one molecule action; -e_i only when n_i>0; infeasible actions omitted and logged` | `S17` | 26 | none |
| `E01-A091` | intervention_outcomes | `intervention.timing_and_frequency` | critical | `PROVISIONAL_PRIMARY` | `post-fission before next growth at every nonterminal generation; exact boundary count requires manifest` | `S17` | 26 | none |
| `E01-A092` | intervention_outcomes | `intervention.score.horizon` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::append_only_primary\|one_step_expected\|generation_rollout` | `S17` | 26 | D08 |
| `E01-A093` | intervention_outcomes | `intervention.score.window_and_partition` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A093` | `S17` | 26 | D08 |
| `E01-A094` | intervention_outcomes | `intervention.counterfactual_rng_and_tie_break` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A094` | `S17` | 26 | none |
| `E01-A095` | intervention_outcomes | `intervention.control_and_pairing` | critical | `CONFLICT_PRESERVED` | `CONFLICT::reuse_original\|paired_rerun\|independent_rerun` | `S17` | 26 | none |
| `E01-A096` | intervention_outcomes | `intervention.persistence.formula` | high | `PAPER_FIXED` | `paper-like persistence=sum_t Y_t; episode metrics retained as separate outcomes` | `S17` | 7 | none |
| `E01-A097` | intervention_outcomes | `intervention.probability.formula_and_scope` | critical | `CONFLICT_PRESERVED` | `CONFLICT::overall_occupancy\|per_generation_trajectory\|final_generation_contrast` | `S17` | 10 | D06 |
| `E01-A098` | intervention_outcomes | `intervention.consistency.formula` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::C_Y\|C_H\|C_Z` | `S17` | 5 | D05 |
| `E01-A099` | intervention_outcomes | `intervention.consistency.reported_direction` | critical | `CONFLICT_PRESERVED` | `CONFLICT::results_min_worse\|table_min_0.42_gt_control_0.38` | `S17` | 4 | D05 |
| `E01-A100` | intervention_outcomes | `intervention.time_to_first.unit` | critical | `CONFLICT_PRESERVED` | `CONFLICT::table_percent_cells\|note_molecular_steps` | `S17` | 5 | D04 |
| `E01-A101` | intervention_outcomes | `intervention.time_to_first.censoring_and_equivalence` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A101` | `S17` | 5 | D04 |
| `E01-A102` | intervention_outcomes | `intervention.table1.center_and_dispersion` | critical | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A102` | `S17` | 22 | D10 |
| `E01-A103` | intervention_outcomes | `intervention.pairwise_inference` | critical | `FROZEN_BRANCH_SET` | `BRANCH_SET::paper_mann_whitney\|paired_inference\|multiplicity_corrected` | `S17` | 14 | D10 |
| `E01-A104` | intervention_outcomes | `intervention.generation_trend.coefficient_identity` | critical | `CONFLICT_PRESERVED` | `CONFLICT::regression_slope\|correlation_coefficient` | `S17` | 3 | D09 |
| `E01-A105` | intervention_outcomes | `intervention.generation_trend.aggregation_model` | high | `UNRESOLVED_REQUIRED` | `UNRESOLVED::E01-A105` | `S17` | 4 | D09 |

The companion CSV is authoritative for source evidence, ambiguity descriptions, admissible branches, resolution bases, claim IDs, risks, and validation rules.
