from plastic_heredity.memory_models import (
    duration_bin,
    fit_iid_probability,
    fit_legacy_mismatched_iid_probability,
    fit_memory_models,
    model_probabilities,
    transition_arrays,
)


def test_iid_fit_uses_only_scored_transition_destinations():
    sequences = [(True, False, True), (False, True), (True,)]
    fitted = fit_memory_models(sequences, include_legacy_iid=True)
    # Destinations are False, True, True: Beta(1,1) posterior mean = 3/5.
    assert fitted.iid_probability == 3 / 5
    assert fitted.iid_trials == 3
    assert fitted.markov_trials.sum() == 3
    assert fitted.semimarkov_trials.sum() == 3
    # Legacy behavior includes all six symbols, including the singleton.
    assert fitted.legacy_iid_probability == 5 / 8


def test_singletons_and_unscored_first_symbols_cannot_change_corrected_iid_fit():
    baseline = [(False, True, False), (True, False, True)]
    changed = [(True, True, False), (False, False, True), (True,), (True,), (False,)]
    assert fit_iid_probability(baseline) == fit_iid_probability(changed)
    assert fit_legacy_mismatched_iid_probability(baseline) != (
        fit_legacy_mismatched_iid_probability(changed)
    )


def test_duration_bins_are_past_only_and_cap_at_five_plus():
    sequence = (True, True, True, True, True, True, False)
    assert [duration_bin(sequence, index) for index in range(1, 7)] == [1, 2, 3, 4, 5, 5]
    previous, destination, durations = transition_arrays([sequence])
    assert previous.tolist() == [True] * 6
    assert destination.tolist() == [True, True, True, True, True, False]
    assert durations.tolist() == [1, 2, 3, 4, 5, 5]


def test_beta_smoothing_handles_empty_and_one_class_cells():
    fitted = fit_memory_models([(True, True, True), (True,)])
    assert fitted.iid_probability == 3 / 4
    assert fitted.markov_probabilities[0] == 0.5
    assert fitted.semimarkov_probabilities[0, 4] == 0.5
    previous, destination, durations = transition_arrays([(True, True, True)])
    probabilities = model_probabilities(fitted, previous, durations)
    assert set(probabilities) == {"iid", "markov", "semimarkov"}
    assert all(values.shape == destination.shape for values in probabilities.values())
