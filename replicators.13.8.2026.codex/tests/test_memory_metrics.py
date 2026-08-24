import numpy as np

from plastic_heredity.memory_metrics import (
    SequenceRecord,
    calibration_rows,
    compute_memory_metrics,
    crossfit_memory_models,
    sequence_count_rows,
)


def _records_from_generator(generator, *, candidates=("02",), matrices=80, branches=20):
    records = []
    state_index = 0
    for candidate in candidates:
        for matrix_id in range(matrices):
            for branch in range(branches):
                symbols = tuple(bool(value) for value in generator(matrix_id, branch))
                records.append(
                    SequenceRecord(
                        state_index=state_index,
                        state_id=f"s-{candidate}-{matrix_id}-{branch}",
                        candidate=candidate,
                        matrix_id=matrix_id,
                        landmark=20,
                        branch=branch,
                        symbols=symbols,
                        completed_horizon=True,
                        observed_fissions=len(symbols) + 1,
                        first_break_index=0,
                    )
                )
                state_index += 1
    return records


def test_variable_length_iid_sequences_have_approximately_zero_markov_gain():
    gains = []
    for repetition in range(10):
        rng = np.random.default_rng(700 + repetition)
        stored = {}
        for matrix_id in range(40):
            for branch in range(12):
                length = int(rng.integers(1, 25))
                stored[(matrix_id, branch)] = rng.binomial(1, 0.68, size=length)
        records = _records_from_generator(
            lambda matrix_id, branch: stored[(matrix_id, branch)],
            matrices=40,
            branches=12,
        )
        scores, _ = crossfit_memory_models(records)
        gains.append(
            float(
                np.mean(
                    scores["02"].losses["iid"]
                    - scores["02"].losses["markov"]
                )
            )
        )
    assert abs(float(np.mean(gains))) < 0.001
    assert sum(gain > 0.0 for gain in gains) <= 5


def test_planted_first_order_process_favors_markov_without_duration_gain():
    rng = np.random.default_rng(701)
    stored = {}
    for matrix_id in range(80):
        for branch in range(24):
            values = [bool(rng.integers(0, 2))]
            for _ in range(29):
                probability = 0.86 if values[-1] else 0.14
                values.append(bool(rng.random() < probability))
            stored[(matrix_id, branch)] = values
    records = _records_from_generator(
        lambda matrix_id, branch: stored[(matrix_id, branch)], branches=24
    )
    scores, _ = crossfit_memory_models(records)
    markov_gain = np.mean(
        scores["02"].losses["iid"] - scores["02"].losses["markov"]
    )
    duration_gain = np.mean(
        scores["02"].losses["markov"] - scores["02"].losses["semimarkov"]
    )
    assert markov_gain > 0.25
    assert duration_gain < 0.003


def test_planted_duration_process_favors_semimarkov_beyond_markov():
    rng = np.random.default_rng(702)
    stored = {}
    for matrix_id in range(80):
        for branch in range(24):
            values = [bool(rng.integers(0, 2))]
            run = 1
            for _ in range(39):
                stay_probability = 0.94 if run < 3 else 0.10
                stay = bool(rng.random() < stay_probability)
                values.append(values[-1] if stay else not values[-1])
                run = run + 1 if stay else 1
            stored[(matrix_id, branch)] = values
    records = _records_from_generator(
        lambda matrix_id, branch: stored[(matrix_id, branch)], branches=24
    )
    scores, _ = crossfit_memory_models(records)
    duration_gain = np.mean(
        scores["02"].losses["markov"] - scores["02"].losses["semimarkov"]
    )
    assert duration_gain > 0.20


def test_metric_family_has_four_registered_tests_and_common_support():
    rng = np.random.default_rng(703)
    stored = {}
    for candidate in ("02", "03"):
        for matrix_id in range(20):
            for branch in range(8):
                stored[(candidate, matrix_id, branch)] = rng.binomial(1, 0.6, size=12)
    records = []
    state_index = 0
    for candidate in ("02", "03"):
        for matrix_id in range(20):
            for branch in range(8):
                records.append(
                    SequenceRecord(
                        state_index=state_index,
                        state_id=f"s-{candidate}-{matrix_id}-{branch}",
                        candidate=candidate,
                        matrix_id=matrix_id,
                        landmark=20,
                        branch=branch,
                        symbols=tuple(stored[(candidate, matrix_id, branch)].astype(bool)),
                        completed_horizon=True,
                        observed_fissions=13,
                        first_break_index=0,
                    )
                )
                state_index += 1
    scores, _ = crossfit_memory_models(records)
    for score in scores.values():
        assert {array.size for array in score.losses.values()} == {
            score.destination.size
        }
    metrics = compute_memory_metrics(
        scores, repetitions=64, master_seed="metric-family", confirmatory=True
    )
    assert metrics["family_size"] == 4
    assert len(metrics["primary_tests"]) == 4
    assert set(metrics["support"]) == {
        "markov_vs_iid",
        "semimarkov_vs_markov",
    }
    assert all("randomization_p_holm" in row for row in metrics["primary_tests"])
    calibration = calibration_rows(scores)
    assert calibration
    assert {row["model"] for row in calibration} == {
        "iid",
        "markov",
        "semimarkov",
    }
    assert all(0.0 < row["fitted_probability"] < 1.0 for row in calibration)


def test_sequence_counts_keep_no_break_empty_singleton_and_extinction_distinct():
    records = [
        SequenceRecord(0, "a", "02", 0, 20, 0, (), True, 32, -1),
        SequenceRecord(0, "a", "02", 0, 20, 1, (), True, 32, 31),
        SequenceRecord(0, "a", "02", 0, 20, 2, (True,), True, 32, 30),
        SequenceRecord(1, "b", "02", 1, 20, 0, (True, False), False, 4, 1),
    ]
    overall = next(
        row for row in sequence_count_rows(records) if row["matrix_fold"] == "all"
    )
    assert overall["no_break"] == 1
    assert overall["empty_post_break_suffix"] == 1
    assert overall["singleton_post_break_suffix"] == 1
    assert overall["usable_suffixes"] == 1
    assert overall["scored_transitions"] == 1
    assert overall["extinctions"] == 1
