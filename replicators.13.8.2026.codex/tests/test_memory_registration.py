from plastic_heredity.config import ExperimentConfig
from plastic_heredity.mechanistic import MECHCONF_MASTER_SEED
from plastic_heredity.memory import (
    MEMORY_BRANCHES,
    MEMORY_CONFIRMATION_HORIZON,
    MEMORY_CONFIRM_MASTER_SEED,
    MEMORY_MATRICES,
    _memory_confirmation_experiment,
    post_break_sequence,
)


def test_post_break_sequence_excludes_break_and_distinguishes_no_break():
    assert post_break_sequence((True, True, True)) == (-1, ())
    assert post_break_sequence((True, False)) == (1, ())
    assert post_break_sequence((False, True, False)) == (0, (True, False))


def test_memory_confirmation_scale_and_seed_are_frozen_and_disjoint():
    experiment = _memory_confirmation_experiment()
    assert experiment.confirmation.matrices == MEMORY_MATRICES == 200
    assert experiment.confirmation.branches_per_state == MEMORY_BRANCHES == 64
    assert experiment.horizon == MEMORY_CONFIRMATION_HORIZON == 32
    assert MEMORY_CONFIRM_MASTER_SEED != ExperimentConfig.scaled5().master_seed
    assert MEMORY_CONFIRM_MASTER_SEED != MECHCONF_MASTER_SEED
